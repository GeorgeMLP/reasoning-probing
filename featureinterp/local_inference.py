from __future__ import annotations
import copy
from einops import pack, rearrange, repeat
from torch import Tensor
import torch
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import HybridCache, DynamicCache, Cache

from jaxtyping import Int, Float

from featureinterp.prompt_builder import (
    Message,
    Role,
)


class LocalInferenceManager:
    """Handles local LLM inference, including K/V caching."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        batch_size: int,
        max_tokens_after_cache: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = self.model.name_or_path
        self.batch_size = batch_size
        self.max_tokens_after_cache = max_tokens_after_cache

        self.using_kv_cache = False
        self.cached_inputs: Int[Tensor, 'batch seq'] | None = None
        self.prompt_cache: Cache | None = None

    @classmethod
    def clone(cls, other: LocalInferenceManager) -> LocalInferenceManager:
        """Clones the cache of another inference manager.
        
        We can then append further messages to the cache."""

        cloned = cls(
            tokenizer=other.tokenizer,
            model=other.model,
            batch_size=other.batch_size,
            max_tokens_after_cache=other.max_tokens_after_cache,
        )
        cloned.using_kv_cache = other.using_kv_cache
        cloned.cached_inputs = copy.deepcopy(other.cached_inputs)
        cloned.prompt_cache = copy.deepcopy(other.prompt_cache)
        return cloned

    def run_batched_inference(
        self, messages_list: list[list[Message]]
    ) -> list[tuple[Int[Tensor, 'seq'], Float[Tensor, 'seq vocab_size']]]:
        """Returns the tokens and logprobs from inference over a batch of prompts.
        
        If KV caching is enabled, the messages are effectively appended to the cached
        prompt. However, tokens and logprobs are returned only for the messages
        passed to run_inference.
        """
        
        batch_n = len(messages_list)
        
        while len(messages_list) < self.batch_size:
            messages_list.append(copy.deepcopy(messages_list[0]))
        
        messages_list = copy.deepcopy(messages_list)
        messages_list = [self.get_prompt_for_model(m) for m in messages_list]
        
        template = self.get_suffix_template_for_model() if self.using_kv_cache else None
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        input_dict = self.tokenizer.apply_chat_template(
            messages_list,
            chat_template=template,
            return_tensors='pt',
            padding=True,
            tokenizer_kwargs={'padding_side': 'right'},
            return_dict=True,
        ).to(self.model.device)
        
        with torch.no_grad():
            if self.using_kv_cache:
                input_dict = self.postprocess_suffix_input_dict_for_model(input_dict)
                cache = copy.deepcopy(self.prompt_cache)
                output = self.model(**input_dict, past_key_values=cache, use_cache=True)
            else:
                output = self.model(**input_dict)
        
        logprobs = torch.nn.functional.softmax(output.logits, dim=-1).log()
        
        # unknown_id = self.tokenizer.convert_tokens_to_ids('unknown')
        # unknown_logprobs = logprobs[:, :, unknown_id]
        # print(unknown_logprobs[:, -100:].exp().sum())
        
        results = []
        for b in range(batch_n):
            seq_len = input_dict['attention_mask'][b].sum().item()
            results.append((
                input_dict['input_ids'][b, :seq_len],
                logprobs[b, :seq_len],
            ))
        return results
    
    def append_to_cache(self, messages: list[Message]) -> None:
        """Caches messages that will be prepended to run_inference messages."""
        
        if len(messages) == 0:
            return

        messages = copy.deepcopy(messages)
        prompt_head = self.get_prompt_for_model(messages)

        input_dict = self.tokenizer.apply_chat_template(
            prompt_head,
            return_tensors='pt',
            return_dict=True,
        ).to(self.model.device)
        
        if self.using_kv_cache:
            input_dict = self.postprocess_suffix_input_dict_for_model(input_dict)
        
        inputs = repeat(input_dict['input_ids'], '1 s -> b s', b=self.batch_size)

        if self.using_kv_cache:
            cache = self.prompt_cache
            self.cached_inputs = pack([self.cached_inputs, inputs], 'b *')[0]
        else:
            cache = self.get_cache_for_model(inputs)
            self.cached_inputs = inputs
        
        with torch.no_grad():
            outputs = self.model(inputs, use_cache=True, past_key_values=cache)
            del outputs.logits
            torch.cuda.empty_cache()

        self.prompt_cache = cache
        self.using_kv_cache = True
    
    def get_prompt_for_model(self, prompt: list[Message]) -> list[Message]:
        """Reformats the prompt to be appropriate for the model."""
        
        if 'gemma' in self.model_name:
            if prompt[0].role == Role.SYSTEM:
                assert prompt[1].role == Role.USER
                # Gemma doesn't have a system message, so we combine the first system
                # message with the first user message.
                prompt[1].content = prompt[0].content + '\n\n' + prompt[1].content
                prompt = prompt[1:]
        
            for message in prompt:
                # Rename the role to match the format expected by Gemma.
                id_map = { Role.USER: 'user', Role.ASSISTANT: 'model' }
                message.role = id_map[message.role]
        elif 'phi' in self.model_name:
            if prompt[0].role == Role.SYSTEM:
                prompt[1].content = prompt[0].content + '\n\n' + prompt[1].content
                prompt = prompt[1:]
        elif 'Distill' in self.model_name:
            if prompt[0].role == Role.SYSTEM:
                prompt[1].content = prompt[0].content + '\n\n' + prompt[1].content
                prompt[0].content = ''
        elif self.model_name == "mistralai/Ministral-8B-Instruct-2410":
            if prompt[0].role == Role.SYSTEM:
                prompt[1].content = prompt[0].content + '\n\n' + prompt[1].content
                prompt[0].content = ''
        
        return prompt
    
    def get_suffix_template_for_model(self):
        """A chat template for the suffix of a simulation prompt."""

        chat_jinja_template = self.tokenizer.chat_template
        if 'gemma' in self.model_name:
            # The suffix chat starts with an assistant message when we're doing kv
            # caching. So we need to remove the check for conversation roles.
            chat_jinja_template = chat_jinja_template.replace(
                "{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }",
                ""
            )
        elif self.model_name == "mistralai/Ministral-8B-Instruct-2410":
            chat_jinja_template = chat_jinja_template.replace(
                "{{- raise_exception(\"After the optional system message, conversation roles must alternate user/assistant/user/assistant/...\") }}",
                ""
            ) 
    
        return chat_jinja_template
    
    def postprocess_suffix_input_dict_for_model(
        self,
        input_dict: dict[str, Int[Tensor, 'batch seq']],
    ) -> dict[str, Int[Tensor, 'batch seq']]:
        """The suffix might be tokenized with a bunch of extra stuff added by the chat
        template. We want to remove this since we want the suffix tokens to be a
        continuation of the header tokens (which are cached in a key/value store)."""
        
        tokens: Int[Tensor, 'batch seq'] = input_dict['input_ids']
        attention_mask: Int[Tensor, 'batch seq'] = input_dict['attention_mask']
        
        if 'gemma' in self.model_name:
            # Remove the start token added by the chat template.
            tokens = tokens[:, 1:]
            attention_mask = attention_mask[:, 1:]
        elif 'llama' in self.model_name:
            raise RuntimeError("Attention mask postprocessing not implemented")
            # Need to get rid of the automatic system prompt header put in by the
            # chat template.
            tokens = rearrange(tokens, '1 s -> s')
            start_header_id = self.tokenizer.encode(
                "<|start_header_id|>", add_special_tokens=False
            )[0]
            header_indices = (tokens == start_header_id).nonzero().flatten()
            assert len(header_indices) == 2, "Expected 2 header indices"
            tokens = tokens[header_indices[1]:]
        elif 'Qwen' in self.model_name:
            raise RuntimeError("Attention mask postprocessing not implemented")
            tokens = rearrange(tokens, '1 s -> s')
            im_start_id = self.tokenizer.encode(
                "<|im_start|>", add_special_tokens=False
            )[0]
            header_indices = (tokens == im_start_id).nonzero().flatten()
            assert len(header_indices) == 2, "Expected 2 header indices"
            tokens = tokens[header_indices[1]:]
        elif 'phi' in self.model_name:
            import pdb; pdb.set_trace()
            
        return { 'input_ids': tokens, 'attention_mask': attention_mask }
    
    def get_cache_for_model(
        self, prompt_inputs: Int[Tensor, 'batch seq']
    ) -> Cache:
        """Construct the right key/value cache for the model."""
        
        if 'gemma' in self.model_name:
            return HybridCache(
                config=self.model.config,
                max_batch_size=prompt_inputs.shape[0],
                max_cache_len=prompt_inputs.shape[1] + self.max_tokens_after_cache,
                device=self.model.device,
                dtype=self.model.dtype
            )
        elif 'llama' in self.model_name:
            return DynamicCache()
