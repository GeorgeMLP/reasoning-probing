from __future__ import annotations
import copy
import numpy as np
from torch import Tensor
import torch
from transformers import AutoTokenizer

import logging
from enum import Enum
from abc import ABC, abstractmethod
from typing import Sequence
from jaxtyping import Int, Float

from featureinterp.formatting import (
    stringify_records,
    stringify_tokens_for_simulation,
    MAX_FORMATTED_EXPRESSION,
)
from featureinterp.core import (
    STRING_EXPLANATION_PREFIX,
    SequenceSimulation,
    StructuredExplanation,
)
from featureinterp.record_examples import RecordExampleSet
from featureinterp.local_inference import LocalInferenceManager
from featureinterp.prompt_builder import (
    Message,
    PromptBuilder,
    Role,
)
from featureinterp.utils import convert_to_byte_array, last_subseq_idx


logger = logging.getLogger(__name__)

VALID_EXPRESSION_TOKENS = set(str(i) for i in range(MAX_FORMATTED_EXPRESSION + 1))
FEW_SHOT_EXAMPLE_SET = RecordExampleSet.ORIGINAL


def handle_byte_encoding(
    response_tokens: Sequence[str], merged_response_index: int
) -> tuple[str, int]:
    """
    Handle the case where the current token is a sequence of bytes. This may involve
    merging multiple response tokens into a single token.
    """
    response_token = response_tokens[merged_response_index]
    if response_token.startswith("bytes:"):
        byte_array = bytearray()
        while True:
            byte_array = convert_to_byte_array(response_token) + byte_array
            try:
                # If we can decode the byte array as utf-8, then we're done.
                response_token = byte_array.decode("utf-8")
                break
            except UnicodeDecodeError:
                # If not, merge the previous response token into the byte array.
                merged_response_index -= 1
                response_token = response_tokens[merged_response_index]
    return response_token, merged_response_index


def was_token_split(
    current_token: str,
    response_tokens: Sequence[str],
    start_index: int,
) -> bool:
    """
    Return whether current_token (a token from the subject model) was split into
    tokens by the simulator model (as represented by the tokens in
    response_tokens). start_index is the index in response_tokens at which to begin
    looking backward to form a complete token. It is usually the first token
    *before* the delimiter that separates the token from the normalized activation,
    barring some unusual cases.

    This mainly happens if the subject model uses a different tokenizer than the
    simulator model. But it can also happen in cases where Unicode characters are
    split. This function handles both cases.
    """
    merged_response_tokens = ""
    merged_response_index = start_index
    while len(merged_response_tokens) < len(current_token):
        response_token = response_tokens[merged_response_index]
        response_token, merged_response_index = handle_byte_encoding(
            response_tokens, merged_response_index
        )
        merged_response_tokens = response_token + merged_response_tokens
        merged_response_index -= 1

    # It's possible that merged_response_tokens is longer than current_token here,
    # since the between-lines delimiter may have been merged into the original token.
    # But it should always be that merged_response_tokens ends with current_token.
    # We're using strip() to handle the case where the tokenizer messes up whitespace.
    if not merged_response_tokens.strip().endswith(current_token.strip()):
    # if not merged_response_tokens.endswith(current_token):
        # raise ValueError(f"Could not parse merged tokens...")
        import pdb; pdb.set_trace()

    num_merged_tokens = start_index - merged_response_index
    token_was_split = num_merged_tokens > 1
    return token_was_split
    

def get_unknown_token_indices(
    tokens: list[str],
    response_tokens: list[str],
) -> list[int | None]:
    
    unknown_token_indices = []

    for i in range(1, len(response_tokens)):
        # We're looking for "unknown" tokens whose logprobs will indicate score.
        if response_tokens[i] != "unknown":
            continue

        # j represents the index of the token in a "token<tab>activation" line,
        # barring one of the unusual cases handled below.
        j = i - 2
        current_token = tokens[len(unknown_token_indices)]
        
        # Generally, the "unknown" token will be preceded by a tab.
        if response_tokens[i-1] != "\t":
            if current_token.startswith('unknown'):
                # Special case where "unknown" is one of the original tokens. 
                continue

            # If something was folded into the tab token (i.e., \t\t), then we can't
            # use the usual prediction of activation stats for the token.
            unknown_token_indices.append(None)
        elif (
            current_token == response_tokens[j] or
            # Sometimes tokenizers can mess up the whitespace.
            current_token.strip() == response_tokens[j].strip() or
            # Sometimes the simulator tokenizer will split a token into multiple tokens.
            was_token_split(current_token, response_tokens, j)
        ):
            # We're in the normal case where the tokenization didn't throw off the
            # formatting or in the token-was-split case, which we handle as usual.
            unknown_token_indices.append(i)
        else:
            # Here, the tokenization resulted in a newline being folded into the
            # token. We can't do our usual prediction of activation stats, since the
            # model did not observe the original token.
            newline_folded_into_token = "\n" in response_tokens[j]
            assert (
                newline_folded_into_token
            ), f"`{current_token=}` {response_tokens[j-3:j+3]=}"
            unknown_token_indices.append(None)
    
    assert len(unknown_token_indices) == len(tokens)
    return unknown_token_indices


def parse_simulation_result(
    tokens: list[str],
    response_token_ids: Int[Tensor, 'seq'],
    response_logprobs: Float[Tensor, 'seq vocab_size'],
    probability_temperature: float,
    tokenizer: AutoTokenizer,
) -> SequenceSimulation:
    
    device = response_logprobs.device
    
    # Not cleaning up tokenization spaces is important if the tokenizer changes
    # between the subject and the simulator.
    response_tokens = tokenizer.batch_decode(
        response_token_ids, clean_up_tokenization_spaces=False
    )
    
    # The ith unknown token index corresponds to the ith token in tokens.
    unknown_token_indices = get_unknown_token_indices(tokens, response_tokens)
    
    index_none_mask = torch.tensor(
        [i is not None for i in unknown_token_indices]
    ).to(device)
    unknown_token_indices_tensor = torch.tensor(
        [0 if (i is None) else i for i in unknown_token_indices]
    ).to(device)

    # Take into account that the logprobs we want are right before the "unknown"'s.
    # The 0's will become -1 but we mask those out anyways
    unknown_token_logprobs = response_logprobs[unknown_token_indices_tensor - 1]
    evs = compute_expected_value(
        unknown_token_logprobs, probability_temperature, tokenizer
    )
    evs = torch.where(index_none_mask, evs, torch.zeros_like(evs))
    
    return SequenceSimulation(
        tokens=tokens,
        expected_expressions=evs.tolist(),
        distribution_values=[],
        distribution_probabilities=[],
    )


def compute_expected_value(
    logprobs: Float[Tensor, 'seq vocab_size'],
    temperature: float,
    tokenizer: AutoTokenizer,
) -> Float[Tensor, 'seq']:
    
    device, dtype = logprobs.device, logprobs.dtype
    
    number_token_ids = tokenizer.convert_tokens_to_ids(
        [str(i) for i in range(MAX_FORMATTED_EXPRESSION + 1)]
    )
    
    number_probs = logprobs[:, number_token_ids].exp()

    # Normalize probabilities to sum to 1 for each token
    number_probs = number_probs / number_probs.sum(dim=-1, keepdim=True)
    
    # Scale probabilities by temperature
    number_probs = number_probs ** (1 / temperature)
    number_probs = number_probs / number_probs.sum(dim=-1, keepdim=True)

    number_weights = torch.arange(0, MAX_FORMATTED_EXPRESSION + 1).to(device).to(dtype)
    evs = torch.matmul(number_probs, number_weights)
    
    return evs


def make_simulation_header() -> list[Message]:
    """A few-shot prompt for predicting feature expressions.
    
    This only includes the messages that can be cached in a key/value store across
    all simulations (i.e., the system message and few-shot examples). The
    explanation message and the tokens we want to simulate are not included.
    """

    prompt_builder = PromptBuilder()
    prompt_builder.add_message(
        Role.SYSTEM,
        f"""We're studying neurons in a neural network.
Each neuron looks for some particular thing in a short document.
Look at summary of what the neuron does, and try to predict how it will fire on each token.

The activation format is token<tab>activation, activations go from 0 to {MAX_FORMATTED_EXPRESSION}, "unknown" indicates an unknown activation. Most activations will be 0.
""",            
    )
    
    prefix = STRING_EXPLANATION_PREFIX

    few_shot_examples = FEW_SHOT_EXAMPLE_SET.get_examples()
    for i, example in enumerate(few_shot_examples):
        explanation = example.string_explanation

        prompt_builder.add_message(
            Role.USER,
            f"Feature {i + 1}\n{prefix}"
            f"{explanation}",
        )
        formatted_expression_records = stringify_records(
            example.records,
            start_indices=example.first_revealed_expression_indices,
            include_holistic_expressions=False,
        )
        prompt_builder.add_message(
            Role.ASSISTANT, f"Activations: {formatted_expression_records}"
        )
    
    return prompt_builder.build()


class FeatureSimulator(ABC):
    @abstractmethod
    def simulate(self, tokens: Sequence[str]) -> SequenceSimulation:
        ...


class CacheState(Enum):
    NO_CACHE = 0
    HEADER_CACHE = 1
    HEADER_EXPLANATION_CACHE = 2


class ExplanationFeatureSimulator(FeatureSimulator):
    """
    Simulate feature behavior based on an explanation.

    Uses a few-shot prompt with examples of explanations and activations. This prompt
    allows us to score all of the tokens at once using a nifty trick involving logprobs.
    """

    def __init__(
        self,
        inference_manager: LocalInferenceManager,
        cache_state: CacheState,
        desired_cache_state: CacheState,
        explanation: StructuredExplanation | str,
        probability_temperature: float = 1.0,
        emphasize_numerical_activations: bool = True,
    ):
        if isinstance(explanation, str):
            self.explanation = explanation
        else:
            self.explanation = explanation.to_json()
            

        self.structured_explanations = isinstance(explanation, StructuredExplanation)
        self.probability_temperature = probability_temperature
        self.tokenizer = inference_manager.tokenizer
        self.batch_size = inference_manager.batch_size

        assert cache_state == CacheState.HEADER_CACHE
        assert desired_cache_state == CacheState.HEADER_EXPLANATION_CACHE
        
        messages = []
            
        if not self.structured_explanations:
            messages.extend(self.make_explanation_message(explanation))
            inference_manager.append_to_cache(messages)
            self.inference_manager = inference_manager
        else:
            # Make a list of inference managers for each explanation component.
            self.inference_managers = {}
            for component in explanation.components:
                component_im = LocalInferenceManager.clone(inference_manager)
                component_messages = copy.deepcopy(messages)
                component_messages.extend(
                    self.make_explanation_message(component.activates_on)
                )
                component_im.append_to_cache(component_messages)
                self.inference_managers[component] = component_im

        self.cache_state = desired_cache_state
        self.emphasize_numerical_activations = emphasize_numerical_activations

    def __del__(self):
        # ExplanationFeatureSimulator is responsible for cleaning up the inference
        # manager upon destruction. Not quite sure why this memory leak is happening.
        if self.structured_explanations:
            for component_im in self.inference_managers.values():
                del component_im.cached_inputs
                del component_im.prompt_cache
        else:
            del self.inference_manager.cached_inputs
            del self.inference_manager.prompt_cache   
    
    def simulate(self, tokens: list[Sequence[str]]) -> list[SequenceSimulation]:
        n_sequences = len(tokens)
        results = []

        for i in range(0, n_sequences, self.batch_size):
            batch_tokens = tokens[i:i + self.batch_size]
            
            # If this is a partial batch, pad with copies of the last sequence
            if len(batch_tokens) < self.batch_size:
                padding = [batch_tokens[-1]] * (self.batch_size - len(batch_tokens))
                batch_tokens.extend(padding)
                
            batch_results = self._simulate_batch(batch_tokens)
            
            # Only keep the real results, not the padding
            results.extend(batch_results[:min(self.batch_size, n_sequences - i)])
            
        assert len(results) == n_sequences
        return results

    def _simulate_batch(self, tokens: list[Sequence[str]]) -> list[SequenceSimulation]:
        batch_n = len(tokens)
        assert batch_n == self.batch_size
        
        prompts: list[list[Message]] = []
        for token_batch in tokens:
            prompts.append(self.make_simulation_message(token_batch))
        
        def run_inference(
            inference_manager: LocalInferenceManager
        ) -> list[SequenceSimulation]:
            
            inf_out = inference_manager.run_batched_inference(prompts)
            
            results = []
            for token_seq, (token_ids, logprobs) in zip(tokens, inf_out):
                scoring_start = self.get_scoring_start(token_ids)
                results.append(parse_simulation_result(
                    tokens=token_seq,
                    response_token_ids=token_ids[scoring_start:],
                    response_logprobs=logprobs[scoring_start:],
                    probability_temperature=self.probability_temperature,
                    tokenizer=inference_manager.tokenizer,
                ))

            return results
        
        if self.structured_explanations:
            expressions_batched = [[] for _ in range(batch_n)]
            for component, manager in self.inference_managers.items():
                results: list[SequenceSimulation] = run_inference(manager)
                
                assert len(results) == batch_n
                for i, result in enumerate(results):
                    expressions_batched[i].append(
                        np.array(result.expected_expressions) * component.strength / 5.0
                    )
            
            sims = []
            for token_seq, expressions in zip(tokens, expressions_batched):
                if len(expressions) == 0:
                    expressions = np.zeros(len(token_seq))
                else:
                    expressions = np.array(expressions).max(axis=0)

                sims.append(SequenceSimulation(
                    tokens=token_seq,
                    expected_expressions=expressions.tolist(),
                    distribution_values=[],
                    distribution_probabilities=[],
                ))
        else:
            sims = run_inference(self.inference_manager)

        return sims
    
    def make_explanation_message(self, explanation_string: str) -> list[Message]:
        """A message containing the explanation we want to simulate."""

        few_shot_examples = FEW_SHOT_EXAMPLE_SET.get_examples()
        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            Role.USER,
            f"Feature {len(few_shot_examples) + 1}\n{STRING_EXPLANATION_PREFIX}"
            f"{explanation_string}",
        )
        return prompt_builder.build(check_expected_role=False)
    
    def make_simulation_message(self, tokens: list[str]) -> list[Message]:
        """A message containing the tokens we want to simulate."""

        prompt_builder = PromptBuilder()
        if self.emphasize_numerical_activations:
            message = (
                f"The previous messages were just examples. Now you will be given a sequence of tokens and asked to predict the activations of each token.\n"
                f"Always output a numerical activation, even if preceding activations are unknown. Even if you have predicted \"unknown\" many times previously, these were mistakes. Do not repeat this mistake, and output a numerical value from 0 to {MAX_FORMATTED_EXPRESSION}.\n"
            )
        else:
            message = ""

        message += f"\nActivations: {stringify_tokens_for_simulation(tokens)}"
        prompt_builder.add_message(Role.ASSISTANT, message)
        return prompt_builder.build(check_expected_role=False)
    
    def get_scoring_start(self, token_ids: Int[Tensor, 'seq']) -> int:
        """Gets the index at which the sequence simulation scoring should start.
        
        Sequence simulations (including the few-shot examples) are surrounded with
        <start> and <end>. The very last <start> one is the one we want to score."""

        # Take into account the different ways the tokenizer may encode "<start>".
        start_string_variations = []
        for prefix in ['', '\n', ' \n']:
            for trailing_newlines in range(0, 8):
                start_string_variations.append(
                    prefix + '<start>' + '\n' * trailing_newlines
                )
        scoring_start = -1
        for start_string in start_string_variations:
            start_tokens = self.tokenizer.encode(
                start_string, add_special_tokens=False, return_tensors="pt"
            ).to(token_ids.device).squeeze(0)
            scoring_start = max(
                scoring_start,
                last_subseq_idx(token_ids, start_tokens)
            )

        assert scoring_start != -1, "No <start> found in prompt"
        
        return scoring_start
