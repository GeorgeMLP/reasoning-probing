from datasets import Dataset
from transformers import AutoTokenizer
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
import json
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer
from jaxtyping import Int
from typing import Any


qwen_chat_template = """<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""


class ReasoningChainDataset(Dataset):
    def __init__(
        self,
        reasoning_chains: dict[str, dict[str, Any]],
        *,
        tokenizer_name: str = "Qwen/Qwen2.5-14B",
        tokenizer_chat_template: str = qwen_chat_template,
        max_length: int = 512,
        include_question: bool = True,
        discard_tokens_after_max_length: bool = False,
    ) -> None:
        self.tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        self.max_length = max_length
        self.chunks: list[dict[str, Any]] = []
        self.chat_template = tokenizer_chat_template
        self.include_question = include_question
        self.one_batch_per_chain = discard_tokens_after_max_length
        self._prepare(reasoning_chains)

    def _prepare(self, reasoning_chains: dict[str, dict[str, Any]]) -> None:
        for chain_id_str, entry in reasoning_chains.items():
            chain_id: int = int(chain_id_str)
            question: str = entry['question']
            parts_gemini: list[tuple[str, str]] = \
                entry['annotated_gemini_thinking_trajectory']
            parts_deepseek: list[tuple[str, str]] = \
                entry['annotated_deepseek_thinking_trajectory']
            
            # Here I use `chain_id * 2` and `chain_id * 2 + 1` as the IDs for
            # the trajectories because there are 2 trajectories in each data
            # entry, produced by Gemini and DeepSeek.
            proc_gemini = self._process_chain(
                chain_id * 2, question, parts_gemini
            )
            proc_deepseek = self._process_chain(
                chain_id * 2 + 1, question, parts_deepseek
            )

            chunks_gemini = self._chunk_chain(proc_gemini)
            chunks_deepseek = self._chunk_chain(proc_deepseek)
            if chunks_gemini is not None:
                for chunk in chunks_gemini:
                    self.chunks.append(chunk)
            if chunks_deepseek is not None:
                for chunk in chunks_deepseek:
                    self.chunks.append(chunk)

    def _process_chain(
        self,
        chain_id: int,
        question: str,
        parts: list[tuple[str, str]]
    ) -> dict[str, Any]:
        all_input_ids: list[int] = []
        part_infos: list[dict[str, str | int]] = []
        cursor: int = 0
        for label, text in parts:
            if len(text) == 0:
                continue
            if text[-1] not in [' ', '\n']:  # add missing spaces
                text += ' '
            toks: list[int] = self.tokenizer(
                text, add_special_tokens=False
            ).input_ids
            start = cursor
            end = cursor + len(toks) - 1
            assert start <= end
            all_input_ids.extend(toks)
            part_infos.append({'label': label, 'start': start, 'end': end})
            cursor = end + 1
        
        return {
            'chain_id': chain_id,
            'question': question,
            'input_ids': all_input_ids,
            'part_infos': part_infos,
        }

    def _chunk_chain(self, proc: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Returns None if question is too long."""
        input_ids: list[int] = proc['input_ids']
        parts: list[dict[str, str | int]] = proc['part_infos']
        question: str = proc['question'] if self.include_question else ''
        cid: int = proc['chain_id']

        prefix = self.chat_template.format(question=question)
        prefix_ids: list[int] = self.tokenizer(prefix).input_ids
        assistant_offset = len(prefix_ids)
        max_length = self.max_length - assistant_offset
        if max_length < self.max_length // 2:
            return None

        chunks: list[dict[str, Any]] = []
        for chunk_start in range(0, len(input_ids), max_length):
            chunk_end = min(chunk_start + max_length, len(input_ids))
            span_ids = input_ids[chunk_start:chunk_end]

            chunk_parts: list[dict[str, str | int]] = []
            for p in parts:
                if p['end'] < chunk_start or p['start'] >= chunk_end:
                    continue
                rel_s = max(0, p['start'] - chunk_start)
                rel_e = min(chunk_end - chunk_start - 1, p['end'] - chunk_start)
                assert rel_s <= rel_e
                chunk_parts.append({
                    'label': p['label'],
                    'start': rel_s,
                    'end': rel_e,
                })

            full_toks = prefix_ids + span_ids

            # Shift each part start/end by assistant_offset.
            adjusted: list[dict[str, str | int]] = []
            for p in chunk_parts:
                adjusted.append({
                    'label': p['label'],
                    'start': assistant_offset + p['start'],
                    'end': assistant_offset + p['end'],
                })

            chunks.append({
                'chain_id': cid,
                'input_ids': full_toks,
                'part_infos': adjusted,
            })
            if self.one_batch_per_chain:  # discard the other chunks
                break
        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.chunks[i]


class ReasoningChainDataLoader:
    def __init__(
        self,
        reasoning_chains: dict[str, dict[str, Any]],
        *,
        tokenizer_name: str = 'Qwen/Qwen2.5-14B',
        tokenizer_chat_template: str = qwen_chat_template,
        max_length: int = 512,
        batch_size: int = 4,
        shuffle: bool = True,
        include_question: bool = True,
        discard_tokens_after_max_length: bool = False,
    ):
        self.dataset = ReasoningChainDataset(
            reasoning_chains,
            tokenizer_name=tokenizer_name,
            tokenizer_chat_template=tokenizer_chat_template,
            max_length=max_length,
            include_question=include_question,
            discard_tokens_after_max_length=discard_tokens_after_max_length,
        )
        self.tokenizer = self.dataset.tokenizer
        self.batch_size = batch_size
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        ids = [torch.tensor(x['input_ids']) for x in batch]
        pad_token_id: int = self.tokenizer.pad_token_id
        masks = [(i != pad_token_id).long() for i in ids]
        
        ids_padded: Int[Tensor, "b l"] = torch.nn.utils.rnn.pad_sequence(
            ids, batch_first=True, padding_value=pad_token_id
        )
        masks_padded: Int[Tensor, "b l"] = torch.nn.utils.rnn.pad_sequence(
            masks, batch_first=True, padding_value=0
        )

        return {
            'input_ids': ids_padded,
            'attention_mask': masks_padded,
            'chain_ids': [x['chain_id'] for x in batch],
            'part_infos': [x['part_infos'] for x in batch],
        }

    def get_dataloader(self) -> DataLoader:
        return self.loader


if __name__ == '__main__':
    with open('reasoning_dataset/annotated_dataset.json', 'r') as f:
        reasoning_chains: dict[str, dict[str, Any]] = json.load(f)
    
    model_name = 'Qwen/Qwen2.5-14B'
    device = 'cuda'
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        dtype=torch.float16,
        default_padding_side='right',
    )

    wrapper = ReasoningChainDataLoader(
        reasoning_chains,
        tokenizer_name=model_name,
        tokenizer_chat_template=qwen_chat_template,
        max_length=150,
        batch_size=2,
        shuffle=True,
        include_question=True,
        discard_tokens_after_max_length=True,
    )
    dataloader = wrapper.get_dataloader()

    for batch_id, batch in enumerate(dataloader):
        input_ids: Int[Tensor, "b l"] = batch['input_ids']
        chain_ids: list[int] = batch['chain_ids']
        part_infos: list[list[dict[str, str | int]]] = batch['part_infos']
        for input_id, chain_id, part_info in zip(
            input_ids, chain_ids, part_infos, strict=True
        ):
            print('=' * 20, f'Batch {batch_id}, chain {chain_id}', '=' * 20)
            for part in part_info:
                label: str = part['label']
                start: int = part['start']
                end: int = part['end']
                print(f'[{label}] {start}--{end}:')
                print(wrapper.tokenizer.decode(input_id[start:(end + 1)]))
                print()
        generated_ids = model.generate(input_ids.to(device), max_new_tokens=128)
        print(model.tokenizer.decode(generated_ids[0]))
        break
