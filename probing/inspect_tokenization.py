"""
Script to inspect tokenization differences between normal and reasoning datasets.
"""

import torch
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from sae_lens import HookedSAETransformer
from datasets import load_dataset, Dataset as HFDataset
from transformer_lens.utils import tokenize_and_concatenate
from reasoning_dataset.reasoning_dataset import ReasoningChainDataset


def inspect_normal_dataset(model_name='google/gemma-2-2b', num_samples=5):
    """Inspect tokenization of normal (Pile) dataset."""
    print("=" * 80)
    print("NORMAL DATASET (Pile)")
    print("=" * 80)
    
    # Load model
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device='cpu',
        dtype=torch.bfloat16,
    )
    
    # Load dataset
    dataset = load_dataset(
        path='monology/pile-uncopyrighted',
        split='train',
        streaming=True,
    )
    
    # Filter and prepare
    dataset = dataset.filter(lambda x: len(x['text']) == len(x['text'].encode()))
    dataset = dataset.map(lambda x: {'text': x['text'][:127 * 4 * 2]})
    dataset = HFDataset.from_list(list(dataset.take(num_samples * 3)))
    
    # Tokenize
    token_dataset = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=model.tokenizer,
        max_length=128,
        add_bos_token=True,
        streaming=True,
    )['tokens']
    
    # Filter special tokens
    special_tokens = torch.tensor(model.tokenizer.all_special_ids)
    is_special_token = torch.isin(token_dataset, special_tokens)
    token_dataset = token_dataset[is_special_token.sum(dim=1) <= 1]
    token_dataset = token_dataset[:num_samples]
    
    print(f"\nShape: {token_dataset.shape}")
    print(f"Special token IDs: {model.tokenizer.all_special_ids}")
    print(f"BOS token: {model.tokenizer.bos_token} (ID: {model.tokenizer.bos_token_id})")
    print(f"EOS token: {model.tokenizer.eos_token} (ID: {model.tokenizer.eos_token_id})")
    print(f"PAD token: {model.tokenizer.pad_token} (ID: {model.tokenizer.pad_token_id})")
    
    # Show samples
    for i in range(min(3, len(token_dataset))):
        tokens = token_dataset[i]
        text = model.tokenizer.decode(tokens)
        print(f"\n--- Sample {i+1} ---")
        print(f"Tokens (first 20): {tokens[:20].tolist()}")
        print(f"Length: {len(tokens)}")
        print(f"Text (first 200 chars): {text[:200]!r}")
        print(f"Has BOS at start: {tokens[0].item() == model.tokenizer.bos_token_id}")
        print(f"Number of special tokens: {is_special_token[i].sum().item()}")
    
    return model


def inspect_reasoning_dataset(model, num_samples=5):
    """Inspect tokenization of reasoning dataset."""
    print("\n" + "=" * 80)
    print("REASONING DATASET")
    print("=" * 80)
    
    # Load reasoning chains
    with open('reasoning_dataset/annotated_dataset.json', 'r') as f:
        reasoning_chains = json.load(f)
    
    # Create reasoning dataset with Gemma tokenizer
    # Test WITHOUT chat template (new approach)
    print("\nUsing chat template:")
    chat_template = ""  # No template
    print(f"  {chat_template!r} (empty - just raw reasoning text)")
    
    reasoning_dataset = ReasoningChainDataset(
        reasoning_chains,
        tokenizer_name='google/gemma-2-2b',  # Use full model name
        tokenizer_chat_template=chat_template,
        max_length=128,
        include_question=False,  # No question
        discard_tokens_after_max_length=False,
    )
    
    print(f"\nTotal chunks: {len(reasoning_dataset)}")
    print(f"Special token IDs: {reasoning_dataset.tokenizer.all_special_ids}")
    
    # Show samples
    for i in range(min(3, min(num_samples, len(reasoning_dataset)))):
        chunk = reasoning_dataset[i]
        tokens = torch.tensor(chunk['input_ids'])
        text = reasoning_dataset.tokenizer.decode(tokens)
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Tokens (first 20): {tokens[:20].tolist()}")
        print(f"Length: {len(tokens)}")
        print(f"Text (first 300 chars): {text[:300]!r}")
        print(f"Has BOS at start: {tokens[0].item() == reasoning_dataset.tokenizer.bos_token_id}")
        
        # Count special tokens
        special_tokens = torch.tensor(reasoning_dataset.tokenizer.all_special_ids)
        is_special = torch.isin(tokens, special_tokens)
        print(f"Number of special tokens: {is_special.sum().item()}")
        
        # Show part infos
        print(f"Part infos: {chunk['part_infos'][:3]}...")  # First 3 parts


def main():
    print("\n" + "=" * 80)
    print("TOKENIZATION INSPECTION")
    print("=" * 80)
    print("\nThis script inspects the tokenization of normal vs reasoning datasets")
    print("to identify why classification might be trivially easy.\n")
    
    model = inspect_normal_dataset(num_samples=500)
    inspect_reasoning_dataset(model, num_samples=5)
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("\nKey differences to look for:")
    print("1. Different special token usage")
    print("2. Chat template formatting in reasoning data")
    print("3. Different sequence patterns or lengths")
    print("4. Obvious markers that make classification trivial")
    print("\nIf the datasets have very different tokenization patterns,")
    print("the probe might be learning these surface features rather than")
    print("actual reasoning content.")
    print("=" * 80)


if __name__ == '__main__':
    main()
