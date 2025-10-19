"""
Detailed inspection of actual data samples to find trivial differences.
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


def inspect_questions_vs_reasoning():
    """Compare questions (control) vs reasoning chains in detail."""
    print("=" * 80)
    print("DETAILED DATA INSPECTION: Questions vs Reasoning")
    print("=" * 80)
    
    # Load model
    model_name = 'google/gemma-2-2b'
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device='cpu',
        dtype=torch.bfloat16,
    )
    
    # Load reasoning chains
    with open('reasoning_dataset/annotated_dataset.json', 'r') as f:
        reasoning_chains = json.load(f)
    
    print("\n" + "=" * 80)
    print("QUESTIONS (Control - Non-reasoning)")
    print("=" * 80)
    
    # Create dataset with questions
    question_dataset = ReasoningChainDataset(
        reasoning_chains,
        tokenizer_name=model_name,
        tokenizer_chat_template="{question}\n",  # No template
        max_length=32,
        include_question=True,  # Include question
        discard_tokens_after_max_length=True,  # Just first chunk
    )
    
    print(f"\nTotal question chunks: {len(question_dataset)}")
    
    # Show 5 question samples
    for i in range(5):
        chunk = question_dataset[i]
        tokens = torch.tensor(chunk['input_ids'])
        text = question_dataset.tokenizer.decode(tokens)
        
        print(f"\n--- Question Sample {i+1} ---")
        print(f"Length: {len(tokens)} tokens")
        print(f"Text: {text!r}")
        print(f"Part infos: {chunk['part_infos']}")
    
    print("\n" + "=" * 80)
    print("REASONING CHAINS (Reasoning)")
    print("=" * 80)
    
    # Create dataset with reasoning (no questions)
    reasoning_dataset = ReasoningChainDataset(
        reasoning_chains,
        tokenizer_name=model_name,
        tokenizer_chat_template="",  # No template
        max_length=32,
        include_question=False,  # NO question
        discard_tokens_after_max_length=False,  # Multiple chunks
    )
    
    print(f"\nTotal reasoning chunks: {len(reasoning_dataset)}")
    
    # Show 5 reasoning samples
    for i in range(5):
        chunk = reasoning_dataset[i]
        tokens = torch.tensor(chunk['input_ids'])
        text = reasoning_dataset.tokenizer.decode(tokens)
        
        print(f"\n--- Reasoning Sample {i+1} ---")
        print(f"Length: {len(tokens)} tokens")
        print(f"Text: {text!r}")
        print(f"Part infos (first 3): {chunk['part_infos'][:3]}")
    
    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Compare token patterns
    question_tokens = [question_dataset[i]['input_ids'] for i in range(min(10, len(question_dataset)))]
    reasoning_tokens = [reasoning_dataset[i]['input_ids'] for i in range(min(10, len(reasoning_dataset)))]
    
    print("\nToken statistics:")
    print(f"  Questions - avg length: {sum(len(t) for t in question_tokens) / len(question_tokens):.1f}")
    print(f"  Reasoning - avg length: {sum(len(t) for t in reasoning_tokens) / len(reasoning_tokens):.1f}")
    
    # Check for common starting patterns
    print("\nFirst 5 tokens of each:")
    print("  Questions:")
    for i, tokens in enumerate(question_tokens[:5]):
        print(f"    Sample {i+1}: {tokens[:5]}")
    
    print("  Reasoning:")
    for i, tokens in enumerate(reasoning_tokens[:5]):
        print(f"    Sample {i+1}: {tokens[:5]}")
    
    # Check if all questions have the same pattern
    if all(q[0] == question_tokens[0][0] for q in question_tokens):
        print(f"\n⚠️  ALL questions start with token {question_tokens[0][0]}")
    
    if all(r[0] == reasoning_tokens[0][0] for r in reasoning_tokens):
        print(f"⚠️  ALL reasoning samples start with token {reasoning_tokens[0][0]}")
    
    # Check for systematic differences in first 10 tokens
    print("\nChecking for systematic patterns in first 10 tokens:")
    question_first_tokens = [set(t[:10]) for t in question_tokens]
    reasoning_first_tokens = [set(t[:10]) for t in reasoning_tokens]
    
    # Find tokens that appear frequently in one but not the other
    question_common = set.intersection(*question_first_tokens) if question_first_tokens else set()
    reasoning_common = set.intersection(*reasoning_first_tokens) if reasoning_first_tokens else set()
    
    question_only = question_common - reasoning_common
    reasoning_only = reasoning_common - question_common
    
    if question_only:
        print(f"  Tokens common to all questions but not reasoning: {question_only}")
        for tok in list(question_only)[:3]:
            print(f"    Token {tok}: {question_dataset.tokenizer.decode([tok])!r}")
    
    if reasoning_only:
        print(f"  Tokens common to all reasoning but not questions: {reasoning_only}")
        for tok in list(reasoning_only)[:3]:
            print(f"    Token {tok}: {reasoning_dataset.tokenizer.decode([tok])!r}")


def inspect_normal_dataset_properly():
    """Inspect normal (Pile) dataset using the correct method from activation_collector."""
    print("\n" + "=" * 80)
    print("NORMAL DATASET (Pile) - Using Correct Method")
    print("=" * 80)
    
    model_name = 'google/gemma-2-2b'
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device='cpu',
        dtype=torch.bfloat16,
    )
    
    max_samples = 500
    max_seq_length = 127
    
    # Load dataset (following activation_collector.py)
    dataset = load_dataset(
        path='monology/pile-uncopyrighted',
        split='train',
        streaming=True,
    )
    
    # Filter out non-ASCII characters and truncate
    dataset = dataset.filter(lambda x: len(x['text']) == len(x['text'].encode()))
    chars_per_token = 4
    dataset = dataset.map(
        lambda x: {'text': x['text'][:max_seq_length * chars_per_token * 2]}
    )
    dataset = HFDataset.from_list(list(dataset.take(max_samples * 3)))
    
    print(f"Initial dataset size: {len(dataset)}")
    
    # Get BOS setting (assuming True for Gemma)
    prepend_bos = True
    
    token_dataset = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=model.tokenizer,
        max_length=(max_seq_length + 1) if prepend_bos else max_seq_length,
        add_bos_token=prepend_bos,
        streaming=True,
    )['tokens']
    
    print(f"After tokenization: {token_dataset.shape}")
    
    # Filter out sequences with too many special tokens
    max_special_tokens = 1 if prepend_bos else 0
    special_tokens = torch.tensor(model.tokenizer.all_special_ids).to(token_dataset.device)
    is_special_token = torch.isin(token_dataset, special_tokens)
    token_dataset = token_dataset[is_special_token.sum(dim=1) <= max_special_tokens]
    
    print(f"After filtering special tokens: {token_dataset.shape}")
    
    token_dataset = token_dataset[:max_samples]
    
    print(f"Final shape: {token_dataset.shape}")
    
    # Show samples
    for i in range(min(3, len(token_dataset))):
        tokens = token_dataset[i]
        text = model.tokenizer.decode(tokens)
        
        print(f"\n--- Normal Sample {i+1} ---")
        print(f"Tokens (first 10): {tokens[:10].tolist()}")
        print(f"Length: {len(tokens)}")
        print(f"Text: {text[:300]!r}")


def main():
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DATA INSPECTION")
    print("=" * 80)
    print("\nThis script inspects actual detokenized data to find trivial differences.\n")
    
    # First check normal dataset
    inspect_normal_dataset_properly()
    
    # Then check questions vs reasoning
    inspect_questions_vs_reasoning()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nLook for:")
    print("  1. Different token patterns at start/end")
    print("  2. Systematic vocabulary differences")
    print("  3. Length differences")
    print("  4. Any obvious markers that make classes separable")
    print("=" * 80)


if __name__ == '__main__':
    main()

