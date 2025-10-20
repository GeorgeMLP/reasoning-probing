"""
Unified dataset interface for probing experiments.
Handles both normal (Pile) and reasoning datasets with activation collection.
"""

import torch
import json
from pathlib import Path
from typing import Literal, Optional
from torch import Tensor
from torch.utils.data import Dataset
from jaxtyping import Int
from datasets import load_dataset, Dataset as HFDataset
from transformer_lens.utils import tokenize_and_concatenate

from sae_lens import SAE, HookedSAETransformer
import tqdm

# Import reasoning dataset components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from reasoning_dataset.reasoning_dataset import ReasoningChainDataset


class UnifiedActivationDataset(Dataset):
    """
    Unified dataset that provides activations and labels for both reasoning and non-reasoning data.
    
    This dataset returns:
    - original_activation: Original layer activation
    - reconstructed_activation: SAE reconstructed activation
    - residue_activation: Difference between original and reconstructed
    - label: 0 for non-reasoning, 1 for reasoning (or fine-grained reasoning type)
    - is_reasoning: Boolean flag
    """
    
    def __init__(
        self,
        normal_activations: dict,
        reasoning_activations: dict,
        label_type: Literal['binary', 'fine_grained'] = 'binary',
    ):
        """
        Args:
            normal_activations: Dict with keys 'original', 'reconstructed', 'residue'
            reasoning_activations: Dict with keys 'original', 'reconstructed', 'residue', 'labels'
            label_type: 'binary' for reasoning vs non-reasoning, 'fine_grained' for specific types
        """
        self.label_type = label_type
        
        # Store normal data (label = 0)
        self.normal_original = normal_activations['original']
        self.normal_reconstructed = normal_activations['reconstructed']
        self.normal_residue = normal_activations['residue']
        self.normal_size = len(self.normal_original)
        
        # Store reasoning data (label = 1 or specific type)
        self.reasoning_original = reasoning_activations['original']
        self.reasoning_reconstructed = reasoning_activations['reconstructed']
        self.reasoning_residue = reasoning_activations['residue']
        self.reasoning_labels = reasoning_activations.get('labels', None)
        self.reasoning_size = len(self.reasoning_original)
        
        self.total_size = self.normal_size + self.reasoning_size
        
        print(f"Dataset created: {self.normal_size} normal + {self.reasoning_size} reasoning = {self.total_size} total")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if idx < self.normal_size:
            # Normal (non-reasoning) sample
            return {
                'original': self.normal_original[idx],
                'reconstructed': self.normal_reconstructed[idx],
                'residue': self.normal_residue[idx],
                'label': 0,  # Non-reasoning
                'is_reasoning': False,
            }
        else:
            # Reasoning sample
            reasoning_idx = idx - self.normal_size
            label = 1  # Binary: reasoning
            if self.label_type == 'fine_grained' and self.reasoning_labels is not None:
                label = self.reasoning_labels[reasoning_idx]
            
            return {
                'original': self.reasoning_original[reasoning_idx],
                'reconstructed': self.reasoning_reconstructed[reasoning_idx],
                'residue': self.reasoning_residue[reasoning_idx],
                'label': label,
                'is_reasoning': True,
            }


class ActivationDatasetBuilder:
    """
    Builds unified activation datasets by collecting activations from both
    normal and reasoning datasets.
    """
    
    def __init__(
        self,
        model_name: str = 'google/gemma-2-2b',
        sae_name: str = 'gemma-scope-2b-pt-res-canonical',
        sae_id_format: str = 'layer_{layer}/width_16k/canonical',
        device: str = 'cuda',
        layer_index: int = 8,
    ):
        self.model_name = model_name
        self.sae_name = sae_name
        self.sae_id_format = sae_id_format
        self.device = device
        self.layer_index = layer_index
        
        self.model = None
        self.sae = None
    
    def load_model_and_sae(self):
        """Load the model and SAE."""
        print(f"Loading model: {self.model_name}")
        self.model = HookedSAETransformer.from_pretrained_no_processing(
            self.model_name,
            device=self.device,
            dtype=torch.bfloat16,
        )
        
        print(f"Loading SAE for layer {self.layer_index}")
        sae_id = self.sae_id_format.format(layer=self.layer_index)
        self.sae = SAE.from_pretrained(
            release=self.sae_name,
            sae_id=sae_id,
            device=self.device,
        )
    
    def collect_normal_activations(
        self,
        max_samples: int = 5000,
        max_seq_length: int = 31,
        batch_size: int = 32,
    ) -> dict:
        """
        Collect activations from the normal (Pile) dataset.
        
        Returns:
            Dict with keys: 'original', 'reconstructed', 'residue'
        """
        print(f"\n=== Collecting activations from normal dataset ===")
        print(f"Target: {max_samples} samples, max_seq_length={max_seq_length}")
        
        # Load dataset
        dataset = load_dataset(
            path='monology/pile-uncopyrighted',
            split='train',
            streaming=True,
        )
        
        # Filter and prepare
        dataset = dataset.filter(lambda x: len(x['text']) == len(x['text'].encode()))
        chars_per_token = 4
        dataset = dataset.map(
            lambda x: {'text': x['text'][:max_seq_length * chars_per_token * 2]}
        )
        dataset = HFDataset.from_list(list(dataset.take(max_samples * 3)))
        
        # Tokenize
        prepend_bos = self.sae.cfg.metadata.prepend_bos
        token_dataset = tokenize_and_concatenate(
            dataset=dataset,
            tokenizer=self.model.tokenizer,
            max_length=(max_seq_length + 1) if prepend_bos else max_seq_length,
            add_bos_token=prepend_bos,
            streaming=True,
        )['tokens']
        
        # Filter special tokens
        max_special_tokens = 1 if prepend_bos else 0
        special_tokens = torch.tensor(self.model.tokenizer.all_special_ids).to(token_dataset.device)
        is_special_token = torch.isin(token_dataset, special_tokens)
        token_dataset = token_dataset[is_special_token.sum(dim=1) <= max_special_tokens]
        token_dataset = token_dataset[:max_samples]
        
        print(f"Tokenized dataset shape: {token_dataset.shape}")
        
        # Collect activations
        return self._collect_activations_from_tokens(token_dataset, batch_size)
    
    def collect_reasoning_activations(
        self,
        reasoning_data_path: str = 'reasoning_dataset/annotated_dataset.json',
        max_samples: Optional[int] = None,
        max_seq_length: int = 32,
        include_question: bool = True,
        label_type: Literal['binary', 'fine_grained'] = 'binary',
        collect_questions_as_control: bool = False,
    ) -> dict:
        """
        Collect activations from the reasoning dataset.
        
        Args:
            collect_questions_as_control: If True, collect questions (non-reasoning)
                                          instead of reasoning chains
        
        Returns:
            Dict with keys: 'original', 'reconstructed', 'residue', 'labels'
        """
        if collect_questions_as_control:
            print(f"\n=== Collecting activations from QUESTIONS (control) ===")
        else:
            print(f"\n=== Collecting activations from reasoning dataset ===")
        
        # Load reasoning chains
        with open(reasoning_data_path, 'r') as f:
            reasoning_chains = json.load(f)
        
        if collect_questions_as_control:
            # Collect ONLY the questions as control (no reasoning)
            reasoning_dataset = ReasoningChainDataset(
                reasoning_chains,
                tokenizer_name=self.model_name,
                tokenizer_chat_template="{question}\n",  # No template
                max_length=max_seq_length,
                include_question=True,  # Only question
                discard_tokens_after_max_length=True,  # Just first chunk (question)
            )
        else:
            # Create reasoning dataset with Gemma tokenizer
            # IMPORTANT: We don't use a chat template to avoid trivial classification
            # The probe should learn about reasoning content, not chat formatting
            reasoning_dataset = ReasoningChainDataset(
                reasoning_chains,
                tokenizer_name=self.model_name,  # Use Gemma tokenizer
                tokenizer_chat_template="",  # No chat template - just raw reasoning text
                max_length=max_seq_length,
                include_question=False,  # Don't include question to focus on reasoning
                discard_tokens_after_max_length=False,
            )
        
        print(f"Reasoning dataset size: {len(reasoning_dataset)} chunks")
        
        # Extract tokens and labels
        all_tokens = []
        all_labels = []
        
        # Label 0 is reserved for non-reasoning samples
        # Reasoning types get labels 1-6
        label_map = {
            'initializing': 1,
            'deduction': 2,
            'adding-knowledge': 3,
            'example-testing': 4,
            'uncertainty-estimation': 5,
            'backtracking': 6,
        }
        
        for i in range(len(reasoning_dataset)):
            if max_samples and i >= max_samples:
                break
            
            chunk = reasoning_dataset[i]
            tokens = torch.tensor(chunk['input_ids'])
            all_tokens.append(tokens)
            
            # Determine label
            if label_type == 'fine_grained':
                # Use the most common label in this chunk
                part_infos = chunk['part_infos']
                if part_infos:
                    most_common_label = max(
                        set(p['label'] for p in part_infos),
                        key=lambda l: sum(1 for p in part_infos if p['label'] == l)
                    )
                    all_labels.append(label_map.get(most_common_label, 1))  # Default to 'initializing' if unknown
                else:
                    all_labels.append(1)  # Default to 'initializing' if no part_infos
            else:
                # Binary: just mark as reasoning (1)
                all_labels.append(1)
        
        # Pad to same length
        max_len = max(len(t) for t in all_tokens)
        pad_token_id = reasoning_dataset.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        
        padded_tokens = []
        for tokens in all_tokens:
            if len(tokens) < max_len:
                padding = torch.full((max_len - len(tokens),), pad_token_id, dtype=tokens.dtype)
                tokens = torch.cat([tokens, padding])
            padded_tokens.append(tokens)
        
        token_tensor = torch.stack(padded_tokens)
        print(f"Tokenized reasoning dataset shape: {token_tensor.shape}")
        
        # Collect activations
        activations = self._collect_activations_from_tokens(token_tensor, batch_size=8)
        activations['labels'] = torch.tensor(all_labels)
        
        return activations
    
    def _collect_activations_from_tokens(
        self,
        tokens: Int[Tensor, 'batch seq'],
        batch_size: int = 32,
    ) -> dict:
        """
        Collect activations for a batch of tokens.
        
        Returns:
            Dict with 'original', 'reconstructed', 'residue' tensors
        """
        original_list = []
        reconstructed_list = []
        residue_list = []
        
        total_samples = len(tokens)
        hook_name = self.sae.cfg.metadata.hook_name
        
        with tqdm.tqdm(total=total_samples, desc='Collecting activations') as pbar:
            for i in range(0, total_samples, batch_size):
                batch = tokens[i:i + batch_size].to(self.device)
                
                with torch.no_grad():
                    _, cache = self.model.run_with_cache_with_saes(
                        batch,
                        saes=[self.sae],
                        use_error_term=False,
                    )
                
                # Extract activations
                original = cache[hook_name + '.hook_sae_input'].cpu().float()
                reconstructed = cache[hook_name + '.hook_sae_recons'].cpu().float()
                residue = original - reconstructed
                
                # Pool over sequence dimension (mean pooling)
                # This gives us one vector per sequence
                original_pooled = original.mean(dim=1)  # [batch, d_model]
                reconstructed_pooled = reconstructed.mean(dim=1)
                residue_pooled = residue.mean(dim=1)
                
                original_list.append(original_pooled)
                reconstructed_list.append(reconstructed_pooled)
                residue_list.append(residue_pooled)
                
                pbar.update(len(batch))
                del cache, batch
                torch.cuda.empty_cache()
        
        return {
            'original': torch.cat(original_list, dim=0),
            'reconstructed': torch.cat(reconstructed_list, dim=0),
            'residue': torch.cat(residue_list, dim=0),
        }
    
    def build_dataset(
        self,
        normal_samples: int = 5000,
        reasoning_samples: Optional[int] = None,
        label_type: Literal['binary', 'fine_grained'] = 'binary',
        save_dir: Optional[Path] = None,
        use_questions_as_control: bool = True,
    ) -> UnifiedActivationDataset:
        """
        Build a complete unified dataset with both normal and reasoning samples.
        
        Args:
            normal_samples: Number of normal (non-reasoning) samples
            reasoning_samples: Number of reasoning samples (None = all available)
            label_type: 'binary' or 'fine_grained'
            save_dir: Optional directory to save the collected activations
            use_questions_as_control: If True, use questions as control instead of Pile.
                                     This ensures both classes are from the same domain.
        """
        if self.model is None or self.sae is None:
            self.load_model_and_sae()
        
        # Collect activations
        if use_questions_as_control:
            print("\n*** Using QUESTIONS as control ***")
            normal_acts = self.collect_reasoning_activations(
                max_samples=normal_samples,
                label_type='binary',
                collect_questions_as_control=True,
            )
        else:
            print("\n*** Using Pile as control (may have domain shift issues) ***")
            normal_acts = self.collect_normal_activations(max_samples=normal_samples)
        
        reasoning_acts = self.collect_reasoning_activations(
            max_samples=reasoning_samples,
            label_type=label_type,
            collect_questions_as_control=False,
        )
        
        # Save if requested
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'normal': normal_acts,
                'reasoning': reasoning_acts,
                'config': {
                    'model_name': self.model_name,
                    'sae_name': self.sae_name,
                    'layer_index': self.layer_index,
                    'label_type': label_type,
                }
            }, save_dir / 'activations.pt')
            print(f"Saved activations to {save_dir / 'activations.pt'}")
        
        # Create unified dataset
        dataset = UnifiedActivationDataset(
            normal_activations=normal_acts,
            reasoning_activations=reasoning_acts,
            label_type=label_type,
        )
        
        return dataset
    
    @staticmethod
    def load_dataset(load_path: Path) -> UnifiedActivationDataset:
        """Load a previously saved dataset."""
        data = torch.load(load_path)
        return UnifiedActivationDataset(
            normal_activations=data['normal'],
            reasoning_activations=data['reasoning'],
            label_type=data['config']['label_type'],
        )


if __name__ == '__main__':
    # Example usage
    builder = ActivationDatasetBuilder(
        model_name='google/gemma-2-2b',
        layer_index=8,
    )
    
    dataset = builder.build_dataset(
        normal_samples=100,  # Small test
        reasoning_samples=50,
        label_type='binary',
        save_dir=Path('data/probing/test'),
    )
    
    print(f"\nDataset created with {len(dataset)} samples")
    print(f"Sample item keys: {dataset[0].keys()}")
    print(f"Original activation shape: {dataset[0]['original'].shape}")
