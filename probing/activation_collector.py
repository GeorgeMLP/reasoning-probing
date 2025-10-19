"""
Activation collection for residual SAE analysis.
Generates original vs reconstructed activations from pretrained models and SAEs.
"""

import torch
from pathlib import Path
import tqdm
from jaxtyping import Int
from torch import Tensor

from sae_lens import SAE, HookedSAETransformer
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset, Dataset

import warnings
warnings.filterwarnings('ignore', module='sae_lens', category=UserWarning)


class ChunkedTensorIterator:
    """Iterator for chunked tensor data without loading everything into memory."""
    
    def __init__(self, layer_dir: Path, tensor_type: str, chunk_size: int = 1000):
        self.layer_dir = layer_dir
        self.tensor_type = tensor_type
        self.chunk_size = chunk_size
        self.chunk_files = sorted(layer_dir.glob(f'{tensor_type}_chunk_*.pt'))
        self.total_samples = 0
        
        # Calculate total samples
        for chunk_file in self.chunk_files:
            chunk = torch.load(chunk_file, map_location='cpu')
            self.total_samples += chunk.shape[0]
    
    def __len__(self):
        return self.total_samples
    
    def __iter__(self):
        for chunk_file in self.chunk_files:
            chunk = torch.load(chunk_file, map_location='cpu')
            # Yield data in smaller sub-chunks
            for i in range(0, chunk.shape[0], self.chunk_size):
                yield chunk[i:i + self.chunk_size]
    
    def get_batch(self, start_idx: int, batch_size: int):
        """Get a specific batch by index."""
        current_idx = 0
        for chunk_file in self.chunk_files:
            chunk = torch.load(chunk_file, map_location='cpu')
            chunk_end = current_idx + chunk.shape[0]
            
            if start_idx < chunk_end:
                # This chunk contains our start_idx
                local_start = max(0, start_idx - current_idx)
                local_end = min(chunk.shape[0], local_start + batch_size)
                
                if local_start < local_end:
                    return chunk[local_start:local_end]
            
            current_idx = chunk_end
        
        return torch.empty(0)


class ActivationPair:
    """Chunked activation pair that doesn't load everything into memory."""
    
    def __init__(self, layer_name: str, layer_index: int, data_dir: Path, save_residuals_only: bool = False):
        self.layer_name = layer_name
        self.layer_index = layer_index
        self.data_dir = data_dir
        self.save_residuals_only = save_residuals_only
        
        layer_dir = data_dir / f'layer_{layer_index}'
        self._original_iter = None if save_residuals_only else ChunkedTensorIterator(layer_dir, 'original')
        self._reconstructed_iter = None if save_residuals_only else ChunkedTensorIterator(layer_dir, 'reconstructed')
        self._residual_iter = ChunkedTensorIterator(layer_dir, 'residual')
    
    def iter_residuals(self, chunk_size: int = 1000):
        """Iterate over residual data in chunks."""
        layer_dir = self.data_dir / f'layer_{self.layer_index}'
        iterator = ChunkedTensorIterator(layer_dir, 'residual', chunk_size)
        return iterator
    
    def get_residual_batch(self, start_idx: int, batch_size: int):
        """Get a specific batch of residuals."""
        return self._residual_iter.get_batch(start_idx, batch_size)
    
    def __len__(self):
        return len(self._residual_iter)
    
    @property
    def residual(self):
        """Legacy property - not supported with chunked data."""
        raise RuntimeError("Loading all residuals into memory is not supported. Use iter_residuals() or get_residual_batch() instead.")


class ActivationCollector:
    """Collects original and reconstructed activations from SAEs."""
    
    def __init__(
        self,
        model_name: str = 'google/gemma-2-9b',
        sae_name: str = 'gemma-scope-9b-pt-res-canonical',
        sae_id_format: str = 'layer_{layer}/width_16k/canonical',
        device: str = 'cuda',
    ):
        self.model_name = model_name
        self.sae_name = sae_name
        self.sae_id_format = sae_id_format
        self.device = device
        
    def load_model_and_saes(self, layer_indices: list[int]):
        """Load the model and SAEs for specified layers."""
        print(f"Loading model: {self.model_name}")
        self.model = HookedSAETransformer.from_pretrained_no_processing(
            self.model_name,
            device=self.device,
            dtype=torch.bfloat16,
        )
        
        print(f"Loading SAEs for layers: {layer_indices}")
        self.saes: dict[int, SAE] = {}
        for layer_idx in tqdm.tqdm(layer_indices, desc="Loading SAEs"):
            sae_id = self.sae_id_format.format(layer=layer_idx)
            sae = SAE.from_pretrained(
                release=self.sae_name,
                sae_id=sae_id,
                device=self.device,
            )
            self.saes[layer_idx] = sae
            
    def construct_dataset(
        self,
        dataset_name: str = 'monology/pile-uncopyrighted',
        max_dataset_size: int = 10000,
        max_seq_length: int = 32,
    ) -> Int[Tensor, 'batch seq']:
        """Load and preprocess the dataset."""
        print(f"Loading dataset: {dataset_name}")
        
        dataset = load_dataset(
            path=dataset_name,
            split='train',
            streaming=True,
        )

        # Filter out non-ASCII characters and truncate
        dataset = dataset.filter(lambda x: len(x['text']) == len(x['text'].encode()))
        chars_per_token = 4
        dataset = dataset.map(
            lambda x: {'text': x['text'][:max_seq_length * chars_per_token * 2]}
        )
        dataset = Dataset.from_list(list(dataset.take(max_dataset_size * 3)))
        
        # Get BOS settings from first SAE
        first_layer = min(self.saes.keys())
        prepend_bos: bool = self.saes[first_layer].cfg.metadata.prepend_bos
        
        token_dataset = tokenize_and_concatenate(
            dataset=dataset,
            tokenizer=self.model.tokenizer,
            max_length=(max_seq_length + 1) if prepend_bos else max_seq_length,
            add_bos_token=prepend_bos,
            streaming=True,
        )['tokens']
        
        # Filter out sequences with too many special tokens
        max_special_tokens = 1 if prepend_bos else 0
        special_tokens = torch.tensor(self.model.tokenizer.all_special_ids).to(token_dataset.device)
        is_special_token = torch.isin(token_dataset, special_tokens)
        token_dataset = token_dataset[is_special_token.sum(dim=1) <= max_special_tokens]
        
        return token_dataset[:max_dataset_size]
    
    def collect_activations(
        self,
        dataset_tokens: Int[Tensor, 'batch seq'],
        batch_size: int = 32,
        save_dir: Path = None,
        save_residuals_only: bool = False,
    ) -> dict[int, ActivationPair]:
        """Collect activations with ultra-efficient disk usage.
        
        Strategy: Stream directly to final files without temporary storage.
        """
        print("Collecting activations with direct streaming to final files...")
        
        # Clear GPU memory before starting
        torch.cuda.empty_cache()
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'set_use_attn_result'):
            self.model.set_use_attn_result(True)
        
        activation_pairs: dict[int, ActivationPair] = {}
        saes_list = list(self.saes.values())
        hook_layers = [int(sae.cfg.metadata.hook_name.split('.')[1]) for sae in saes_list]
        
        # Determine save directory
        if save_dir is None:
            save_dir = Path("/tmp/sae_activations")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize output directories for chunked storage
        output_files = {}
        for layer_idx in hook_layers:
            layer_dir = save_dir / f'layer_{layer_idx}'
            layer_dir.mkdir(exist_ok=True, parents=True)
            
            output_files[layer_idx] = {
                'layer_dir': layer_dir,
                'total_samples': 0,
                'chunk_count': 0,
                'save_residuals_only': save_residuals_only
            }
        
        total_samples = len(dataset_tokens)
        
        print(f"Processing {total_samples} samples in chunks of {batch_size}")
        
        # Process one mini-batch at a time and write immediately
        with tqdm.tqdm(total=total_samples, desc='Streaming activations') as pbar:
            
            # Collect chunks in small batches to reduce I/O overhead
            collected_chunks = {layer_idx: {'orig': [], 'recon': [], 'resid': []} for layer_idx in hook_layers}
            chunks_to_write = 32  # Write every N chunks to balance memory vs I/O
            
            for i in range(0, total_samples, batch_size):
                actual_chunk_size = min(batch_size, total_samples - i)
                batch = dataset_tokens[i:i + actual_chunk_size].to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    _, cache = self.model.run_with_cache_with_saes(
                        batch,
                        saes=saes_list,
                        use_error_term=False,
                    )
                
                # Collect activations in memory temporarily
                for sae in saes_list:
                    layer_idx = int(sae.cfg.metadata.hook_name.split('.')[1])
                    hook_name = sae.cfg.metadata.hook_name
                    
                    original_acts = cache[hook_name + '.hook_sae_input'].detach().cpu().float()  # Keep float32 for numerical stability
                    reconstructed_acts = cache[hook_name + '.hook_sae_recons'].detach().cpu().float()
                    residual_acts = original_acts - reconstructed_acts
                    
                    if save_residuals_only:
                        # Only save residuals to save 66% disk space
                        collected_chunks[layer_idx]['resid'].append(residual_acts)
                    else:
                        collected_chunks[layer_idx]['orig'].append(original_acts)
                        collected_chunks[layer_idx]['recon'].append(reconstructed_acts)
                        collected_chunks[layer_idx]['resid'].append(residual_acts)
                    
                    output_files[layer_idx]['total_samples'] += actual_chunk_size
                
                # Clear GPU memory
                del cache, batch
                torch.cuda.empty_cache()
                pbar.update(actual_chunk_size)
                
                # Write collected chunks to disk periodically
                chunks_ready = len(collected_chunks[hook_layers[0]]['resid']) >= chunks_to_write or i + batch_size >= total_samples
                if chunks_ready:
                    self._write_chunks_to_files(collected_chunks, output_files, save_residuals_only)
                    # Clear collected chunks
                    collected_chunks = {layer_idx: {'orig': [], 'recon': [], 'resid': []} for layer_idx in hook_layers}
        
        # Create activation pairs that point to the final files
        for sae in saes_list:
            layer_idx = int(sae.cfg.metadata.hook_name.split('.')[1])
            hook_name = sae.cfg.metadata.hook_name
            
            # Create chunked ActivationPair
            activation_pairs[layer_idx] = ActivationPair(
                layer_name=hook_name,
                layer_index=layer_idx,
                data_dir=save_dir,
                save_residuals_only=save_residuals_only,
            )
            
            print(f"Layer {layer_idx}: {output_files[layer_idx]['total_samples']} samples written to {save_dir}")
        
        return activation_pairs
    
    def _write_chunks_to_files(self, collected_chunks, output_files, save_residuals_only=False):
        """Write collected chunks as individual chunk files (no concatenation)."""
        for layer_idx, chunks in collected_chunks.items():
            if not chunks['resid']:  # Skip if no chunks
                continue
                
            layer_dir = output_files[layer_idx]['layer_dir']
            chunk_idx = output_files[layer_idx]['chunk_count']
            
            if save_residuals_only:
                # Only save residuals (66% disk space reduction)
                resid_batch = torch.cat(chunks['resid'], dim=0)
                torch.save(resid_batch, layer_dir / f'residual_chunk_{chunk_idx}.pt')
                del resid_batch
            else:
                # Save all tensor types as separate chunks
                orig_batch = torch.cat(chunks['orig'], dim=0)
                recon_batch = torch.cat(chunks['recon'], dim=0) 
                resid_batch = torch.cat(chunks['resid'], dim=0)
                
                torch.save(orig_batch, layer_dir / f'original_chunk_{chunk_idx}.pt')
                torch.save(recon_batch, layer_dir / f'reconstructed_chunk_{chunk_idx}.pt')
                torch.save(resid_batch, layer_dir / f'residual_chunk_{chunk_idx}.pt')
                
                del orig_batch, recon_batch, resid_batch
            
            # Increment chunk counter
            output_files[layer_idx]['chunk_count'] += 1
    
    def save_activations(
        self,
        activation_pairs: dict[int, ActivationPair],
        dataset_tokens: Int[Tensor, 'batch seq'],
        save_dir: Path,
    ):
        """Save dataset tokens and metadata (activations already saved during collection)."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset tokens
        torch.save(dataset_tokens, save_dir / 'tokens.pt')
        print(f"Saved dataset tokens with shape {dataset_tokens.shape}")
        
        # Save metadata for each layer
        for layer_idx, pair in activation_pairs.items():
            layer_dir = save_dir / f'layer_{layer_idx}'
            layer_dir.mkdir(exist_ok=True)
            
            # Count chunks to include in metadata
            residual_chunks = len(list(layer_dir.glob('residual_chunk_*.pt')))
            original_chunks = len(list(layer_dir.glob('original_chunk_*.pt')))
            
            metadata = {
                'layer_name': pair.layer_name,
                'layer_index': pair.layer_index,
                'storage_type': 'chunked',
                'residual_chunks': residual_chunks,
                'original_chunks': original_chunks,
                'save_residuals_only': getattr(pair, '_save_residuals_only', False),
            }
            
            import json
            with open(layer_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Activation collection complete. Files saved to {save_dir}")
    
    def load_activations(
        self,
        load_dir: Path,
        layer_indices: list[int] = None,
    ) -> dict[int, ActivationPair]:
        """Load chunked activation pairs."""
        if not load_dir.exists():
            raise FileNotFoundError(f"Directory {load_dir} does not exist")
        
        activation_pairs: dict[int, ActivationPair] = {}
        
        # Auto-discover layers if not specified
        if layer_indices is None:
            layer_indices = []
            for item in load_dir.iterdir():
                if item.is_dir() and item.name.startswith('layer_'):
                    try:
                        layer_idx = int(item.name.split('_')[1])
                        layer_indices.append(layer_idx)
                    except (ValueError, IndexError):
                        continue
            layer_indices.sort()
        
        print(f"Loading chunked activation pairs from {load_dir} for layers: {layer_indices}")
        
        for layer_idx in layer_indices:
            layer_dir = load_dir / f'layer_{layer_idx}'
            if not layer_dir.exists():
                print(f"Warning: Layer directory {layer_dir} does not exist, skipping")
                continue
            
            # Check for chunk files
            residual_chunks = list(layer_dir.glob('residual_chunk_*.pt'))
            if not residual_chunks:
                print(f"Warning: No residual chunks found for layer {layer_idx}, skipping")
                continue
            
            # Load metadata
            layer_name = f'blocks.{layer_idx}.hook_resid_post'
            save_residuals_only = False
            
            metadata_path = layer_dir / 'metadata.json'
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    layer_name = metadata.get('layer_name', layer_name)
                    save_residuals_only = metadata.get('save_residuals_only', False)
            
            # Create chunked ActivationPair
            activation_pairs[layer_idx] = ActivationPair(
                layer_name=layer_name,
                layer_index=layer_idx,
                data_dir=load_dir,
                save_residuals_only=save_residuals_only,
            )
            
            print(f"Loaded chunked data for layer {layer_idx} ({len(residual_chunks)} chunks)")
        
        print(f"Successfully loaded chunked activation pairs for {len(activation_pairs)} layers")
        return activation_pairs
    
    def get_chunked_iterator(self, save_dir: Path, layer_idx: int, tensor_type: str) -> ChunkedTensorIterator:
        """Get an iterator for chunked data."""
        layer_dir = save_dir / f'layer_{layer_idx}'
        return ChunkedTensorIterator(layer_dir, tensor_type)


def main():
    """Example usage of ActivationCollector with both save and load functionality."""
    model_name = 'google/gemma-2-2b'
    sae_name = 'gemma-scope-2b-pt-res-canonical'
    sae_id_format = 'layer_{layer}/width_16k/canonical'
    collector = ActivationCollector(
        model_name=model_name,
        sae_name=sae_name,
        sae_id_format=sae_id_format,
    )
    
    save_dir = Path(f'data/{model_name.replace("/", "_")}/activations')
    layer_indices = [8, 16]
    
    # Check if activation data already exists
    if save_dir.exists() and any((save_dir / f'layer_{i}').exists() for i in layer_indices):
        print("=== Loading existing activation data ===")
        activation_pairs = collector.load_activations(save_dir, layer_indices)
        
        # Load tokens
        tokens_path = save_dir / 'tokens.pt'
        if tokens_path.exists():
            dataset_tokens = torch.load(tokens_path, map_location='cpu')
            print(f"Loaded dataset tokens with shape: {dataset_tokens.shape}")
        else:
            print("Warning: tokens.pt not found")
            dataset_tokens = None
            
    else:
        print("=== Collecting new activation data ===")
        # Load model and SAEs
        collector.load_model_and_saes(layer_indices)
        
        # Generate dataset
        dataset_tokens = collector.construct_dataset(max_dataset_size=1000)
        
        # Collect activations (saves directly to final location)
        activation_pairs = collector.collect_activations(dataset_tokens, save_dir=save_dir)
        
        # Save tokens and metadata
        collector.save_activations(activation_pairs, dataset_tokens, save_dir)
    
    print(f"\n=== Summary ===")
    print(f"Loaded activation pairs for {len(activation_pairs)} layers")
    for layer_idx, pair in activation_pairs.items():
        # Calculate residual norm from first chunk to avoid loading all data
        residual_iter = pair.iter_residuals(chunk_size=100)
        first_chunk = next(iter(residual_iter))
        residual_norm = float(first_chunk.norm(2))
        batch_size, seq_len, d_model = first_chunk.shape
        total_samples = len(pair)
        print(f"Layer {layer_idx}: {total_samples} samples, shape ({batch_size}, {seq_len}, {d_model}), Residual norm (sample) = {residual_norm:.2f}")
    
    if dataset_tokens is not None:
        print(f"Dataset tokens shape: {dataset_tokens.shape}")
        
    return activation_pairs, dataset_tokens


if __name__ == '__main__':
    main()
