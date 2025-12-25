"""Feature activation collection from SAEs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
from torch import Tensor
from jaxtyping import Float, Int
import tqdm

from sae_lens import SAE, HookedSAETransformer


@dataclass
class FeatureActivations:
    """Container for feature activations with metadata."""
    
    # Feature activations: [n_samples, seq_len, n_features]
    activations: Float[Tensor, "samples seq features"]
    
    # Input tokens: [n_samples, seq_len]
    tokens: Int[Tensor, "samples seq"]
    
    # Labels: True for reasoning, False for non-reasoning
    is_reasoning: list[bool]
    
    # Source dataset for each sample
    sources: list[str]
    
    # Layer index
    layer_index: int
    
    # Model and SAE info
    model_name: str
    sae_name: str
    
    def save(self, path: Path):
        """Save activations to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "activations": self.activations,
            "tokens": self.tokens,
            "is_reasoning": self.is_reasoning,
            "sources": self.sources,
            "layer_index": self.layer_index,
            "model_name": self.model_name,
            "sae_name": self.sae_name,
        }, path)
    
    @classmethod
    def load(cls, path: Path) -> "FeatureActivations":
        """Load activations from disk."""
        data = torch.load(path, map_location="cpu")
        return cls(
            activations=data["activations"],
            tokens=data["tokens"],
            is_reasoning=data["is_reasoning"],
            sources=data["sources"],
            layer_index=data["layer_index"],
            model_name=data["model_name"],
            sae_name=data["sae_name"],
        )
    
    @property
    def n_samples(self) -> int:
        return self.activations.shape[0]
    
    @property
    def seq_len(self) -> int:
        return self.activations.shape[1]
    
    @property
    def n_features(self) -> int:
        return self.activations.shape[2]
    
    def get_reasoning_mask(self) -> Tensor:
        """Get boolean tensor mask for reasoning samples."""
        return torch.tensor(self.is_reasoning, dtype=torch.bool)
    
    def get_max_activations(self) -> Float[Tensor, "samples features"]:
        """Get max activation per feature per sample (across sequence)."""
        return self.activations.max(dim=1).values
    
    def get_mean_activations(self) -> Float[Tensor, "samples features"]:
        """Get mean activation per feature per sample (across sequence)."""
        return self.activations.mean(dim=1)


class FeatureCollector:
    """
    Collects SAE feature activations for text samples.
    
    This class handles loading models, tokenizing text, and extracting
    feature activations from specified SAE layers.
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        sae_name: str = "gemma-scope-2b-pt-res-canonical",
        sae_id_format: str = "layer_{layer}/width_16k/canonical",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            model_name: HuggingFace model name
            sae_name: SAE release name from sae_lens
            sae_id_format: Format string for SAE ID with {layer} placeholder
            device: Device to run on
            dtype: Model dtype
        """
        self.model_name = model_name
        self.sae_name = sae_name
        self.sae_id_format = sae_id_format
        self.device = device
        self.dtype = dtype
        
        self.model: Optional[HookedSAETransformer] = None
        self.sae: Optional[SAE] = None
        self.current_layer: Optional[int] = None
    
    def load_model(self):
        """Load the transformer model."""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.model = HookedSAETransformer.from_pretrained_no_processing(
                self.model_name,
                device=self.device,
                dtype=self.dtype,
            )
    
    def load_sae(self, layer_index: int):
        """Load SAE for a specific layer."""
        if self.current_layer != layer_index:
            print(f"Loading SAE for layer {layer_index}")
            sae_id = self.sae_id_format.format(layer=layer_index)
            self.sae = SAE.from_pretrained(
                release=self.sae_name,
                sae_id=sae_id,
                device=self.device,
            )
            if isinstance(self.sae, tuple):
                self.sae = self.sae[0]
            self.current_layer = layer_index
    
    def tokenize_texts(
        self,
        texts: list[str],
        max_length: int = 512,
    ) -> Int[Tensor, "batch seq"]:
        """Tokenize a list of texts."""
        self.load_model()
        
        encoded = self.model.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return encoded["input_ids"]
    
    def collect_activations(
        self,
        tokens: Int[Tensor, "batch seq"],
        layer_index: int,
        is_reasoning: list[bool],
        sources: list[str],
        batch_size: int = 8,
        max_features: Optional[int] = None,
    ) -> FeatureActivations:
        """
        Collect feature activations for tokenized inputs.
        
        Args:
            tokens: Tokenized input tensor [batch, seq]
            layer_index: Which layer's SAE to use
            is_reasoning: Boolean labels for each sample
            sources: Source dataset name for each sample
            batch_size: Batch size for processing
            max_features: Maximum number of features to collect (None = all)
        
        Returns:
            FeatureActivations containing collected data
        """
        self.load_model()
        self.load_sae(layer_index)
        
        all_activations = []
        n_samples = tokens.shape[0]
        
        try:
            hook_name = self.sae.cfg.metadata.hook_name
        except:
            hook_name = self.sae.cfg.hook_name
        
        with tqdm.tqdm(total=n_samples, desc="Collecting activations") as pbar:
            for i in range(0, n_samples, batch_size):
                batch = tokens[i:i + batch_size].to(self.device)
                
                with torch.no_grad():
                    _, cache = self.model.run_with_cache_with_saes(
                        batch,
                        saes=[self.sae],
                        use_error_term=True,  # Don't modify forward pass
                    )
                
                # Get feature activations (post top-k)
                acts = cache[f"{hook_name}.hook_sae_acts_post"]
                
                if max_features is not None:
                    acts = acts[:, :, :max_features]
                
                all_activations.append(acts.cpu().float())
                
                pbar.update(len(batch))
                del cache, batch
                torch.cuda.empty_cache()
        
        activations = torch.cat(all_activations, dim=0)
        
        return FeatureActivations(
            activations=activations,
            tokens=tokens.cpu(),
            is_reasoning=is_reasoning,
            sources=sources,
            layer_index=layer_index,
            model_name=self.model_name,
            sae_name=self.sae_name,
        )
    
    def collect_from_datasets(
        self,
        reasoning_dataset,
        nonreasoning_dataset,
        layer_index: int,
        max_length: int = 512,
        batch_size: int = 8,
        max_features: Optional[int] = None,
    ) -> FeatureActivations:
        """
        Collect activations from reasoning and non-reasoning datasets.
        
        Args:
            reasoning_dataset: Dataset with reasoning text (BaseDataset)
            nonreasoning_dataset: Dataset with non-reasoning text (BaseDataset)
            layer_index: Which layer's SAE to use
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            max_features: Maximum features to collect
        
        Returns:
            Combined FeatureActivations
        """
        # Ensure datasets are loaded
        reasoning_dataset.load()
        nonreasoning_dataset.load()
        
        # Combine texts and metadata
        all_texts = []
        is_reasoning = []
        sources = []
        
        for sample in reasoning_dataset:
            all_texts.append(sample.text)
            is_reasoning.append(True)
            sources.append(sample.source)
        
        for sample in nonreasoning_dataset:
            all_texts.append(sample.text)
            is_reasoning.append(False)
            sources.append(sample.source)
        
        # Tokenize
        tokens = self.tokenize_texts(all_texts, max_length=max_length)
        
        # Collect activations
        return self.collect_activations(
            tokens=tokens,
            layer_index=layer_index,
            is_reasoning=is_reasoning,
            sources=sources,
            batch_size=batch_size,
            max_features=max_features,
        )
