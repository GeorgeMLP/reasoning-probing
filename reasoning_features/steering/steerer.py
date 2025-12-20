"""
Feature steering for SAE-based intervention experiments.

This module implements activation steering by modifying SAE feature
activations during model inference.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import torch
from torch import Tensor
from jaxtyping import Float

from sae_lens import SAE, HookedSAETransformer


@dataclass
class SteeringConfig:
    """Configuration for feature steering."""
    
    # Features to steer (indices into SAE feature space)
    feature_indices: list[int]
    
    # Steering multiplier (>1 amplifies, <1 suppresses, 0 removes)
    multiplier: float = 2.0
    
    # Alternative: add fixed value to activations
    additive_value: Optional[float] = None
    
    # Apply steering only after this token position
    start_position: int = 0
    
    # Layer index to steer
    layer_index: int = 8
    
    def __post_init__(self):
        if not self.feature_indices:
            raise ValueError("Must specify at least one feature to steer")


class FeatureSteerer:
    """
    Steers model behavior by modifying SAE feature activations.
    
    This class hooks into the model's forward pass and modifies the
    activations of specified SAE features, allowing us to test whether
    amplifying "reasoning features" actually improves reasoning performance.
    
    ## Steering Methods
    
    1. **Multiplicative**: Multiply feature activations by a scalar
       - multiplier > 1: Amplify the feature
       - multiplier < 1: Suppress the feature
       - multiplier = 0: Remove the feature entirely
    
    2. **Additive**: Add a fixed value to feature activations
       - Useful for features that may not be active
    
    ## Usage Example
    
    ```python
    steerer = FeatureSteerer(model, sae)
    config = SteeringConfig(
        feature_indices=[42, 128, 256],
        multiplier=2.0,
        layer_index=8,
    )
    
    # Generate with steering
    output = steerer.generate_with_steering(
        prompt="Solve: 2 + 2 = ",
        config=config,
        max_new_tokens=100,
    )
    ```
    """
    
    def __init__(
        self,
        model: HookedSAETransformer,
        sae: SAE,
    ):
        """
        Args:
            model: The transformer model
            sae: The SAE for the layer to steer
        """
        self.model = model
        self.sae = sae
        self.hook_name = sae.cfg.metadata.hook_name
        self._hooks_active = False
    
    def _create_steering_hook(
        self,
        config: SteeringConfig,
    ) -> Callable:
        """Create a hook function that modifies SAE feature activations."""
        
        def steering_hook(
            activations: Float[Tensor, "batch seq d_model"],
            hook,
        ) -> Float[Tensor, "batch seq d_model"]:
            """Hook that modifies activations via SAE encode/decode."""
            # Get original shape
            original_shape = activations.shape
            batch_size, seq_len, d_model = original_shape
            
            # Encode through SAE
            # Note: SAE expects [batch, d_model] so we need to reshape
            flat_acts = activations.reshape(-1, d_model)
            
            # Get feature activations
            feature_acts = self.sae.encode(flat_acts)
            
            # Apply steering to specified features
            for feat_idx in config.feature_indices:
                if config.additive_value is not None:
                    # Additive steering
                    feature_acts[:, feat_idx] += config.additive_value
                else:
                    # Multiplicative steering
                    feature_acts[:, feat_idx] *= config.multiplier
            
            # Decode back to activation space
            steered_acts = self.sae.decode(feature_acts)
            
            # Reshape back
            steered_acts = steered_acts.reshape(original_shape)
            
            # Optionally only apply after start_position
            if config.start_position > 0:
                result = activations.clone()
                result[:, config.start_position:] = steered_acts[:, config.start_position:]
                return result
            
            return steered_acts
        
        return steering_hook
    
    def _register_hook(self, config: SteeringConfig):
        """Register the steering hook."""
        hook_fn = self._create_steering_hook(config)
        self.model.add_hook(self.hook_name, hook_fn)
        self._hooks_active = True
    
    def _clear_hooks(self):
        """Remove all active hooks."""
        if self._hooks_active:
            self.model.reset_hooks(clear_contexts=True, including_permanent=False)
            self._hooks_active = False
    
    def generate_with_steering(
        self,
        prompt: str,
        config: SteeringConfig,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **generate_kwargs,
    ) -> str:
        """
        Generate text with feature steering applied.
        
        Args:
            prompt: Input prompt
            config: Steering configuration
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (vs greedy)
            **generate_kwargs: Additional arguments for generate()
        
        Returns:
            Generated text (without the prompt)
        """
        self._clear_hooks()
        
        try:
            self._register_hook(config)
            
            # Tokenize
            inputs = self.model.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.model.cfg.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    verbose=False,
                    **generate_kwargs,
                )
            
            # Decode (excluding prompt)
            prompt_length = inputs["input_ids"].shape[1]
            generated = outputs[0, prompt_length:]
            text = self.model.tokenizer.decode(generated, skip_special_tokens=True)
            
            return text
            
        finally:
            self._clear_hooks()
    
    def generate_baseline(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **generate_kwargs,
    ) -> str:
        """Generate text without any steering (baseline)."""
        self._clear_hooks()
        
        inputs = self.model.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.cfg.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                verbose=False,
                **generate_kwargs,
            )
        
        prompt_length = inputs["input_ids"].shape[1]
        generated = outputs[0, prompt_length:]
        return self.model.tokenizer.decode(generated, skip_special_tokens=True)
    
    def compare_generations(
        self,
        prompt: str,
        config: SteeringConfig,
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> dict:
        """
        Generate with and without steering for comparison.
        
        Returns:
            Dict with 'baseline' and 'steered' generations
        """
        baseline = self.generate_baseline(
            prompt, max_new_tokens=max_new_tokens, **generate_kwargs
        )
        steered = self.generate_with_steering(
            prompt, config, max_new_tokens=max_new_tokens, **generate_kwargs
        )
        
        return {
            "prompt": prompt,
            "baseline": baseline,
            "steered": steered,
            "config": {
                "feature_indices": config.feature_indices,
                "multiplier": config.multiplier,
                "additive_value": config.additive_value,
            },
        }


class MultiLayerSteerer:
    """
    Steers features across multiple layers simultaneously.
    
    Useful for experiments that require coordinated steering
    across the model's depth.
    """
    
    def __init__(
        self,
        model: HookedSAETransformer,
        saes: dict[int, SAE],  # layer_index -> SAE
    ):
        self.model = model
        self.saes = saes
        self.steerers = {
            layer: FeatureSteerer(model, sae)
            for layer, sae in saes.items()
        }
    
    def generate_with_multi_layer_steering(
        self,
        prompt: str,
        configs: dict[int, SteeringConfig],  # layer_index -> config
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> str:
        """Generate with steering applied to multiple layers."""
        # Clear all hooks first
        for steerer in self.steerers.values():
            steerer._clear_hooks()
        
        try:
            # Register hooks for each layer
            for layer, config in configs.items():
                if layer in self.steerers:
                    self.steerers[layer]._register_hook(config)
            
            # Use first steerer for generation (they share the model)
            first_steerer = list(self.steerers.values())[0]
            
            inputs = self.model.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.model.cfg.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    verbose=False,
                    **generate_kwargs,
                )
            
            prompt_length = inputs["input_ids"].shape[1]
            generated = outputs[0, prompt_length:]
            return self.model.tokenizer.decode(generated, skip_special_tokens=True)
            
        finally:
            for steerer in self.steerers.values():
                steerer._clear_hooks()
