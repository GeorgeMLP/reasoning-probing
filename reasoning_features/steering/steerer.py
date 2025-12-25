"""
Feature steering for SAE-based intervention experiments.

This module implements activation steering by adding scaled decoder directions
to the residual stream during model inference.

## Steering Formula

We use direct decoder direction steering:

    x' = x + γ * f_max * W_dec[i]

Where:
- x: Original residual stream activation
- γ: Steering strength (typically -4 to 4)
- f_max: Maximum activation of feature i (pre-computed)
- W_dec[i]: The i-th decoder direction (row of SAE decoder matrix)

This approach directly modifies the residual stream by adding a scaled version
of the decoder direction, rather than modifying feature activations through
encode/decode.
"""

from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor
from jaxtyping import Float

from sae_lens import SAE, HookedSAETransformer


@dataclass
class SteeringConfig:
    """Configuration for feature steering.
    
    Attributes:
        feature_index: Single feature index to steer
        gamma: Steering strength (typically -4 to 4)
        max_feature_activation: Maximum activation f_max for this feature
        layer_index: Layer index to steer
        start_position: Apply steering only after this token position
    """
    
    # Feature to steer (single index)
    feature_index: int
    
    # Steering strength γ (typically -4 to 4)
    # Positive values amplify, negative values suppress
    gamma: float = 1.0
    
    # Maximum activation of this feature (f_max)
    max_feature_activation: float = 1.0
    
    # Layer index to steer
    layer_index: int = 8
    
    # Apply steering only after this token position
    start_position: int = 0


class FeatureSteerer:
    """
    Steers model behavior by adding scaled decoder directions.
    
    This class hooks into the model's forward pass and modifies the
    residual stream by adding a scaled decoder direction for the
    specified feature.
    
    ## Steering Formula
    
    x' = x + γ * f_max * W_dec[i]
    
    Where:
    - x: Original residual stream activation [batch, seq, d_model]
    - γ: Steering strength (config.gamma)
    - f_max: Maximum activation (config.max_feature_activation)
    - W_dec[i]: Decoder direction for feature i [d_model]
    
    ## Usage Example
    
    ```python
    steerer = FeatureSteerer(model, sae)
    config = SteeringConfig(
        feature_index=42,
        gamma=2.0,
        max_feature_activation=15.0,
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
        try:
            self.hook_name = sae.cfg.metadata.hook_name
        except:
            self.hook_name = sae.cfg.hook_name
        self._hooks_active = False
        
        # Get decoder matrix: shape (n_features, d_model)
        self.W_dec = sae.W_dec.detach()
    
    def _create_steering_hook(
        self,
        config: SteeringConfig,
    ) -> Callable:
        """Create a hook function that adds scaled decoder direction."""
        
        # Pre-compute the steering vector: γ * f_max * W_dec[i]
        # W_dec has shape (n_features, d_model)
        decoder_direction = self.W_dec[config.feature_index]  # shape: (d_model,)
        steering_vector = config.gamma * config.max_feature_activation * decoder_direction
        
        def steering_hook(
            activations: Float[Tensor, "batch seq d_model"],
            hook,
        ) -> Float[Tensor, "batch seq d_model"]:
            """Hook that adds steering vector to residual stream."""
            # Apply steering: x' = x + steering_vector
            if config.start_position > 0:
                result = activations.clone()
                result[:, config.start_position:] = (
                    activations[:, config.start_position:] + steering_vector
                )
                return result
            else:
                return activations + steering_vector
        
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
            try:
                device = self.model.cfg.device
            except:
                device = self.model.device
            
            inputs = self.model.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(device)
            
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
        
        try:
            device = self.model.cfg.device
        except:
            device = self.model.device
        
        inputs = self.model.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)
        
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
                "feature_index": config.feature_index,
                "gamma": config.gamma,
                "max_feature_activation": config.max_feature_activation,
            },
        }
