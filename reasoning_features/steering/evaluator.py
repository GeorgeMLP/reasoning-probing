"""
Benchmark evaluation for steering experiments.

This module runs standardized evaluations on benchmarks with and without
feature steering to measure the impact on reasoning performance.

Steering uses the formula: x' = x + γ * f_max * W_dec[i]
Where γ is the steering strength and f_max is the maximum feature activation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import tqdm

from sae_lens import SAE, HookedSAETransformer

from ..datasets.benchmarks import get_benchmark
from .steerer import FeatureSteerer, SteeringConfig


@dataclass
class EvaluationResult:
    """Results from a benchmark evaluation."""
    
    benchmark_name: str
    condition: str  # "baseline" or "steered"
    
    # Metrics
    accuracy: float
    correct: int
    total: int
    
    # Detailed results
    predictions: list[str]
    expected: list[str]
    is_correct: list[bool]
    
    # Steering config (if applicable)
    steering_config: Optional[dict] = None
    
    # Generation parameters
    generation_params: dict = field(default_factory=dict)
    
    def save(self, path: Path):
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump({
                "benchmark_name": self.benchmark_name,
                "condition": self.condition,
                "accuracy": self.accuracy,
                "correct": self.correct,
                "total": self.total,
                "predictions": self.predictions,
                "expected": self.expected,
                "is_correct": self.is_correct,
                "steering_config": self.steering_config,
                "generation_params": self.generation_params,
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "EvaluationResult":
        """Load results from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class BenchmarkEvaluator:
    """
    Evaluates model performance on benchmarks with optional steering.
    
    This class orchestrates running benchmarks with and without feature
    steering, allowing comparison of reasoning performance.
    
    ## Steering Formula
    
    x' = x + γ * f_max * W_dec[i]
    
    Where:
    - γ: Steering strength (gamma)
    - f_max: Maximum activation of feature i
    - W_dec[i]: Decoder direction for feature i
    
    ## Usage
    
    ```python
    evaluator = BenchmarkEvaluator(model, sae, layer_index=8)
    
    # Run baseline evaluation
    baseline = evaluator.evaluate("aime24", condition="baseline")
    
    # Run steered evaluation for a single feature
    config = SteeringConfig(
        feature_index=42,
        gamma=2.0,
        max_feature_activation=15.0,
    )
    steered = evaluator.evaluate("aime24", condition="steered", steering_config=config)
    
    # Compare
    print(f"Baseline: {baseline.accuracy:.2%}")
    print(f"Steered: {steered.accuracy:.2%}")
    ```
    """
    
    def __init__(
        self,
        model: HookedSAETransformer,
        sae: SAE,
        layer_index: int = 8,
    ):
        """
        Args:
            model: The transformer model
            sae: The SAE for steering
            layer_index: Layer index for the SAE
        """
        self.model = model
        self.sae = sae
        self.layer_index = layer_index
        self.steerer = FeatureSteerer(model, sae)
    
    def evaluate(
        self,
        benchmark_name: str,
        condition: str = "baseline",
        steering_config: Optional[SteeringConfig] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,  # Low temp for more deterministic outputs
        top_p: float = 0.95,
        do_sample: bool = True,
        apply_chat_template: bool = True,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate on a benchmark.
        
        Args:
            benchmark_name: Name of benchmark ("aime24", "gpqa_diamond", "math500")
            condition: "baseline" or "steered"
            steering_config: Required if condition is "steered"
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample
            max_samples: Limit number of samples (for testing)
            verbose: Show progress bar
        
        Returns:
            EvaluationResult with metrics and detailed results
        """
        if condition == "steered" and steering_config is None:
            raise ValueError("steering_config required for steered condition")
        
        # Load benchmark
        benchmark = get_benchmark(benchmark_name)
        benchmark.load()
        
        samples = list(benchmark)
        if max_samples is not None:
            samples = samples[:max_samples]
        
        predictions = []
        expected = []
        is_correct = []
        
        iterator = samples
        if verbose:
            iterator = tqdm.tqdm(samples, desc=f"Evaluating {benchmark_name} ({condition})")
        
        for sample in iterator:
            prompt = benchmark.format_prompt(sample.question)
            
            # Generate
            if condition == "baseline":
                response = self.steerer.generate_baseline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    apply_chat_template=apply_chat_template,
                )
            else:
                response = self.steerer.generate_with_steering(
                    prompt,
                    steering_config,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    apply_chat_template=apply_chat_template,
                )
            
            # Check answer
            correct = benchmark.check_answer(response, sample.expected_answer)
            
            predictions.append(response)
            expected.append(sample.expected_answer)
            is_correct.append(correct)
        
        accuracy = sum(is_correct) / len(is_correct) if is_correct else 0.0
        
        return EvaluationResult(
            benchmark_name=benchmark_name,
            condition=condition,
            accuracy=accuracy,
            correct=sum(is_correct),
            total=len(is_correct),
            predictions=predictions,
            expected=expected,
            is_correct=is_correct,
            steering_config={
                "feature_index": steering_config.feature_index,
                "gamma": steering_config.gamma,
                "max_feature_activation": steering_config.max_feature_activation,
                "layer_index": steering_config.layer_index,
            } if steering_config else None,
            generation_params={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
            },
        )
