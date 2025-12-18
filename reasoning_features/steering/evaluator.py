"""
Benchmark evaluation for steering experiments.

This module runs standardized evaluations on benchmarks with and without
feature steering to measure the impact on reasoning performance.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import torch
import tqdm

from sae_lens import SAE, HookedSAETransformer

from ..datasets.base import BaseBenchmark
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
    
    ## Usage
    
    ```python
    evaluator = BenchmarkEvaluator(model, sae, layer_index=8)
    
    # Run baseline evaluation
    baseline = evaluator.evaluate("aime24", condition="baseline")
    
    # Run steered evaluation
    config = SteeringConfig(feature_indices=[42, 128], multiplier=2.0)
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
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate on a benchmark.
        
        Args:
            benchmark_name: Name of benchmark ("aime24" or "gpqa_diamond")
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
                )
            else:
                response = self.steerer.generate_with_steering(
                    prompt,
                    steering_config,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
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
                "feature_indices": steering_config.feature_indices,
                "multiplier": steering_config.multiplier,
                "additive_value": steering_config.additive_value,
                "layer_index": steering_config.layer_index,
            } if steering_config else None,
            generation_params={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
            },
        )
    
    def run_steering_experiment(
        self,
        benchmark_name: str,
        feature_indices: list[int],
        multipliers: list[float] = [0.0, 0.5, 1.0, 2.0, 4.0],
        max_new_tokens: int = 512,
        max_samples: Optional[int] = None,
        save_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> dict[float, EvaluationResult]:
        """
        Run a steering experiment with multiple multiplier values.
        
        Args:
            benchmark_name: Benchmark to evaluate
            feature_indices: Features to steer
            multipliers: List of multiplier values to test
            max_new_tokens: Max tokens for generation
            max_samples: Limit samples (for testing)
            save_dir: Directory to save results
            verbose: Show progress
        
        Returns:
            Dict mapping multiplier -> EvaluationResult
        """
        results = {}
        
        for mult in multipliers:
            if mult == 1.0:
                # multiplier=1.0 is same as baseline
                condition = "baseline"
                config = None
            else:
                condition = "steered"
                config = SteeringConfig(
                    feature_indices=feature_indices,
                    multiplier=mult,
                    layer_index=self.layer_index,
                )
            
            if verbose:
                print(f"\n=== Multiplier: {mult} ===")
            
            result = self.evaluate(
                benchmark_name=benchmark_name,
                condition=condition,
                steering_config=config,
                max_new_tokens=max_new_tokens,
                max_samples=max_samples,
                verbose=verbose,
            )
            
            results[mult] = result
            
            if verbose:
                print(f"Accuracy: {result.accuracy:.2%} ({result.correct}/{result.total})")
            
            if save_dir:
                save_path = Path(save_dir) / f"result_mult_{mult:.2f}.json"
                result.save(save_path)
        
        return results
    
    def compare_feature_sets(
        self,
        benchmark_name: str,
        feature_sets: dict[str, list[int]],  # name -> feature_indices
        multiplier: float = 2.0,
        max_new_tokens: int = 512,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> dict[str, EvaluationResult]:
        """
        Compare performance when steering different feature sets.
        
        Args:
            benchmark_name: Benchmark to evaluate
            feature_sets: Dict mapping set name to feature indices
            multiplier: Steering multiplier
            max_new_tokens: Max tokens for generation
            max_samples: Limit samples
            verbose: Show progress
        
        Returns:
            Dict mapping feature set name -> EvaluationResult
        """
        results = {}
        
        # First run baseline
        if verbose:
            print("\n=== Baseline (no steering) ===")
        
        baseline = self.evaluate(
            benchmark_name=benchmark_name,
            condition="baseline",
            max_new_tokens=max_new_tokens,
            max_samples=max_samples,
            verbose=verbose,
        )
        results["baseline"] = baseline
        
        if verbose:
            print(f"Accuracy: {baseline.accuracy:.2%}")
        
        # Run each feature set
        for name, feature_indices in feature_sets.items():
            if verbose:
                print(f"\n=== Feature set: {name} ({len(feature_indices)} features) ===")
            
            config = SteeringConfig(
                feature_indices=feature_indices,
                multiplier=multiplier,
                layer_index=self.layer_index,
            )
            
            result = self.evaluate(
                benchmark_name=benchmark_name,
                condition="steered",
                steering_config=config,
                max_new_tokens=max_new_tokens,
                max_samples=max_samples,
                verbose=verbose,
            )
            results[name] = result
            
            if verbose:
                delta = result.accuracy - baseline.accuracy
                print(f"Accuracy: {result.accuracy:.2%} (Δ={delta:+.2%})")
        
        return results


def load_model_and_sae(
    model_name: str = "google/gemma-2-2b",
    sae_name: str = "gemma-scope-2b-pt-res-canonical",
    sae_id_format: str = "layer_{layer}/width_16k/canonical",
    layer_index: int = 8,
    device: str = "cuda",
) -> tuple[HookedSAETransformer, SAE]:
    """
    Convenience function to load model and SAE.
    
    Returns:
        Tuple of (model, sae)
    """
    print(f"Loading model: {model_name}")
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        dtype=torch.bfloat16,
    )
    
    print(f"Loading SAE for layer {layer_index}")
    sae_id = sae_id_format.format(layer=layer_index)
    sae = SAE.from_pretrained(
        release=sae_name,
        sae_id=sae_id,
        device=device,
    )
    
    return model, sae

