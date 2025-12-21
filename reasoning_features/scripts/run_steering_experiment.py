"""
Run steering experiments to test whether amplifying reasoning features improves performance.

This script evaluates model performance on benchmarks with and without steering
of identified reasoning features, testing the hypothesis that these features
capture genuine reasoning vs. shallow token cues.

## Hypothesis

If "reasoning features" capture actual reasoning ability:
- Amplifying them (multiplier > 1) should improve benchmark performance
- Suppressing them (multiplier < 1) should decrease performance

If "reasoning features" are spurious (token correlations):
- Amplifying them will likely decrease performance
- The model may produce text that "looks like" reasoning but isn't correct

## Usage

```bash
# Run with detected features
python run_steering_experiment.py \\
    --features-file results/layer8/reasoning_features.json \\
    --benchmark aime24

# Run with specific features
python run_steering_experiment.py \\
    --feature-indices 42 128 256 \\
    --benchmark gpqa_diamond \\
    --multipliers 0.5 1.0 2.0 4.0

# Quick test
python run_steering_experiment.py \\
    --features-file results/layer8/reasoning_features.json \\
    --benchmark aime24 \\
    --max-samples 5
```
"""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reasoning_features.steering import BenchmarkEvaluator
from reasoning_features.steering.evaluator import load_model_and_sae


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run steering experiments on benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Model configuration
    parser.add_argument(
        "--model-name",
        default="google/gemma-2-2b",
        help="HuggingFace model name (default: google/gemma-2-2b)",
    )
    parser.add_argument(
        "--sae-name",
        default="gemma-scope-2b-pt-res-canonical",
        help="SAE release name (default: gemma-scope-2b-pt-res-canonical)",
    )
    parser.add_argument(
        "--sae-id-format",
        default="layer_{layer}/width_16k/canonical",
        help="SAE ID format string",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Layer index for steering (default: 8)",
    )
    
    # Feature specification
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument(
        "--features-file",
        type=Path,
        help="JSON file with reasoning features (from find_reasoning_features.py)",
    )
    feature_group.add_argument(
        "--feature-indices",
        type=int,
        nargs="+",
        help="Specific feature indices to steer",
    )
    
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=20,
        help="Number of top features to use from file (default: 20)",
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--benchmark",
        choices=["aime24", "gpqa_diamond"],
        required=True,
        help="Benchmark to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of benchmark samples (for testing)",
    )
    
    # Steering configuration
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 4.0],
        help="Steering multipliers to test (default: 0.0 0.5 1.0 2.0 4.0)",
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum tokens to generate (default: 20)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter (default: 0.95)",
    )
    
    # Output
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save results",
    )
    
    # Runtime
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    
    return parser.parse_args()


def load_features_from_file(file_path: Path, top_k: int) -> list[int]:
    """Load feature indices from a reasoning features JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    
    feature_indices = data.get("feature_indices", [])
    
    if not feature_indices and "features" in data:
        feature_indices = [f["feature_index"] for f in data["features"]]
    
    return feature_indices[:top_k]


def main():
    args = parse_args()
    
    print("=" * 60)
    print("STEERING EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"SAE: {args.sae_name}")
    print(f"Layer: {args.layer}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Multipliers: {args.multipliers}")
    print("=" * 60)
    
    # Load feature indices
    if args.features_file:
        print(f"\nLoading features from {args.features_file}")
        feature_indices = load_features_from_file(
            args.features_file, args.top_k_features
        )
    else:
        feature_indices = args.feature_indices
    
    print(f"Steering {len(feature_indices)} features: {feature_indices[:10]}...")
    
    # Create save directory
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and SAE
    model, sae = load_model_and_sae(
        model_name=args.model_name,
        sae_name=args.sae_name,
        sae_id_format=args.sae_id_format,
        layer_index=args.layer,
        device=args.device,
    )
    
    # Create evaluator
    evaluator = BenchmarkEvaluator(model, sae, layer_index=args.layer)
    
    # Run experiment
    print(f"\n--- Running Steering Experiment ---")
    results = evaluator.run_steering_experiment(
        benchmark_name=args.benchmark,
        feature_indices=feature_indices,
        multipliers=args.multipliers,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
        save_dir=args.save_dir,
        verbose=True,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Multiplier':>12} {'Accuracy':>12} {'Correct':>12} {'Delta':>12}")
    print("-" * 50)
    
    baseline_acc = results.get(1.0, results.get(list(results.keys())[0])).accuracy
    
    for mult, result in sorted(results.items()):
        delta = result.accuracy - baseline_acc
        delta_str = f"{delta:+.2%}" if mult != 1.0 else "baseline"
        print(
            f"{mult:>12.1f} "
            f"{result.accuracy:>12.2%} "
            f"{result.correct:>8}/{result.total:<3} "
            f"{delta_str:>12}"
        )
    
    # Analysis
    print("\n--- Analysis ---")
    
    # Find best and worst multipliers
    best_mult = max(results.keys(), key=lambda m: results[m].accuracy)
    worst_mult = min(results.keys(), key=lambda m: results[m].accuracy)
    
    print(f"Best multiplier: {best_mult:.1f} (accuracy: {results[best_mult].accuracy:.2%})")
    print(f"Worst multiplier: {worst_mult:.1f} (accuracy: {results[worst_mult].accuracy:.2%})")
    
    # Interpret results
    amplified_results = [r for m, r in results.items() if m > 1.0]
    suppressed_results = [r for m, r in results.items() if m < 1.0 and m > 0]
    
    if amplified_results:
        avg_amplified = sum(r.accuracy for r in amplified_results) / len(amplified_results)
        print(f"\nAverage accuracy with amplification (mult > 1): {avg_amplified:.2%}")
        
        if avg_amplified > baseline_acc + 0.02:
            print("✓ Amplifying features IMPROVED performance")
            print("  → Features may capture genuine reasoning")
        elif avg_amplified < baseline_acc - 0.02:
            print("✗ Amplifying features DECREASED performance")
            print("  → Features likely capture spurious correlations (token cues)")
        else:
            print("○ Amplifying features had minimal effect")
            print("  → Features may not be strongly related to task performance")
    
    if suppressed_results:
        avg_suppressed = sum(r.accuracy for r in suppressed_results) / len(suppressed_results)
        print(f"\nAverage accuracy with suppression (0 < mult < 1): {avg_suppressed:.2%}")
    
    # Save summary
    if args.save_dir:
        summary_path = args.save_dir / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "config": {
                        "model_name": args.model_name,
                        "sae_name": args.sae_name,
                        "layer": args.layer,
                        "benchmark": args.benchmark,
                        "feature_indices": feature_indices,
                        "multipliers": args.multipliers,
                    },
                    "results": {
                        str(mult): {
                            "accuracy": result.accuracy,
                            "correct": result.correct,
                            "total": result.total,
                        }
                        for mult, result in results.items()
                    },
                    "analysis": {
                        "baseline_accuracy": baseline_acc,
                        "best_multiplier": best_mult,
                        "best_accuracy": results[best_mult].accuracy,
                        "worst_multiplier": worst_mult,
                        "worst_accuracy": results[worst_mult].accuracy,
                    },
                },
                f,
                indent=2,
            )
        print(f"\nSaved summary to {summary_path}")
    
    return results


if __name__ == "__main__":
    main()
