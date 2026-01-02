"""
Find Minimal Set of Token Injection Strategies

This script analyzes injection experiment results to identify the minimal set
of strategies needed to classify features. A strategy is redundant if:
1. It never achieves the highest Cohen's d for any feature, OR
2. When removed, no feature's classification changes from "not context_dependent" 
   to "context_dependent"

The goal is to reduce the number of strategies reported in papers while 
maintaining the same classification outcomes.

Usage:
    # Analyze specific models and datasets
    python find_minimal_strategies.py \\
        --base-dir results/cohens_d \\
        --models gemma-3-4b-it gemma-3-12b-it \\
        --datasets s1k general_inquiry_cot \\
        --output minimal_strategies_analysis.json
    
    # Analyze all models and datasets
    python find_minimal_strategies.py \\
        --base-dir results/cohens_d \\
        --output minimal_strategies_analysis.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import sys


def load_injection_results(results_path: Path) -> Dict:
    """Load injection_results.json file."""
    with open(results_path) as f:
        return json.load(f)


def get_classification_from_cohens_d(
    cohens_d: float,
    p_value: float,
    d_large: float = 0.8,
    d_medium: float = 0.5,
    d_small: float = 0.2,
    alpha: float = 0.01,
    alpha_weak: float = 0.05,
) -> str:
    """Determine classification based on Cohen's d and p-value."""
    if cohens_d >= d_large and p_value < alpha:
        return "token_driven"
    elif cohens_d >= d_medium and p_value < alpha:
        return "partially_token_driven"
    elif cohens_d >= d_small and p_value < alpha_weak:
        return "weakly_token_driven"
    else:
        return "context_dependent"


def analyze_strategy_redundancy(
    injection_results: Dict,
    verbose: bool = False
) -> Dict:
    """
    Analyze which strategies are redundant.
    
    Returns:
        Dictionary with analysis results including:
        - strategy_best_counts: How many times each strategy achieves best Cohen's d
        - strategy_usage: Which features each strategy is best for
        - minimal_set: The minimal set of strategies needed
        - redundant_strategies: Strategies that can be removed
        - classification_changes: What happens when strategies are removed
    """
    features = injection_results.get("features", [])
    config = injection_results.get("config", {})
    
    d_large = config.get("d_large", 0.8)
    d_medium = config.get("d_medium", 0.5)
    d_small = config.get("d_small", 0.2)
    alpha = config.get("alpha", 0.01)
    alpha_weak = config.get("alpha_weak", 0.05)
    
    # Track which strategy is best for each feature
    strategy_best_counts = Counter()
    strategy_usage = defaultdict(list)  # strategy -> [feature_indices]
    
    # Track each feature's best strategy and classification
    feature_info = []
    
    for feature in features:
        feature_idx = feature["feature_index"]
        strategies = feature.get("strategies", {})
        
        if not strategies:
            continue
        
        # Find best strategy (highest Cohen's d)
        best_strategy = None
        best_cohens_d = -float('inf')
        
        for strategy_name, strategy_data in strategies.items():
            cohens_d = strategy_data.get("cohens_d", 0)
            if cohens_d > best_cohens_d:
                best_cohens_d = cohens_d
                best_strategy = strategy_name
        
        if best_strategy:
            strategy_best_counts[best_strategy] += 1
            strategy_usage[best_strategy].append(feature_idx)
            
            # Get classification from best strategy
            best_p = strategies[best_strategy].get("p_value", 1.0)
            classification = get_classification_from_cohens_d(
                best_cohens_d, best_p, d_large, d_medium, d_small, alpha, alpha_weak
            )
            
            feature_info.append({
                "feature_index": feature_idx,
                "best_strategy": best_strategy,
                "best_cohens_d": best_cohens_d,
                "best_p_value": best_p,
                "classification": classification,
                "all_strategies": {
                    name: {
                        "cohens_d": data.get("cohens_d", 0),
                        "p_value": data.get("p_value", 1.0),
                    }
                    for name, data in strategies.items()
                }
            })
    
    all_strategies = set(strategy_best_counts.keys())
    
    if verbose:
        print("\nStrategy usage (times each achieves best Cohen's d):")
        for strategy, count in strategy_best_counts.most_common():
            print(f"  {strategy}: {count}")
    
    # Find minimal set: iteratively try removing strategies
    minimal_set = set(all_strategies)
    redundant = set()
    
    for strategy in sorted(all_strategies, key=lambda s: strategy_best_counts[s]):
        # Try removing this strategy
        candidate_set = minimal_set - {strategy}
        
        # Check if any feature's classification would change
        changes_classification = False
        
        for feat_info in feature_info:
            if feat_info["classification"] == "context_dependent":
                # Already context_dependent, can't get worse
                continue
            
            # Find new best strategy from candidate set
            new_best_cohens_d = -float('inf')
            new_best_p = 1.0
            
            for strat_name, strat_data in feat_info["all_strategies"].items():
                if strat_name in candidate_set:
                    if strat_data["cohens_d"] > new_best_cohens_d:
                        new_best_cohens_d = strat_data["cohens_d"]
                        new_best_p = strat_data["p_value"]
            
            # Get new classification
            new_classification = get_classification_from_cohens_d(
                new_best_cohens_d, new_best_p, d_large, d_medium, d_small, alpha, alpha_weak
            )
            
            # Check if it changed to context_dependent
            if new_classification == "context_dependent" and feat_info["classification"] != "context_dependent":
                changes_classification = True
                break
        
        if not changes_classification:
            # This strategy is redundant
            minimal_set.remove(strategy)
            redundant.add(strategy)
            if verbose:
                print(f"  Strategy '{strategy}' is redundant (can be removed)")
        elif verbose:
            print(f"  Strategy '{strategy}' is essential (removing changes classifications)")
    
    return {
        "all_strategies": sorted(all_strategies),
        "strategy_best_counts": dict(strategy_best_counts),
        "strategy_usage": {k: v for k, v in strategy_usage.items()},
        "minimal_set": sorted(minimal_set),
        "redundant_strategies": sorted(redundant),
        "n_features": len(feature_info),
        "classification_distribution": dict(Counter(f["classification"] for f in feature_info)),
    }


def find_results_files(
    base_dir: Path,
    models: List[str] = None,
    datasets: List[str] = None,
) -> List[Tuple[str, str, str, Path]]:
    """
    Find all injection_results.json files in the directory structure.
    
    Returns:
        List of (model_name, dataset_name, layer_dir, results_path) tuples
    """
    results_files = []
    
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        if models and model_name not in models:
            continue
        
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            if datasets and dataset_name not in datasets:
                continue
            
            for layer_dir in dataset_dir.iterdir():
                if not layer_dir.is_dir():
                    continue
                
                results_path = layer_dir / "injection_results.json"
                if results_path.exists():
                    results_files.append((
                        model_name,
                        dataset_name,
                        layer_dir.name,
                        results_path
                    ))
    
    return results_files


def main():
    parser = argparse.ArgumentParser(
        description="Find minimal set of token injection strategies"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/cohens_d"),
        help="Base directory containing results (default: results/cohens_d)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to analyze (default: all models in base-dir)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Datasets to analyze (default: all datasets)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for analysis results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information during analysis"
    )
    
    args = parser.parse_args()
    
    if not args.base_dir.exists():
        print(f"Error: Base directory {args.base_dir} does not exist")
        sys.exit(1)
    
    # Find all results files
    print(f"Searching for injection_results.json files in {args.base_dir}...")
    results_files = find_results_files(args.base_dir, args.models, args.datasets)
    
    if not results_files:
        print("No injection_results.json files found!")
        sys.exit(1)
    
    print(f"Found {len(results_files)} result files")
    
    # Analyze each file
    all_analyses = {}
    global_strategy_counts = Counter()
    global_minimal_sets = []
    
    for model_name, dataset_name, layer_name, results_path in results_files:
        key = f"{model_name}/{dataset_name}/{layer_name}"
        
        if args.verbose:
            print(f"\nAnalyzing {key}...")
        
        try:
            injection_results = load_injection_results(results_path)
            analysis = analyze_strategy_redundancy(injection_results, verbose=args.verbose)
            
            all_analyses[key] = analysis
            
            # Update global counts
            for strategy, count in analysis["strategy_best_counts"].items():
                global_strategy_counts[strategy] += count
            
            global_minimal_sets.append(set(analysis["minimal_set"]))
            
            if not args.verbose:
                print(f"  {key}: {len(analysis['minimal_set'])}/{len(analysis['all_strategies'])} strategies needed")
        
        except Exception as e:
            print(f"  Error analyzing {key}: {e}")
            continue
    
    # Find global minimal set (intersection of all minimal sets)
    if global_minimal_sets:
        global_minimal = set.intersection(*global_minimal_sets) if global_minimal_sets else set()
        
        # Find strategies in at least one minimal set
        global_union = set.union(*global_minimal_sets) if global_minimal_sets else set()
    else:
        global_minimal = set()
        global_union = set()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"Analyzed {len(all_analyses)} experiment results")
    print(f"\nGlobal strategy usage (total times each achieves best Cohen's d):")
    for strategy, count in global_strategy_counts.most_common():
        print(f"  {strategy}: {count}")
    
    print(f"\nStrategies in ALL minimal sets (always needed):")
    for strategy in sorted(global_minimal):
        print(f"  - {strategy}")
    
    print(f"\nStrategies in SOME minimal sets (sometimes needed):")
    for strategy in sorted(global_union - global_minimal):
        print(f"  - {strategy}")
    
    print(f"\nStrategies NEVER in minimal sets (always redundant):")
    all_strategies_seen = set(global_strategy_counts.keys())
    always_redundant = all_strategies_seen - global_union
    for strategy in sorted(always_redundant):
        print(f"  - {strategy}")
    
    # Save results
    output_data = {
        "summary": {
            "n_experiments": len(all_analyses),
            "global_strategy_counts": dict(global_strategy_counts),
            "always_needed": sorted(global_minimal),
            "sometimes_needed": sorted(global_union - global_minimal),
            "always_redundant": sorted(always_redundant),
            "recommended_minimal_set": sorted(global_union),
        },
        "per_experiment": all_analyses,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    print(f"\nFor your paper, use these {len(global_union)} strategies:")
    for strategy in sorted(global_union):
        print(f"  - {strategy}")
    print(f"\nThis ensures all features maintain their classification across all experiments.")


if __name__ == "__main__":
    main()
