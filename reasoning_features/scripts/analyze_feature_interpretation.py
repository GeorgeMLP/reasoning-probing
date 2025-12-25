"""
LLM-guided feature interpretation and counterexample discovery.

This script uses an intelligent LLM to analyze SAE features and discover counterexamples
that either:
- Are NOT reasoning but activate the feature (false positives)
- ARE reasoning but do NOT activate the feature (false negatives)

The script iteratively:
1. Analyzes high-activation examples to form hypotheses
2. Generates counterexample candidates
3. Tests them against the actual model
4. Refines interpretations based on results

Usage:
    # Mode 1: Analyze context-dependent features from injection results
    python analyze_feature_interpretation.py \
        --injection-results results/layer16/injection_results.json \
        --mode context_dependent
    
    # Mode 2: Analyze all reasoning features
    python analyze_feature_interpretation.py \
        --reasoning-features results/layer16/reasoning_features.json \
        --mode all_reasoning
    
    # Mode 3: Analyze specific features
    python analyze_feature_interpretation.py \
        --feature-indices 715 494 13302 \
        --token-analysis results/layer16/token_analysis.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time

import requests
import torch
import numpy as np
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class CounterExample:
    """A counterexample with its activation result."""
    text: str
    category: str  # 'false_positive' or 'false_negative'
    expected_reasoning: bool
    max_activation: float
    is_valid_counterexample: bool
    explanation: str


@dataclass 
class FeatureInterpretation:
    """Complete interpretation of a feature."""
    feature_index: int
    initial_hypothesis: str
    refined_interpretation: str
    activates_on: list[str]
    does_not_activate_on: list[str]
    false_positive_examples: list[dict]
    false_negative_examples: list[dict]
    confidence: str
    is_genuine_reasoning_feature: bool
    summary: str


class LLMClient:
    """Client for calling OpenRouter API."""
    
    def __init__(self, api_key: str, model: str = "google/gemini-3-pro-preview"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Send a chat request and return the response."""
        response = requests.post(
            url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }),
            timeout=120,
        )
        
        result = response.json()
        if "error" in result:
            raise Exception(f"API Error: {result['error']}")
        
        return result["choices"][0]["message"]["content"]


class FeatureAnalyzer:
    """Analyzes SAE features using LLM guidance and model activation testing."""
    
    def __init__(
        self,
        model,
        sae,
        tokenizer,
        llm_client: LLMClient,
        layer: int,
        device: str = "cuda",
    ):
        self.model = model
        self.sae = sae
        self.tokenizer = tokenizer
        self.llm = llm_client
        self.layer = layer
        self.device = device
    
    def get_activation(self, text: str, feature_index: int) -> tuple[float, float, list]:
        """Get feature activation for a text."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                inputs["input_ids"],
                names_filter=[f"blocks.{self.layer}.hook_resid_post"],
            )
            hidden = cache[f"blocks.{self.layer}.hook_resid_post"]
            sae_acts = self.sae.encode(hidden)
            acts = sae_acts[0, :, feature_index].cpu().numpy()
        
        tokens = [self.tokenizer.decode([t]) for t in inputs["input_ids"][0].tolist()]
        
        # Get top activating tokens
        top_indices = np.argsort(acts)[-5:][::-1]
        top_tokens = [(tokens[i], float(acts[i])) for i in top_indices if acts[i] > 0]
        
        return float(acts.max()), float(acts.mean()), top_tokens
    
    def collect_activation_examples(
        self, 
        feature_index: int,
        reasoning_texts: list[str],
        n_examples: int = 10,
    ) -> list[dict]:
        """Collect examples of high activation with context."""
        examples = []
        
        for text in reasoning_texts[:100]:
            max_act, mean_act, top_tokens = self.get_activation(text, feature_index)
            
            if max_act > 5:
                examples.append({
                    "text": text[:500],
                    "max_activation": max_act,
                    "mean_activation": mean_act,
                    "top_tokens": top_tokens[:5],
                })
        
        # Sort by activation and return top N
        examples.sort(key=lambda x: x["max_activation"], reverse=True)
        return examples[:n_examples]
    
    def generate_hypothesis(
        self, 
        feature_index: int,
        high_activation_examples: list[dict],
        top_tokens: list[str],
    ) -> str:
        """Use LLM to generate an initial hypothesis about what the feature detects."""
        
        examples_text = "\n\n".join([
            f"Example {i+1} (max activation: {ex['max_activation']:.1f}):\n"
            f"Text: {ex['text'][:300]}...\n"
            f"Top activating tokens: {ex['top_tokens']}"
            for i, ex in enumerate(high_activation_examples[:5])
        ])
        
        prompt = f"""You are analyzing an SAE (Sparse Autoencoder) feature from a language model.

This feature was identified as potentially detecting "reasoning" in text, but we need to determine what it ACTUALLY detects.

## Top Tokens (by mean activation)
{', '.join(top_tokens[:15])}

## High Activation Examples
{examples_text}

## Task
Based on these examples, generate a hypothesis about what linguistic pattern this feature detects.

Consider:
1. Is it detecting specific vocabulary/topics?
2. Is it detecting a writing style (formal, elaborate, simple)?
3. Is it detecting discourse structure (reasoning, narrative, explanatory)?
4. Is it detecting something else entirely?

Provide your hypothesis in 2-3 sentences. Be specific and testable.
"""
        
        response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.3)
        return response.strip()
    
    def generate_counterexamples(
        self,
        feature_index: int,
        hypothesis: str,
        high_activation_examples: list[dict],
        category: str,  # 'false_positive' or 'false_negative'
    ) -> list[str]:
        """Use LLM to generate counterexample candidates."""
        
        if category == "false_positive":
            task = """Generate 5 text examples that:
- Are clearly NOT reasoning/problem-solving/deliberation
- But SHOULD activate this feature according to the hypothesis
- Could be: recipes, product reviews, sports commentary, narrative fiction, etc.

Each example should be 50-100 words and aim to "fool" the feature."""
        else:
            task = """Generate 5 text examples that:
- ARE clearly reasoning/problem-solving/deliberation
- But SHOULD NOT activate this feature according to the hypothesis
- Could be: simple reasoning, informal problem-solving, etc.

Each example should be 50-100 words and aim to show the feature misses real reasoning."""
        
        prompt = f"""You are testing a hypothesis about what an SAE feature detects.

## Hypothesis
{hypothesis}

## Example of high-activation text
{high_activation_examples[0]['text'][:300]}...

## Task
{task}

Format your response as a JSON array of 5 strings, each containing one example text.
Only output the JSON array, nothing else."""
        
        response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.8)
        
        # Parse JSON response
        try:
            # Clean up response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            examples = json.loads(response)
            return examples[:5]
        except json.JSONDecodeError:
            print(f"Warning: Could not parse LLM response as JSON")
            return []
    
    def test_counterexamples(
        self,
        feature_index: int,
        candidates: list[str],
        category: str,
        reference_max_activation: float,
        threshold_ratio: float = 0.5,
    ) -> list[CounterExample]:
        """Test counterexample candidates against the model.
        
        Args:
            feature_index: The feature to test
            candidates: List of candidate texts
            category: 'false_positive' or 'false_negative'
            reference_max_activation: The max activation from high-activation examples
            threshold_ratio: Percentage of reference_max to use as threshold (0-1)
        """
        results = []
        activation_threshold = reference_max_activation * threshold_ratio
        
        for text in candidates:
            max_act, mean_act, top_tokens = self.get_activation(text, feature_index)
            
            if category == "false_positive":
                # Non-reasoning text that activates = valid counterexample
                is_valid = max_act > activation_threshold
                expected_reasoning = False
            else:
                # Reasoning text that doesn't activate = valid counterexample
                is_valid = max_act < activation_threshold * 0.5
                expected_reasoning = True
            
            results.append(CounterExample(
                text=text,
                category=category,
                expected_reasoning=expected_reasoning,
                max_activation=max_act,
                is_valid_counterexample=is_valid,
                explanation=f"Max activation: {max_act:.2f}, threshold: {activation_threshold:.2f} ({threshold_ratio*100:.0f}% of {reference_max_activation:.2f})",
            ))
        
        return results
    
    def refine_interpretation(
        self,
        feature_index: int,
        initial_hypothesis: str,
        valid_counterexamples: list[CounterExample],
        high_activation_examples: list[dict],
    ) -> FeatureInterpretation:
        """Refine the interpretation based on counterexample results."""
        
        fp_examples = [ce for ce in valid_counterexamples if ce.category == "false_positive" and ce.is_valid_counterexample]
        fn_examples = [ce for ce in valid_counterexamples if ce.category == "false_negative" and ce.is_valid_counterexample]
        
        counterexample_summary = ""
        if fp_examples:
            counterexample_summary += "\n\nFalse Positives (non-reasoning that activated):\n"
            for ce in fp_examples[:3]:
                counterexample_summary += f"- {ce.text[:150]}... (activation: {ce.max_activation:.1f})\n"
        
        if fn_examples:
            counterexample_summary += "\n\nFalse Negatives (reasoning that didn't activate):\n"
            for ce in fn_examples[:3]:
                counterexample_summary += f"- {ce.text[:150]}... (activation: {ce.max_activation:.1f})\n"
        
        prompt = f"""Based on testing, we need to refine our understanding of this SAE feature.

## Initial Hypothesis
{initial_hypothesis}

## Test Results
Found {len(fp_examples)} non-reasoning texts that activated the feature (false positives)
Found {len(fn_examples)} reasoning texts that didn't activate the feature (false negatives)
{counterexample_summary}

## Original High-Activation Examples (for reference)
{high_activation_examples[0]['text'][:200]}...

## Task
1. Provide a REFINED interpretation of what this feature actually detects (2-3 sentences)
2. List 3-5 types of content this feature DOES activate on
3. List 3-5 types of content this feature does NOT activate on
4. Rate your confidence: HIGH, MEDIUM, or LOW
5. Is this a genuine "reasoning" feature? YES or NO
6. One-sentence summary

Format as JSON:
{{
    "refined_interpretation": "...",
    "activates_on": ["...", "..."],
    "does_not_activate_on": ["...", "..."],
    "confidence": "HIGH/MEDIUM/LOW",
    "is_genuine_reasoning_feature": true/false,
    "summary": "..."
}}"""
        
        response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.3)
        
        # Parse response
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {
                "refined_interpretation": initial_hypothesis,
                "activates_on": ["Unknown"],
                "does_not_activate_on": ["Unknown"],
                "confidence": "LOW",
                "is_genuine_reasoning_feature": False,
                "summary": "Could not refine interpretation",
            }
        
        return FeatureInterpretation(
            feature_index=feature_index,
            initial_hypothesis=initial_hypothesis,
            refined_interpretation=result.get("refined_interpretation", initial_hypothesis),
            activates_on=result.get("activates_on", []),
            does_not_activate_on=result.get("does_not_activate_on", []),
            false_positive_examples=[asdict(ce) for ce in fp_examples],
            false_negative_examples=[asdict(ce) for ce in fn_examples],
            confidence=result.get("confidence", "LOW"),
            is_genuine_reasoning_feature=result.get("is_genuine_reasoning_feature", False),
            summary=result.get("summary", ""),
        )
    
    def analyze_feature(
        self,
        feature_index: int,
        reasoning_texts: list[str],
        top_tokens: list[str],
        max_iterations: int = 2,
        min_false_positives: int = 3,
        min_false_negatives: int = 3,
        threshold_ratio: float = 0.5,
    ) -> FeatureInterpretation:
        """Complete analysis of a single feature."""
        print(f"\n{'='*60}")
        print(f"Analyzing Feature {feature_index}")
        print(f"{'='*60}")
        
        # Step 1: Collect high-activation examples
        print("Step 1: Collecting high-activation examples...")
        examples = self.collect_activation_examples(feature_index, reasoning_texts)
        
        if not examples:
            print("  No high-activation examples found!")
            return FeatureInterpretation(
                feature_index=feature_index,
                initial_hypothesis="No high-activation examples found",
                refined_interpretation="Feature does not activate on reasoning texts",
                activates_on=[],
                does_not_activate_on=["reasoning texts"],
                false_positive_examples=[],
                false_negative_examples=[],
                confidence="LOW",
                is_genuine_reasoning_feature=False,
                summary="Feature does not activate on the reasoning dataset",
            )
        
        # Compute reference max activation from collected examples
        reference_max_activation = max(ex["max_activation"] for ex in examples)
        activation_threshold = reference_max_activation * threshold_ratio
        
        print(f"  Found {len(examples)} high-activation examples")
        print(f"  Reference max activation: {reference_max_activation:.2f}")
        print(f"  Threshold: {activation_threshold:.2f} ({threshold_ratio*100:.0f}% of max)")
        
        # Step 2: Generate initial hypothesis
        print("Step 2: Generating hypothesis...")
        hypothesis = self.generate_hypothesis(feature_index, examples, top_tokens)
        print(f"  Hypothesis: {hypothesis[:200]}...")
        
        all_counterexamples = []
        
        for iteration in range(max_iterations):
            # Count current valid counterexamples
            total_valid_fp = sum(1 for ce in all_counterexamples 
                                 if ce.category == "false_positive" and ce.is_valid_counterexample)
            total_valid_fn = sum(1 for ce in all_counterexamples 
                                 if ce.category == "false_negative" and ce.is_valid_counterexample)
            
            # Check for early stopping
            if total_valid_fp >= min_false_positives and total_valid_fn >= min_false_negatives:
                print(f"\nEarly stopping: Found {total_valid_fp} false positives and {total_valid_fn} false negatives")
                print(f"  (Required: {min_false_positives} FP and {min_false_negatives} FN)")
                break
            
            print(f"\nIteration {iteration + 1}/{max_iterations}")
            print(f"  Current: {total_valid_fp}/{min_false_positives} FP, {total_valid_fn}/{min_false_negatives} FN")
            
            # Step 3: Generate counterexamples
            print("  Generating false positive candidates...")
            fp_candidates = self.generate_counterexamples(
                feature_index, hypothesis, examples, "false_positive"
            )
            
            print("  Generating false negative candidates...")
            fn_candidates = self.generate_counterexamples(
                feature_index, hypothesis, examples, "false_negative"
            )
            
            # Step 4: Test counterexamples
            print(f"  Testing {len(fp_candidates)} false positive candidates...")
            fp_results = self.test_counterexamples(
                feature_index, fp_candidates, "false_positive",
                reference_max_activation, threshold_ratio
            )
            valid_fp = sum(1 for ce in fp_results if ce.is_valid_counterexample)
            print(f"    Valid counterexamples: {valid_fp}/{len(fp_results)}")
            
            print(f"  Testing {len(fn_candidates)} false negative candidates...")
            fn_results = self.test_counterexamples(
                feature_index, fn_candidates, "false_negative",
                reference_max_activation, threshold_ratio
            )
            valid_fn = sum(1 for ce in fn_results if ce.is_valid_counterexample)
            print(f"    Valid counterexamples: {valid_fn}/{len(fn_results)}")
            
            all_counterexamples.extend(fp_results)
            all_counterexamples.extend(fn_results)
        
        # Step 5: Refine interpretation
        print("\nRefining interpretation...")
        interpretation = self.refine_interpretation(
            feature_index, hypothesis, all_counterexamples, examples
        )
        
        print(f"\n{'='*60}")
        print(f"RESULT for Feature {feature_index}")
        print(f"{'='*60}")
        print(f"Interpretation: {interpretation.refined_interpretation}")
        print(f"Is genuine reasoning feature: {interpretation.is_genuine_reasoning_feature}")
        print(f"Confidence: {interpretation.confidence}")
        print(f"False positives found: {len(interpretation.false_positive_examples)}")
        print(f"False negatives found: {len(interpretation.false_negative_examples)}")
        
        return interpretation


def load_feature_indices(args) -> list[int]:
    """Load feature indices based on mode."""
    
    if args.feature_indices:
        return args.feature_indices
    
    if args.mode == "context_dependent" and args.injection_results:
        with open(args.injection_results) as f:
            data = json.load(f)
        return [
            f["feature_index"] for f in data.get("features", [])
            if f.get("classification") == "context_dependent"
        ]
    
    if args.mode == "all_reasoning" and args.reasoning_features:
        with open(args.reasoning_features) as f:
            data = json.load(f)
        return data.get("feature_indices", [])
    
    raise ValueError("Must specify either --feature-indices or appropriate mode with data files")


def load_token_data(token_analysis_path: Path, feature_index: int) -> list[str]:
    """Load top tokens for a feature."""
    with open(token_analysis_path) as f:
        data = json.load(f)
    
    for feat in data.get("features", []):
        if feat["feature_index"] == feature_index:
            return [t["token_str"] for t in feat.get("top_tokens", [])]
    return []


def main():
    parser = argparse.ArgumentParser(
        description="LLM-guided feature interpretation and counterexample discovery"
    )
    
    # Input modes
    parser.add_argument("--injection-results", type=Path,
                        help="Path to injection_results.json")
    parser.add_argument("--reasoning-features", type=Path,
                        help="Path to reasoning_features.json")
    parser.add_argument("--token-analysis", type=Path, required=True,
                        help="Path to token_analysis.json")
    parser.add_argument("--feature-indices", type=int, nargs="+",
                        help="Specific feature indices to analyze")
    parser.add_argument("--mode", choices=["context_dependent", "all_reasoning", "specific"],
                        default="specific",
                        help="Analysis mode")
    
    # Model configuration
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--model-name", default="google/gemma-2-9b")
    parser.add_argument("--sae-name", default="gemma-scope-9b-pt-res-canonical")
    parser.add_argument("--sae-id-format", default="layer_{layer}/width_16k/canonical")
    parser.add_argument("--device", default="cuda")
    
    # LLM configuration
    parser.add_argument("--llm-model", default="google/gemini-3-pro-preview",
                        help="OpenRouter model to use for analysis")
    
    # Analysis configuration
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Maximum counterexample generation iterations per feature")
    parser.add_argument("--min-false-positives", type=int, default=3,
                        help="Minimum false positives to find before stopping")
    parser.add_argument("--min-false-negatives", type=int, default=3,
                        help="Minimum false negatives to find before stopping")
    parser.add_argument("--threshold-ratio", type=float, default=0.5,
                        help="Activation threshold as ratio of max activation (0-1, default: 0.5)")
    parser.add_argument("--max-features", type=int, default=20,
                        help="Maximum number of features to analyze")
    
    # Dataset
    parser.add_argument("--reasoning-dataset", default="general_inquiry_cot",
                        choices=["s1k", "general_inquiry_cot"])
    
    # Output
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    # Load feature indices
    feature_indices = load_feature_indices(args)[:args.max_features]
    print(f"Analyzing {len(feature_indices)} features: {feature_indices}")
    
    # Load model and SAE
    print("\nLoading model and SAE...")
    from sae_lens import SAE, HookedSAETransformer
    
    model = HookedSAETransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        dtype=torch.bfloat16,
    )
    
    sae_id = args.sae_id_format.format(layer=args.layer)
    sae = SAE.from_pretrained(
        release=args.sae_name,
        sae_id=sae_id,
        device=args.device,
    )
    
    tokenizer = model.tokenizer
    print("Loaded!")
    
    # Load reasoning texts
    print("\nLoading reasoning dataset...")
    from datasets import load_dataset
    
    reasoning_texts = []
    if args.reasoning_dataset == "s1k":
        ds = load_dataset("simplescaling/s1K-1.1", split="train")
        for row in ds:
            for key in ["gemini_thinking_trajectory", "deepseek_thinking_trajectory"]:
                if row.get(key):
                    reasoning_texts.append(row[key][:1000])
                    if len(reasoning_texts) >= 200:
                        break
            if len(reasoning_texts) >= 200:
                break
    else:
        ds = load_dataset("moremilk/General_Inquiry_Thinking-Chain-Of-Thought", split="train")
        for row in ds:
            metadata = row.get("metadata", {})
            if isinstance(metadata, dict):
                text = metadata.get("reasoning", "")
                if text:
                    text = text.replace("<think>", "").replace("</think>", "").strip()
                    reasoning_texts.append(text[:1000])
                    if len(reasoning_texts) >= 200:
                        break
    
    print(f"Loaded {len(reasoning_texts)} reasoning texts")
    
    # Initialize analyzer
    llm_client = LLMClient(api_key, args.llm_model)
    analyzer = FeatureAnalyzer(model, sae, tokenizer, llm_client, args.layer, args.device)
    
    # Analyze each feature
    results = []
    
    for feature_index in feature_indices:
        try:
            top_tokens = load_token_data(args.token_analysis, feature_index)
            interpretation = analyzer.analyze_feature(
                feature_index,
                reasoning_texts,
                top_tokens,
                max_iterations=args.max_iterations,
                min_false_positives=args.min_false_positives,
                min_false_negatives=args.min_false_negatives,
                threshold_ratio=args.threshold_ratio,
            )
            results.append(asdict(interpretation))
            
            # Build summary
            summary = {
                "total_features_analyzed": len(results),
                "genuine_reasoning_features": sum(1 for r in results if r.get("is_genuine_reasoning_feature")),
                "non_reasoning_features": sum(1 for r in results if not r.get("is_genuine_reasoning_feature")),
                "high_confidence": sum(1 for r in results if r.get("confidence") == "HIGH"),
                "medium_confidence": sum(1 for r in results if r.get("confidence") == "MEDIUM"),
                "low_confidence": sum(1 for r in results if r.get("confidence") == "LOW"),
                "total_false_positives": sum(len(r.get("false_positive_examples", [])) for r in results),
                "total_false_negatives": sum(len(r.get("false_negative_examples", [])) for r in results),
            }
            
            # Save intermediate results
            with open(args.output, "w") as f:
                json.dump({
                    "summary": summary,
                    "config": {
                        "layer": args.layer,
                        "model": args.model_name,
                        "llm_model": args.llm_model,
                        "max_iterations": args.max_iterations,
                        "min_false_positives": args.min_false_positives,
                        "min_false_negatives": args.min_false_negatives,
                        "threshold_ratio": args.threshold_ratio,
                    },
                    "features": results,
                }, f, indent=2)
            
            # Rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"Error analyzing feature {feature_index}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    final_summary = {
        "total_features_analyzed": len(results),
        "genuine_reasoning_features": sum(1 for r in results if r.get("is_genuine_reasoning_feature")),
        "non_reasoning_features": sum(1 for r in results if not r.get("is_genuine_reasoning_feature")),
        "high_confidence": sum(1 for r in results if r.get("confidence") == "HIGH"),
        "medium_confidence": sum(1 for r in results if r.get("confidence") == "MEDIUM"),
        "low_confidence": sum(1 for r in results if r.get("confidence") == "LOW"),
        "total_false_positives": sum(len(r.get("false_positive_examples", [])) for r in results),
        "total_false_negatives": sum(len(r.get("false_negative_examples", [])) for r in results),
    }
    
    print(f"Total features analyzed: {final_summary['total_features_analyzed']}")
    print(f"Genuine reasoning features: {final_summary['genuine_reasoning_features']}")
    print(f"Non-reasoning features: {final_summary['non_reasoning_features']}")
    print(f"Confidence: {final_summary['high_confidence']} HIGH, {final_summary['medium_confidence']} MEDIUM, {final_summary['low_confidence']} LOW")
    print(f"Total counterexamples: {final_summary['total_false_positives']} FP, {final_summary['total_false_negatives']} FN")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
