# SAE Reasoning Features Analysis

Investigating whether **Sparse Autoencoders (SAEs)** capture higher-level reasoning features, or merely learn spurious correlations with reasoning-associated tokens.

## Research Question

**Do SAE features that activate on reasoning text capture actual reasoning behavior, or are they proxies for shallow token-level cues?**

We hypothesize that:
1. Features showing high correlation with reasoning text primarily respond to **token-level cues** (e.g., "Let me think...", mathematical notation, hedging words)
2. These features do **not** capture the underlying reasoning process that leads to correct answers
3. Steering (amplifying) these features will **not** improve reasoning performance—and may even decrease it

## Framework Overview

```
reasoning_features/
├── datasets/           # Dataset loaders
│   ├── pile.py         # Non-reasoning text (Pile)
│   ├── reasoning.py    # Reasoning datasets (s1K, General Inquiry CoT)
│   └── benchmarks.py   # Evaluation benchmarks (AIME24, GPQA Diamond, MATH-500)
├── features/           # Feature analysis
│   ├── collector.py    # SAE activation collection
│   ├── detector.py     # Reasoning feature detection
│   ├── tokens.py       # Token dependency analysis
│   └── selection.py    # Feature selection for steering
├── steering/           # Intervention experiments
│   ├── steerer.py      # Activation steering
│   └── evaluator.py    # Benchmark evaluation
├── utils/              # Utility functions
│   └── llm_judge.py    # LLM-based answer equivalence checking
└── scripts/            # Main experiment scripts
    ├── find_reasoning_features.py        # Find reasoning-correlated features
    ├── run_token_injection_experiment.py # Causal token injection test ⭐
    ├── run_steering_experiment.py        # Steering benchmark evaluation
    └── plot_results.py                   # Visualization
```

## Experiment 1: Finding Reasoning Features

Identify SAE features that show differential activation between reasoning and non-reasoning text.

### Metrics for Feature Detection

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **ROC-AUC** | Binary classification performance | ≥ 0.6 |
| **Cohen's d** | Standardized effect size | ≥ 0.3 |
| **Mann-Whitney U** | Non-parametric distribution test | p ≤ 0.01 (Bonferroni) |
| **Reasoning Score** | Weighted composite metric | Ranked |

### Token Dependency Analysis

For each detected feature, we analyze which tokens drive its activation:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Mean Activation** | Average activation when token appears | Higher = stronger trigger |
| **PMI** | Pointwise Mutual Information | Co-occurrence strength |
| **Token Concentration** | % of high activations from top-k tokens | >50% = shallow cue reliance |
| **Reasoning Specificity** | Activation ratio (reasoning/non-reasoning) | >1.5x = reasoning-specific |

### Usage

```bash
# Activate environment
conda activate probing

# Run feature detection
python reasoning_features/scripts/find_reasoning_features.py \
    --layer 8 \
    --reasoning-dataset s1k \
    --reasoning-samples 500 \
    --nonreasoning-samples 500 \
    --save-dir results/layer8

# Options
--model-name        # Model to analyze (default: google/gemma-2-2b)
--sae-name          # SAE release (default: gemma-scope-2b-pt-res-canonical)
--layer             # Layer index to analyze
--reasoning-dataset # s1k, general_inquiry_cot, or combined
--min-auc           # Minimum ROC-AUC threshold (default: 0.6)
--min-effect-size   # Minimum Cohen's d (default: 0.3)
--top-k-features    # Number of top features to analyze
```

### Output

The results are organized in the following directory structure:
```
results/{setting}/{model}/{dataset}/layer{N}/
├── activations.pt          # Cached activations for reuse
├── feature_stats.json      # Statistics for all features
├── reasoning_features.json # Detected reasoning features
├── token_analysis.json     # Token dependency analysis
├── injection_results.json  # Token injection experiment results
└── {benchmark}/            # Steering experiment results per benchmark
    ├── experiment_summary.json
    └── feature_{index}/    # Per-feature results
        ├── feature_summary.json
        └── result_gamma_{value}.json
```

Example: `results/initial-setting/gemma-2-9b/s1k/layer12/`

## Experiment 2: Token Injection (Causal Test) ⭐

**The key causal experiment**: If a feature is truly a "token detector", injecting those tokens into non-reasoning text should activate the feature.

### Experimental Design

1. Take non-reasoning text samples
2. Inject feature's top tokens using various strategies
3. Measure activation before and after injection
4. Compare to activation on actual reasoning text

**Injection Strategies:**

*Simple strategies:*
- `prepend`: Add tokens at the beginning
- `append`: Add tokens at the end  
- `intersperse`: Distribute throughout the text
- `replace`: Replace random words

*Contextual strategies* (for features sensitive to token sequences):
- `bigram_before`: Inject [context, token] pairs (e.g., "to identify")
- `bigram_after`: Inject [token, context] pairs (e.g., "need to")
- `trigram`: Inject [before, token, after] triplets
- `comma_list`: Inject as comma-separated list

Contextual strategies are crucial for detecting features that activate on token combinations rather than individual tokens, such as:
- Verbs that only activate after "to"
- Pronouns that only activate before specific modals
- Items in enumerated lists

### Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Cohen's d** | Standardized effect size (injected vs baseline) | High = token-driven |
| **p-value** | Statistical significance of activation difference | Low = reliable effect |
| **Transfer Ratio** | (Injected activation) / (Reasoning activation) | Interpretability metric |

### Classification (Based on Cohen's d, 1988)

Classification uses well-established effect size conventions from Cohen (1988), providing statistically principled thresholds:

| Classification | Criteria | Interpretation |
|---------------|----------|----------------|
| **Token-driven** | d ≥ 0.8, p < 0.01 | Large effect: tokens strongly activate feature |
| **Partially token-driven** | d ≥ 0.5, p < 0.01 | Medium effect: tokens moderately activate feature |
| **Weakly token-driven** | d ≥ 0.2, p < 0.05 | Small effect: tokens weakly activate feature |
| **Context-dependent** | d < 0.2 or p ≥ 0.05 | Negligible effect: may capture reasoning structure |

### Usage

```bash
# Run experiment with simple strategies
python reasoning_features/scripts/run_token_injection_experiment.py \
    --token-analysis results/layer8/token_analysis.json \
    --reasoning-features results/layer8/reasoning_features.json \
    --layer 8 \
    --top-k-features 10 \
    --n-samples 100 \
    --save-dir results/layer8

# Run with contextual strategies (for context-sensitive features)
python reasoning_features/scripts/run_token_injection_experiment.py \
    --token-analysis results/layer8/token_analysis.json \
    --reasoning-features results/layer8/reasoning_features.json \
    --layer 8 \
    --strategies bigram_before bigram_after trigram \
    --n-inject-contextual 2 \
    --save-dir results/layer8

# Visualize token-level activations
python reasoning_features/scripts/visualize_injection_features.py \
    --injection-results results/layer8/injection_results.json \
    --token-analysis results/layer8/token_analysis.json \
    --layer 8 \
    --n-features 5 \
    --n-examples 3 \
    --output-dir visualizations/token_level/layer8
```

This creates interactive HTML visualizations showing:
- **Token-level activations** with color-coded backgrounds (darker = higher activation)
- **Comparison across conditions**: baseline, reasoning, and injected (all strategies)
- **Feature metadata**: Cohen's d, p-value, transfer ratio, classification, best strategy
- Multiple example texts per condition

### Interpretation

| Result | Implication |
|--------|-------------|
| Most features token-driven (d ≥ 0.8) | Supports hypothesis: features are shallow |
| Most features context-dependent (d < 0.2) | Against hypothesis: features may capture reasoning |

### Statistical Justification

The classification uses **Cohen's d effect size**, a standardized measure of the difference between two means:

```
d = (μ_injected - μ_baseline) / σ_pooled
```

Cohen's d thresholds (0.2, 0.5, 0.8) are well-established conventions in psychology and social sciences (Cohen, 1988), making them defensible for reviewers:
- **d = 0.2**: 58% of injected samples exceed baseline median (small effect)
- **d = 0.5**: 69% of injected samples exceed baseline median (medium effect)  
- **d = 0.8**: 79% of injected samples exceed baseline median (large effect)

Combined with t-test p-values (α = 0.01 for large/medium effects, α = 0.05 for small effects), this provides rigorous statistical classification.

## Experiment 3: LLM-Guided Feature Interpretation

**Addressing the limitation of token injection**: While token injection effectively identifies features driven by simple lexical patterns, it cannot exhaust all possible linguistic confounds. A feature classified as "context-dependent" might still be a confound (e.g., detecting prose sophistication, syntactic complexity, or formal style) rather than genuine reasoning.

### Approach

We use an intelligent LLM (Google Gemini 2.0 Flash) to systematically probe feature behavior through iterative hypothesis testing:

1. **Hypothesis Generation**: LLM analyzes high-activation examples and top tokens to hypothesize what linguistic pattern the feature detects

2. **Counterexample Generation**: LLM generates two types of test cases:
   - **False positives**: Non-reasoning text predicted to activate the feature
   - **False negatives**: Reasoning text predicted to NOT activate the feature

3. **Empirical Testing**: Each candidate is tested against the actual model to validate predictions

4. **Iterative Refinement**: Results from previous iterations inform subsequent generation

5. **Final Classification**: LLM determines if feature is a genuine reasoning detector or a linguistic confound

### Example: Feature 715 Discovery

Token injection classified Feature 715 (Layer 16) as "context-dependent" with d=0.18. LLM analysis revealed:

**True nature**: Prose sophistication/complexity detector
- Activates on: Formal writing, complex sentences (>20 words), abstract vocabulary
- Does NOT activate on: Casual language, simple syntax—even when expressing reasoning

**Counterexamples found**:
- 5 false positives (formal non-reasoning text activated)
- 4 false negatives (casual reasoning did not activate)

This demonstrates the feature is a **confound**, not a reasoning detector—a discovery impossible with token injection alone.

### Key Advantages

| Advantage | Description |
|-----------|-------------|
| **Systematic coverage** | Tests diverse patterns beyond manual heuristics |
| **Scalable** | Analyzes hundreds of features automatically |
| **Explainable** | Generates human-interpretable feature descriptions |
| **Iterative** | Learns from failures to improve hypothesis testing |

### Usage

```bash
# Analyze context-dependent features from injection results
python reasoning_features/scripts/analyze_feature_interpretation.py \
    --injection-results results/layer16/injection_results.json \
    --token-analysis results/layer16/token_analysis.json \
    --layer 16 \
    --mode context_dependent \
    --output results/layer16/feature_interpretations.json

# Analyze all reasoning features
python reasoning_features/scripts/analyze_feature_interpretation.py \
    --reasoning-features results/layer16/reasoning_features.json \
    --token-analysis results/layer16/token_analysis.json \
    --layer 16 \
    --mode all_reasoning \
    --output results/layer16/feature_interpretations.json

# Customize parameters
python reasoning_features/scripts/analyze_feature_interpretation.py \
    --feature-indices 715 494 13302 \
    --token-analysis results/layer16/token_analysis.json \
    --layer 16 \
    --max-iterations 5 \
    --min-false-positives 3 \
    --min-false-negatives 3 \
    --threshold-ratio 0.5 \
    --output results/layer16/feature_interpretations.json
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-iterations` | 5 | Maximum counterexample generation rounds |
| `--min-false-positives` | 3 | Minimum FPs to find before stopping |
| `--min-false-negatives` | 3 | Minimum FNs to find before stopping |
| `--threshold-ratio` | 0.5 | Activation threshold (% of max activation) |

### Output

Results include summary statistics and per-feature analyses:

```json
{
  "summary": {
    "total_features_analyzed": 20,
    "genuine_reasoning_features": 2,
    "non_reasoning_features": 18,
    "high_confidence": 15,
    "max_iterations_required": 3.5
  },
  "features": [...]
}
```

## Experiment 4: Steering Experiments

Test whether amplifying "reasoning features" actually improves performance on reasoning benchmarks.

### Steering Formula

We use decoder direction steering:

```
x' = x + γ * f_max * W_dec[i]
```

Where:
- `x`: Original residual stream activation
- `γ`: Steering strength (typically -4 to 4)
- `f_max`: Maximum activation of feature i
- `W_dec[i]`: Decoder direction for feature i

Each feature is steered **individually** to isolate its effect on model behavior.

### Supported Benchmarks

| Benchmark | Task | Metric | Notes |
|-----------|------|--------|-------|
| **AIME24** | Math competition problems (30) | Exact numerical match | `math-ai/aime24` |
| **GPQA Diamond** | Graduate-level science MCQ (198) | A/B/C/D accuracy | `fingertap/GPQA-Diamond` |
| **MATH-500** | Diverse math problems (500) | LLM-judged equivalence | Requires `OPENROUTER_API_KEY` |

### Usage

```bash
# Run steering experiment (each feature individually)
python reasoning_features/scripts/run_steering_experiment.py \
    --features-file results/layer8/reasoning_features.json \
    --benchmark aime24 \
    --gamma-values -2 -1 0 1 2 \
    --top-k-features 10 \
    --save-dir results/layer8/aime24

# Run with specific feature indices
python reasoning_features/scripts/run_steering_experiment.py \
    --feature-indices 42 128 256 \
    --benchmark gpqa_diamond \
    --gamma-values -1 0 1 2

# Run MATH-500 (requires OpenRouter API key)
export OPENROUTER_API_KEY=your_key_here
python reasoning_features/scripts/run_steering_experiment.py \
    --features-file results/layer8/reasoning_features.json \
    --benchmark math500 \
    --gamma-values -2 0 2 \
    --max-samples 50 \
    --save-dir results/layer8/math500

# Quick test (5 samples, 2 features)
python reasoning_features/scripts/run_steering_experiment.py \
    --features-file results/layer8/reasoning_features.json \
    --benchmark aime24 \
    --max-samples 5 \
    --top-k-features 2
```

### MATH-500 Benchmark

The MATH-500 benchmark (`HuggingFaceH4/MATH-500`) contains diverse math problems with answers in various formats:
- Numerical: `-4`, `1.5`
- Algebraic: `1 \pm \sqrt{19}`, `2k+2`
- Text: `\text{east}`, `\text{Monday}`

Because exact string matching would fail for equivalent expressions, we use an LLM judge (Gemini 2.0 Flash via OpenRouter) to evaluate mathematical equivalence. This requires setting the `OPENROUTER_API_KEY` environment variable.

### Steering Gamma Values

| Gamma (γ) | Effect |
|-----------|--------|
| -2.0 | Strong suppression |
| -1.0 | Mild suppression |
| 0.0 | Baseline (no steering) |
| 1.0 | Mild amplification |
| 2.0 | Strong amplification |

### Output Structure

Results are saved per-feature:
```
{save_dir}/
├── experiment_summary.json           # Overall summary
└── feature_{index}/
    ├── feature_summary.json          # Per-feature summary  
    └── result_gamma_{value}.json     # Per-gamma detailed results
```

### Interpreting Results

| Observation | Interpretation |
|-------------|----------------|
| Positive γ improves accuracy | Features may capture genuine reasoning |
| Positive γ hurts accuracy | Features capture spurious token correlations |
| No significant change | Features not task-relevant |

## Datasets

### Reasoning Datasets

| Dataset | Source | Content |
|---------|--------|---------|
| **s1K** | `simplescaling/s1K-1.1` | Gemini/DeepSeek reasoning traces |
| **General Inquiry CoT** | `moremilk/General_Inquiry_Thinking-Chain-Of-Thought` | `<think>` reasoning chains |

### Non-Reasoning Dataset

| Dataset | Source | Content |
|---------|--------|---------|
| **Pile** | `monology/pile-uncopyrighted` | General web text |

## Visualization

Generate plots from experiment results:

```bash
# Generate all plots for a specific experiment
python reasoning_features/scripts/plot_results.py \
    --results-dir results/initial-setting/gemma-2-9b/s1k \
    --plots-dir plots/s1k

# Generate only injection plots
python reasoning_features/scripts/plot_results.py \
    --results-dir results/initial-setting/gemma-2-9b/s1k \
    --only-injection
```

Plot categories: `--only-layer-stats`, `--only-distributions`, `--only-token`, `--only-scatter`, `--only-steering`, `--only-injection`, `--only-interpretation`, `--only-summary`

## Installation

```bash
conda create -n probing python=3.11
conda activate probing
pip install -e .
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
pip uninstall transformer-lens
cd TransformerLens
pip install -e .
```

## Quick Start

```bash
# 1. Find reasoning features in layer 8
python reasoning_features/scripts/find_reasoning_features.py \
    --layer 8 \
    --reasoning-samples 200 \
    --nonreasoning-samples 200 \
    --save-dir results/quick_test

# 2. Run token injection experiment
python reasoning_features/scripts/run_token_injection_experiment.py \
    --token-analysis results/quick_test/token_analysis.json \
    --reasoning-features results/quick_test/reasoning_features.json \
    --layer 8 \
    --top-k-features 3 \
    --save-dir results/quick_test

# 3. Run steering experiment (per-feature)
python reasoning_features/scripts/run_steering_experiment.py \
    --features-file results/quick_test/reasoning_features.json \
    --benchmark gpqa_diamond \
    --gamma-values -1 0 1 \
    --max-samples 10 \
    --top-k-features 3 \
    --save-dir results/quick_steering
```
