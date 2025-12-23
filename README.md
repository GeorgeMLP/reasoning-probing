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
│   ├── benchmarks.py   # Evaluation benchmarks (AIME24, GPQA Diamond, MATH-500)
│   └── anova.py        # ANOVA utilities and statistics
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
    ├── run_anova_experiment.py           # Token-level ANOVA analysis
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
├── anova_results.json      # ANOVA analysis results
├── injection_results.json  # Token injection experiment results
└── {benchmark}/            # Steering experiment results per benchmark
    └── experiment_summary.json
```

Example: `results/initial-setting/gemma-2-9b/s1k/layer12/`

## Experiment 2: Steering Experiments

Test whether amplifying "reasoning features" actually improves performance on reasoning benchmarks.

### Supported Benchmarks

| Benchmark | Task | Metric | Notes |
|-----------|------|--------|-------|
| **AIME24** | Math competition problems (30) | Exact numerical match | `math-ai/aime24` |
| **GPQA Diamond** | Graduate-level science MCQ (198) | A/B/C/D accuracy | `fingertap/GPQA-Diamond` |
| **MATH-500** | Diverse math problems (500) | LLM-judged equivalence | Requires `OPENROUTER_API_KEY` |

### Prompt Design

The prompts are designed to allow the model to reason before providing its answer:

1. **One-shot example**: Each prompt includes a worked example (not from the benchmark) demonstrating the expected format
2. **Reasoning-friendly**: Models are asked to "show your reasoning step by step"
3. **Boxed format**: Final answers should be in `\boxed{}` format for reliable extraction

This design allows us to test whether steering "reasoning features" actually affects the model's reasoning process.

### Usage

```bash
# Run steering experiment with detected features
python reasoning_features/scripts/run_steering_experiment.py \
    --features-file results/layer8/reasoning_features.json \
    --benchmark aime24 \
    --multipliers 0.0 0.5 1.0 2.0 4.0 \
    --save-dir results/steering_aime24

# Run with specific feature indices
python reasoning_features/scripts/run_steering_experiment.py \
    --feature-indices 42 128 256 512 \
    --benchmark gpqa_diamond \
    --multipliers 0.5 1.0 2.0

# Run MATH-500 (requires OpenRouter API key)
export OPENROUTER_API_KEY=your_key_here
python reasoning_features/scripts/run_steering_experiment.py \
    --features-file results/layer8/reasoning_features.json \
    --benchmark math500 \
    --max-samples 50 \
    --save-dir results/steering_math500

# Quick test (5 samples)
python reasoning_features/scripts/run_steering_experiment.py \
    --features-file results/layer8/reasoning_features.json \
    --benchmark aime24 \
    --max-samples 5
```

### MATH-500 Benchmark

The MATH-500 benchmark (`HuggingFaceH4/MATH-500`) contains diverse math problems with answers in various formats:
- Numerical: `-4`, `1.5`
- Algebraic: `1 \pm \sqrt{19}`, `2k+2`
- Text: `\text{east}`, `\text{Monday}`

Because exact string matching would fail for equivalent expressions, we use an LLM judge (Gemini 2.0 Flash via OpenRouter) to evaluate mathematical equivalence. This requires setting the `OPENROUTER_API_KEY` environment variable.

### Steering Multipliers

| Multiplier | Effect |
|------------|--------|
| 0.0 | Remove feature entirely |
| 0.5 | Suppress by 50% |
| 1.0 | Baseline (no change) |
| 2.0 | Amplify by 2x |
| 4.0 | Amplify by 4x |

### Interpreting Results

| Observation | Interpretation |
|-------------|----------------|
| Amplification improves accuracy | Features may capture genuine reasoning |
| Amplification hurts accuracy | Features capture spurious token correlations |
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

## Experiment 3: ANOVA Analysis

Disentangle **token-level cues** from **reasoning context** using 2×2 factorial ANOVA.

### 2×2 Factorial Design

| | Has Feature's Top Tokens | No Feature's Top Tokens |
|---|--------------------------|-------------------------|
| **Reasoning Text** | Quadrant A | Quadrant B |
| **Non-Reasoning Text** | Quadrant C | Quadrant D |

### Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **η²_token** | Variance explained by token presence | High = token-dependent |
| **η²_context** | Variance explained by context | High = context-sensitive |

### Usage

```bash
python reasoning_features/scripts/run_anova_experiment.py \
    --token-analysis results/layer8/token_analysis.json \
    --layer 8 \
    --top-k-features 10 \
    --n-reasoning-texts 500 \
    --n-nonreasoning-texts 2000 \
    --save-dir results/layer8
```

## Experiment 4: Token Injection (Causal Test) ⭐

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
| **Transfer Ratio** | (Injected activation) / (Reasoning activation) | High = token-driven |
| **Activation Increase** | Post-injection - pre-injection | Significant = tokens matter |

### Classification

| Classification | Criteria | Interpretation |
|---------------|----------|----------------|
| **Token-driven** | Transfer ratio > 0.5, significant | Shallow pattern detector |
| **Partially token-driven** | Transfer ratio 0.2-0.5 | Mixed behavior |
| **Context-dependent** | Transfer ratio < 0.2 | May capture reasoning structure |

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
- **Comparison across conditions**: baseline, reasoning, and injected (all 3 strategies)
- **Feature metadata**: transfer ratio, classification, best strategy
- Multiple example texts per condition

### Interpretation

| Result | Implication |
|--------|-------------|
| Most features token-driven | Supports hypothesis: features are shallow |
| Most features context-dependent | Against hypothesis: features may capture reasoning |

### Preliminary Results

In layer 12 of Gemma-2-9B:
- **62% average transfer ratio** - Injecting tokens achieves ~62% of reasoning-level activation
- **50% token-driven** - Half of features strongly respond to token injection
- **0% context-dependent** - No features are purely context-sensitive

These results support our hypothesis that "reasoning features" are shallow pattern detectors.

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

Plot categories: `--only-layer-stats`, `--only-distributions`, `--only-token`, `--only-scatter`, `--only-steering`, `--only-anova`, `--only-injection`, `--only-summary`

## Installation

```bash
conda create -n probing python=3.10
bash setup.sh
```

## Quick Start

```bash
# 1. Find reasoning features in layer 8
python reasoning_features/scripts/find_reasoning_features.py \
    --layer 8 \
    --reasoning-samples 200 \
    --nonreasoning-samples 200 \
    --save-dir results/quick_test

# 2. Run steering experiment
python reasoning_features/scripts/run_steering_experiment.py \
    --features-file results/quick_test/reasoning_features.json \
    --benchmark gpqa_diamond \
    --max-samples 10 \
    --save-dir results/quick_steering
```
