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
│   └── benchmarks.py   # Evaluation benchmarks (AIME24, GPQA Diamond)
├── features/           # Feature analysis
│   ├── collector.py    # SAE activation collection
│   ├── detector.py     # Reasoning feature detection
│   └── tokens.py       # Token dependency analysis
├── steering/           # Intervention experiments
│   ├── steerer.py      # Activation steering
│   └── evaluator.py    # Benchmark evaluation
└── scripts/            # Main experiment scripts
    ├── find_reasoning_features.py
    └── run_steering_experiment.py
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

```
results/layer8/
├── activations.pt          # Cached activations for reuse
├── feature_stats.json      # Statistics for all features
├── reasoning_features.json # Detected reasoning features
└── token_analysis.json     # Token dependency analysis
```

## Experiment 2: Steering Experiments

Test whether amplifying "reasoning features" actually improves performance on reasoning benchmarks.

### Supported Benchmarks

| Benchmark | Task | Metric |
|-----------|------|--------|
| **AIME24** | Math competition problems | Exact match (boxed answer) |
| **GPQA Diamond** | Graduate-level science MCQ | A/B/C/D accuracy |

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

# Quick test (5 samples)
python reasoning_features/scripts/run_steering_experiment.py \
    --features-file results/layer8/reasoning_features.json \
    --benchmark aime24 \
    --max-samples 5
```

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

## Future Work: ANOVA Analysis

The next step is to disentangle **token-level cues** from **reasoning behavior** using ANOVA.

### 2×2 Factorial Design

| | Has Reasoning Tokens | No Reasoning Tokens |
|---|---------------------|---------------------|
| **Is Reasoning** | Quadrant A | Quadrant B |
| **Not Reasoning** | Quadrant C | Quadrant D |

### Analysis Plan

For each feature, fit a linear model:
```
activation ~ token_factor + behavior_factor + interaction
```

**Expected findings:**
- Token factor explains most variance → Feature relies on shallow cues
- Behavior factor explains significant variance → Feature may capture reasoning

This framework is designed to support this analysis—the `FeatureActivations` class stores all necessary metadata.

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
