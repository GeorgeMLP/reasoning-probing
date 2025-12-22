# Methodology: Investigating Reasoning Features in Sparse Autoencoders

This document provides mathematical definitions, intuitive justifications, and design rationales for our experimental framework investigating whether Sparse Autoencoder (SAE) features capture genuine reasoning ability or merely learn spurious correlations with reasoning-associated tokens.

## Table of Contents

1. [Research Hypothesis](#1-research-hypothesis)
2. [Reasoning Feature Detection](#2-reasoning-feature-detection)
3. [Token Dependency Analysis](#3-token-dependency-analysis)
4. [ANOVA Analysis](#4-anova-analysis)
5. [Steering Experiments](#5-steering-experiments)
6. [Preliminary Results Analysis](#6-preliminary-results-analysis)
7. [Token Injection Experiments](#7-token-injection-experiments-causal-test)
8. [Limitations and Future Work](#8-limitations-and-future-work)

---

## 1. Research Hypothesis

### Core Hypothesis

We hypothesize that SAE features identified as "reasoning features" primarily capture **shallow token-level distributional patterns** rather than **genuine reasoning processes**. Specifically:

1. Features that activate differently on reasoning vs. non-reasoning text are driven by surface tokens (e.g., "Let", "Prove", "First", "therefore") rather than underlying reasoning operations.
2. Steering (amplifying) these features will not improve—and may even decrease—performance on reasoning benchmarks.

### Why This Matters

If SAE features capture only token-level correlates of reasoning:
- Interpretability claims based on these features may be misleading
- Steering interventions based on "reasoning features" will be ineffective
- The linear decomposition assumption of SAEs may be fundamentally limited for higher-order cognitive processes

---

## 2. Reasoning Feature Detection

### 2.1 Overview

We use multiple complementary statistical metrics to identify features that show differential activation between reasoning and non-reasoning text. Each metric captures a different aspect of the difference, and their combination provides robustness against spurious detections.

### 2.2 Metric Definitions

#### ROC-AUC (Area Under the Receiver Operating Characteristic Curve)

For feature $f$ with activation values $\{a_i^{\text{r}}\}_{i=1}^{n_\text{r}}$ on reasoning samples and $\{a_j^{\text{nr}}\}_{j=1}^{n_\text{nr}}$ on non-reasoning samples:

$$\mathrm{AUC} = \mathbb{P}(a^{\text{r}} > a^{\text{nr}}) = \frac{1}{n_\text{r} \cdot n_\text{nr}} \sum_{i=1}^{n_\text{r}} \sum_{j=1}^{n_\text{nr}} \mathbf{1}[a_i^{\text{r}} > a_j^{\text{nr}}].$$

**Threshold:** AUC ≥ 0.6 (10% improvement over random chance)

**Justification:** ROC-AUC is distribution-free and robust to class imbalance. It evaluates discrimination ability across all possible thresholds.

#### Cohen's d (Standardized Effect Size)

$$d = \frac{\bar{a}^{\text{r}} - \bar{a}^{\text{nr}}}{s_{\text{pooled}}},$$

where the pooled standard deviation is:

$$s_{\text{pooled}} = \sqrt{\frac{(n_\text{r} - 1)s_\text{r}^2 + (n_\text{nr} - 1)s_\text{nr}^2}{n_\text{r} + n_\text{nr} - 2}}.$$

**Threshold:** |d| ≥ 0.3 (small-to-medium effect)

#### Mann-Whitney U Test

The Mann-Whitney U statistic tests whether the distributions of activations differ:

$$U = \sum_{i=1}^{n_\text{r}} \sum_{j=1}^{n_\text{nr}} S(a_i^{\text{r}}, a_j^{\text{nr}}),$$

where $S(x, y) = \mathbf{1}[x > y] + 0.5 \cdot \mathbf{1}[x = y]$.

**Threshold:** p-value ≤ 0.01 after Bonferroni correction

### 2.3 Composite Reasoning Score

We combine metrics into a single score:

$$\mathrm{Score} = \mathrm{dir} \times (w_1 \cdot \mathrm{AUC}_{\text{contrib}} + w_2 \cdot \mathrm{Effect}_{\text{contrib}} + w_3 \cdot P_{\text{contrib}} + w_4 \cdot \mathrm{Freq}_{\text{contrib}}),$$

where:
- $\mathrm{dir} = +1$ if mean activation higher in reasoning, $-1$ otherwise
- $\mathrm{AUC}_{\text{contrib}} = |2 \cdot \mathrm{AUC} - 1|$
- $\mathrm{Effect}_{\text{contrib}} = \min(|d|/2, 1)$
- $P_{\text{contrib}} = \min(-\log_{10}(p)/20, 1)$
- $\mathrm{Freq}_{\text{contrib}} = \min(\log_2(\mathrm{FreqRatio})/3, 1)$

Default weights: $w_1 = 0.3, w_2 = 0.3, w_3 = 0.2, w_4 = 0.2$.

---

## 3. Token Dependency Analysis

### 3.1 Purpose

For each detected reasoning feature, we identify which tokens most strongly activate it and measure how concentrated the feature's activations are on specific tokens.

### 3.2 Key Metrics

#### Mean Activation per Token

For token $t$ and feature $f$:

$$\bar{a}_{f,t} = \frac{1}{N_t} \sum_{i: \text{token}_i = t} a_{f,i}$$

Tokens are ranked by mean activation to find the top-k tokens driving each feature.

#### Token Concentration

$$\mathrm{TokenConc} = \frac{\sum_{i=1}^{k} a_{t_i}}{\sum_{\text{all } t} a_t},$$

the fraction of total activation from the top-k tokens. High concentration (>50%) suggests shallow pattern reliance.

#### Normalized Entropy

$$H_{\text{norm}} = -\frac{1}{\log N} \sum_t p_t \log p_t,$$

where $p_t = a_t / \sum_{t'} a_{t'}$. Low entropy indicates concentrated activation patterns.

### 3.3 Interpretation

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Token Concentration | >50% | Feature relies heavily on specific tokens |
| Normalized Entropy | <0.3 | Feature activation is highly concentrated |
| Top tokens | Domain-specific | Feature may be a vocabulary detector |

---

## 4. ANOVA Analysis

### 4.1 Motivation

The central question: **Are reasoning features driven by specific tokens or by the reasoning context?**

Standard correlation analysis cannot disentangle these because tokens and context are confounded in natural data—reasoning text contains both reasoning-associated tokens and reasoning structure.

### 4.2 The 2×2 Factorial Design

We construct four conditions by crossing two factors:

|   | **Has Feature's Top Tokens** | **No Feature's Top Tokens** |
|---|--------------------------|-------------------------|
| **Reasoning Text** | Quadrant A | Quadrant B |
| **Non-Reasoning Text** | Quadrant C | Quadrant D |

This design allows us to separately estimate:
- **Token effect:** Does having specific tokens matter?
- **Context effect:** Does being reasoning text matter?
- **Interaction:** Do tokens work differently in different contexts?

### 4.3 Token-Level Analysis

We analyze activations at the **token level**, not text level:

- **Unit of analysis:** Each token position
- **Token Factor:** Is this specific token in the feature's top-k token set?
- **Context Factor:** Is this text from the reasoning corpus?
- **Dependent variable:** Feature activation at this token position

This directly tests whether individual tokens drive activation.

### 4.4 Statistical Model

For each feature $f$, we fit a two-way ANOVA model:

$$a_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk},$$

where:
- $a_{ijk}$: Activation at token position $k$ in condition $(i, j)$
- $\mu$: Grand mean
- $\alpha_i$: Main effect of **Token Factor**
- $\beta_j$: Main effect of **Context Factor**
- $(\alpha\beta)_{ij}$: Interaction effect

### 4.5 Variance Explained (η²)

$$\eta^2_{\text{Token}} = \frac{\mathit{SS}_{\text{Token}}}{\mathit{SS}_{\text{Total}}}, \quad \eta^2_{\text{Context}} = \frac{\mathit{SS}_{\text{Context}}}{\mathit{SS}_{\text{Total}}}.$$

### 4.6 Classification Criteria

| Classification | Criteria |
|---------------|----------|
| **Token-dominated** | $\eta^2_{\text{token}} > 2 \cdot \eta^2_{\text{context}}$ AND $\eta^2_{\text{token}} > 0.06$ |
| **Context-dominated** | $\eta^2_{\text{context}} > 2 \cdot \eta^2_{\text{token}}$ AND $\eta^2_{\text{context}} > 0.06$ |
| **Mixed** | No single factor dominates but effects are non-negligible |

### 4.7 Interpreting Results

| Finding | Interpretation |
|---------|----------------|
| Token-dominated | Feature responds to specific tokens regardless of context |
| Context-dominated | Feature responds to reasoning context, not just tokens |
| Mixed | Feature responds to token-context combinations |

**Important Note:** A "context-dominated" result does NOT necessarily mean the feature captures genuine reasoning. It could mean:
1. The feature captures reasoning structure (what we'd hope)
2. The feature's activating tokens are highly specific to reasoning text (vocabulary effect)
3. The feature responds to contextual patterns beyond our selected token set

Further investigation is needed to distinguish these possibilities.

### 4.8 Usage

```bash
# Run ANOVA experiment
python reasoning_features/scripts/run_anova_experiment.py \
    --token-analysis results/layer8/token_analysis.json \
    --layer 8 \
    --top-k-features 10 \
    --save-dir results/layer8
```

---

## 5. Steering Experiments

### 5.1 Purpose

Test whether amplifying "reasoning features" actually improves performance on reasoning benchmarks. This provides causal evidence for whether features capture genuine reasoning.

### 5.2 Multiplicative Steering

We use multiplicative steering to scale feature activations:

$$\tilde{f}_i = m \cdot f_i,$$

where $m > 1$ amplifies and $m < 1$ suppresses the feature.

**Why multiplicative:** 
- Respects feature sparsity (only affects active positions)
- Scale-appropriate (amplifies high activations more)
- Interpretable ("2× amplification" has clear meaning)

### 5.3 Supported Benchmarks

| Benchmark | Task | Metric |
|-----------|------|--------|
| AIME24 | Math competition problems | Exact numerical match |
| GPQA Diamond | Graduate-level science MCQ | A/B/C/D accuracy |
| MATH-500 | Diverse math problems | LLM-judged equivalence |

### 5.4 Expected Results

**If features capture genuine reasoning:**
- Amplification (m > 1) should improve accuracy
- Suppression (m < 1) should decrease accuracy

**If features capture shallow patterns:**
- Amplification may decrease accuracy (outputs look reasoning-like but aren't)
- No consistent relationship between multiplier and performance

### 5.5 Usage

```bash
# Run steering experiment
python reasoning_features/scripts/run_steering_experiment.py \
    --features-file results/layer8/reasoning_features.json \
    --benchmark aime24 \
    --multipliers 0.0 0.5 1.0 2.0 4.0 \
    --save-dir results/steering
```

---

## 6. Preliminary Results Analysis

### 6.1 Token Injection Experiment Results (Layer 12, Gemma-2-9B)

Initial experiments on 10 reasoning features from layer 12 show:

| Classification | Count | Percentage |
|---------------|-------|------------|
| Token-driven | 5 | 50% |
| Partially token-driven | 4 | 40% |
| Weakly token-driven | 1 | 10% |
| Context-dependent | 0 | 0% |

**Key Findings:**
- **Average transfer ratio: 62%** - Injecting top tokens into non-reasoning text achieves 62% of reasoning-level activation
- **All features show some token dependency** - No feature was purely context-dependent
- **Prepend strategy works best** - Simply prepending tokens is most effective

### 6.2 Interpretation

These results strongly support the hypothesis that "reasoning features" are largely shallow pattern detectors rather than genuine reasoning mechanisms. The high transfer ratio means that simply prepending a few tokens (like "Let", "Therefore", mathematical notation) can recover most of the activation difference - suggesting the features respond to vocabulary, not reasoning structure.

---

### 6.1 Current Limitations

1. **Token set selection:** The ANOVA analysis depends on which tokens we classify as "reasoning tokens." Different selections may yield different results.

2. **Context confounds:** Even with 2×2 ANOVA, we cannot fully rule out unmeasured confounds. Features may respond to patterns we haven't identified.

3. **Causal interpretation:** Correlation between feature activation and reasoning text does not imply the feature is causally involved in reasoning computation.

## 7. Token Injection Experiments (Causal Test)

### 7.1 Motivation

ANOVA analysis shows correlational relationships, but cannot prove causation. The token injection experiment provides direct causal evidence for whether features are token-driven.

### 7.2 Experimental Design

1. **Baseline**: Measure feature activation on non-reasoning text
2. **Target**: Measure feature activation on reasoning text  
3. **Injection**: Inject the feature's top-k tokens into non-reasoning text
4. **Comparison**: If injected activation ≈ reasoning activation, the feature is token-driven

### 7.3 Injection Strategies

| Strategy | Description |
|----------|-------------|
| **Prepend** | Add tokens at the beginning of text |
| **Intersperse** | Distribute tokens throughout the text |
| **Replace** | Replace random words with tokens |

### 7.4 Key Metrics

**Transfer Ratio**:
$$\text{TransferRatio} = \frac{\bar{a}_{\text{injected}} - \bar{a}_{\text{baseline}}}{\bar{a}_{\text{reasoning}} - \bar{a}_{\text{baseline}}}$$

This measures what fraction of the reasoning-level activation is achieved by token injection alone.

**Statistical Significance**: Independent t-test comparing injected vs. baseline activations, with Cohen's d for effect size.

### 7.5 Classification Criteria

| Classification | Criteria | Interpretation |
|---------------|----------|----------------|
| **Token-driven** | Transfer ratio > 0.5, Cohen's d > 0.3, p < 0.01 | Feature is a shallow token detector |
| **Partially token-driven** | Transfer ratio 0.2-0.5, significant | Tokens partially explain activation |
| **Weakly token-driven** | Transfer ratio < 0.2, but significant | Tokens have minor effect |
| **Context-dependent** | Not significant | Feature may capture deeper patterns |

### 7.6 Interpreting Results

A high average transfer ratio (e.g., 62% as observed in preliminary experiments) indicates that:
- Simply injecting top tokens recovers most of the activation difference
- Features are primarily responding to specific vocabulary, not reasoning structure
- "Reasoning features" are better characterized as "reasoning vocabulary detectors"

### 7.7 Rigorous Transfer Ratio Evaluation

To determine whether a transfer ratio is "high" or "low", we recommend:

1. **Null baseline comparison**: Compare against transfer ratios obtained by injecting random tokens (should be near 0)
2. **Bootstrap confidence intervals**: Report 95% CIs on transfer ratios
3. **Effect size standards** (following Cohen's conventions):
   - Strong token-driven: Transfer ratio > 0.5, Cohen's d > 0.8
   - Moderate: Transfer ratio 0.3-0.5, Cohen's d 0.5-0.8  
   - Weak: Transfer ratio < 0.3, Cohen's d < 0.5

### 7.8 Usage

```bash
python reasoning_features/scripts/run_token_injection_experiment.py \
    --token-analysis results/layer8/token_analysis.json \
    --reasoning-features results/layer8/reasoning_features.json \
    --layer 8 \
    --reasoning-dataset s1k \
    --top-k-features 10 \
    --n-samples 100 \
    --save-dir results/layer8
```

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Token set selection:** The ANOVA analysis depends on which tokens we classify as "reasoning tokens." Different selections may yield different results.

2. **Context confounds:** Even with 2×2 ANOVA, we cannot fully rule out unmeasured confounds. Features may respond to patterns we haven't identified.

3. **Causal interpretation:** Correlation between feature activation and reasoning text does not imply the feature is causally involved in reasoning computation.

4. **Injection strategy effects:** Different injection strategies may yield different results; prepending tends to work best but may not reflect natural token distributions.

### 8.2 Future Directions

1. **Gradient-based attribution:** Use input gradients to identify which tokens causally drive feature activations.

2. **Cross-model validation:** Replicate findings across different model families and sizes.

3. **Mechanistic analysis:** Study how features interact with attention patterns and other model components.

4. **Random baseline experiments:** Compare token injection results against random token injection to establish statistical baselines.

---

## References

1. Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. 2nd ed.
2. Mann, H. B., & Whitney, D. R. (1947). On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other.
3. Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models.
4. Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.
