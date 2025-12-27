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

### 5.2 Decoder Direction Steering

We use direct decoder direction steering to modify the residual stream:

$$x' = x + \gamma \cdot f_i^{\max} \cdot W_{\text{dec},i},$$

where:
- $x$ is the original residual stream activation (shape: `[batch, seq, d_model]`)
- $\gamma$ is the steering strength (typically -4 to 4)
- $f_i^{\max}$ is the maximum activation of feature $i$ (pre-computed from the dataset)
- $W_{\text{dec},i}$ is the $i$-th row of the SAE decoder matrix (shape: `[d_model]`)

**Why decoder direction steering:**
- **Direct intervention**: Adds the decoder direction without encode/decode roundtrip
- **Scaled appropriately**: Uses maximum activation to scale the intervention magnitude
- **Bidirectional**: Negative $\gamma$ suppresses, positive $\gamma$ amplifies
- **Per-feature**: Each feature is steered individually to isolate effects

### 5.3 Per-Feature Evaluation

Unlike previous approaches that steered all top-k features simultaneously, we now evaluate each feature individually:

1. For each reasoning feature $i$:
   - Compute $f_i^{\max}$ from the dataset
   - Test multiple $\gamma$ values (e.g., -2, -1, 0, 1, 2)
   - Measure benchmark accuracy for each $\gamma$
   
2. This allows:
   - Identifying which specific features affect reasoning
   - Distinguishing features with positive vs. negative effects
   - Understanding individual feature contributions

### 5.4 Supported Benchmarks

| Benchmark | Task | Metric |
|-----------|------|--------|
| AIME24 | Math competition problems | Exact numerical match |
| GPQA Diamond | Graduate-level science MCQ | A/B/C/D accuracy |
| MATH-500 | Diverse math problems | LLM-judged equivalence |

### 5.5 Expected Results

**If features capture genuine reasoning:**
- Positive $\gamma$ should improve accuracy
- Negative $\gamma$ should decrease accuracy

**If features capture shallow patterns:**
- Positive $\gamma$ may decrease accuracy (outputs look reasoning-like but aren't)
- No consistent relationship between $\gamma$ and performance

### 5.6 Output Structure

Results are saved per-feature:
```
{save_dir}/{benchmark}/
├── experiment_summary.json      # Overall summary with per_feature_results
└── feature_{index}/
    ├── feature_summary.json     # Per-feature summary
    └── result_gamma_{value}.json  # Detailed results for each gamma
```

### 5.7 Usage

```bash
# Run steering experiment (each feature individually)
python reasoning_features/scripts/run_steering_experiment.py \
    --features-file results/layer8/reasoning_features.json \
    --benchmark aime24 \
    --gamma-values -2 -1 0 1 2 \
    --top-k-features 10 \
    --save-dir results/layer8/aime24
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
- **Average Cohen's d: 1.2** - Large effect size across features
- **All features show some token dependency** - No feature was purely context-dependent  
- **Prepend strategy works best** - Simply prepending tokens is most effective

### 6.2 Interpretation

These results strongly support the hypothesis that "reasoning features" are largely shallow pattern detectors rather than genuine reasoning mechanisms. The large average Cohen's d (1.2) and high proportion of token-driven features (50%) indicate that simply prepending a few tokens (like "Let", "Therefore", mathematical notation) produces substantial activation increases - suggesting the features respond to vocabulary, not reasoning structure.

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

### 7.2.1 Why Contextual Strategies?

Initial experiments revealed that many features are not triggered by individual tokens, but by **token sequences** or **positional patterns**:

- **Syntactic patterns**: "to [verb]" constructions (e.g., "to identify", "to consider")
- **Modal constructions**: "I [modal verb]" patterns (e.g., "I need", "I should")
- **List structures**: Comma-separated items
- **Phrasal units**: Multi-word expressions

These features are still **shallow pattern detectors** - they don't capture reasoning structure, just slightly more sophisticated lexical patterns. Contextual injection strategies test whether these sequence-sensitive features can be activated by injecting the appropriate token combinations.

### 7.3 Injection Strategies

We test both simple and contextual injection strategies:

**Simple Strategies** (baseline):
| Strategy | Description |
|----------|-------------|
| **Prepend** | Add tokens at the beginning of text |
| **Append** | Add tokens at the end of text |
| **Intersperse** | Distribute tokens throughout the text |
| **Replace** | Replace random words with tokens |

**Contextual Strategies** (token + surrounding context):
| Strategy | Description | Example |
|----------|-------------|---------|
| **Bigram Before** | Inject [context, token] pairs | "to identify" (if "to" often precedes "identify") |
| **Bigram After** | Inject [token, context] pairs | "need to" (if "to" often follows "need") |
| **Trigram** | Inject [before, token, after] triplets | "to identify the" |
| **Comma List** | Inject tokens as comma-separated list | "first, second, third" |

Contextual strategies are designed to capture features that are sensitive to token sequences rather than individual tokens, such as:
- Features that activate on verbs only after "to"
- Features that activate on "I" only before "need" or "should"
- Features that activate on items in enumerated lists

**Injection Count**: Since contextual strategies inject multi-token sequences (bigrams/trigrams), we inject fewer of them (default: 2 sequences for contextual strategies vs. 3 tokens for simple strategies) to avoid overwhelming the text with injected content.

### 7.4 Key Metrics

**Transfer Ratio**:
$$\mathrm{TransferRatio} = \frac{\bar{a}_{\text{injected}} - \bar{a}_{\text{baseline}}}{\bar{a}_{\text{reasoning}} - \bar{a}_{\text{baseline}}}.$$

This measures what fraction of the reasoning-level activation is achieved by token injection alone (for interpretability).

**Cohen's d Effect Size**:
$$d = \frac{\bar{a}_{\text{injected}} - \bar{a}_{\text{baseline}}}{\sigma_{\text{pooled}}}.$$

This provides a standardized measure of the difference between injected and baseline activations, independent of scale.

**Statistical Significance**: Independent t-test comparing injected vs. baseline activations.

### 7.5 Classification Criteria (Cohen, 1988)

Classification uses well-established effect size conventions from Cohen (1988):

| Classification | Criteria | Interpretation |
|---------------|----------|----------------|
| **Token-driven** | d ≥ 0.8, p < 0.01 | Large effect: feature is a strong token detector |
| **Partially token-driven** | d ≥ 0.5, p < 0.01 | Medium effect: tokens moderately activate feature |
| **Weakly token-driven** | d ≥ 0.2, p < 0.05 | Small effect: tokens have minor but reliable effect |
| **Context-dependent** | d < 0.2 or p ≥ 0.05 | Negligible effect: feature may capture deeper patterns |

These thresholds correspond to the percentage of injected samples exceeding the baseline median:
- d = 0.2: 58% (small effect)
- d = 0.5: 69% (medium effect)
- d = 0.8: 79% (large effect)

### 7.6 Interpreting Results

A high proportion of token-driven features (d ≥ 0.8) indicates that:
- Simply injecting top tokens produces large activation increases
- Features are primarily responding to specific vocabulary, not reasoning structure
- "Reasoning features" are better characterized as "reasoning vocabulary detectors"

### 7.7 Statistical Justification

The Cohen's d approach provides several advantages over arbitrary thresholds:

1. **Standardized**: Effect sizes are scale-independent and comparable across features
2. **Established conventions**: Cohen (1988) thresholds are widely accepted in psychology and social sciences
3. **Interpretable**: Maps directly to overlap between distributions and probability of superiority
4. **Rigorous**: Combined with p-values, provides both practical significance (effect size) and statistical reliability (p-value)

### 7.8 Usage

```bash
# Simple strategies (default)
python reasoning_features/scripts/run_token_injection_experiment.py \
    --token-analysis results/layer8/token_analysis.json \
    --reasoning-features results/layer8/reasoning_features.json \
    --layer 8 \
    --reasoning-dataset s1k \
    --top-k-features 10 \
    --n-samples 100 \
    --save-dir results/layer8

# Customize Cohen's d thresholds
python reasoning_features/scripts/run_token_injection_experiment.py \
    --token-analysis results/layer8/token_analysis.json \
    --reasoning-features results/layer8/reasoning_features.json \
    --layer 8 \
    --d-large 0.8 \
    --d-medium 0.5 \
    --d-small 0.2 \
    --alpha 0.01 \
    --alpha-weak 0.05 \
    --save-dir results/layer8
    --token-analysis results/layer8/token_analysis.json \
    --reasoning-features results/layer8/reasoning_features.json \
    --layer 8 \
    --reasoning-dataset s1k \
    --strategies bigram_before bigram_after trigram comma_list \
    --n-inject-contextual 2 \
    --save-dir results/layer8

# Mixed strategies
python reasoning_features/scripts/run_token_injection_experiment.py \
    --token-analysis results/layer8/token_analysis.json \
    --reasoning-features results/layer8/reasoning_features.json \
    --layer 8 \
    --strategies prepend bigram_before trigram \
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
