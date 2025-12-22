# Methodology: Detecting and Analyzing Reasoning Features in Sparse Autoencoders

This document provides mathematical definitions, intuitive justifications, and design rationales for our experimental framework investigating whether Sparse Autoencoder (SAE) features capture genuine reasoning ability or merely learn spurious correlations with reasoning-associated tokens.

## Table of Contents

1. [Research Hypothesis](#1-research-hypothesis)
2. [Reasoning Feature Detection Metrics](#2-reasoning-feature-detection-metrics)
3. [Token Dependency Analysis](#3-token-dependency-analysis)
4. [Steering Experiment Design](#4-steering-experiment-design)
5. [ANOVA Analysis for Disentangling Token vs. Behavior Effects](#5-anova-analysis-for-disentangling-token-vs-behavior-effects)
6. [Empirical Validation](#6-empirical-validation)
7. [Improved Feature Selection for Steering](#7-improved-feature-selection-for-steering)
8. [Limitations and Future Work](#8-limitations-and-future-work)
9. [Summary](#9-summary)

---

## 1. Research Hypothesis

### Core Hypothesis

We hypothesize that SAE features identified as "reasoning features" primarily capture **shallow token-level distributional patterns** rather than **genuine reasoning processes**. Specifically:

1. Features that activate differently on reasoning vs. non-reasoning text are driven by surface tokens (e.g., "Let", "Prove", "First", "therefore") rather than underlying reasoning operations.
2. Steering (amplifying) these features will not improve—and may even decrease—performance on reasoning benchmarks.
3. A 2×2 ANOVA analysis will show that the **token factor** explains significantly more variance than the **reasoning behavior factor**.

### Why This Matters

If SAE features capture only token-level correlates of reasoning:
- Interpretability claims based on these features may be misleading
- Steering interventions based on "reasoning features" will be ineffective
- The linear decomposition assumption of SAEs may be fundamentally limited for higher-order cognitive processes

---

## 2. Reasoning Feature Detection Metrics

### 2.1 Overview

We use multiple complementary statistical metrics to identify features that show differential activation between reasoning and non-reasoning text. Each metric captures a different aspect of the difference, and their combination provides robustness against spurious detections.

### 2.2 Metric Definitions

#### 2.2.1 ROC-AUC (Area Under the Receiver Operating Characteristic Curve)

**Definition:** For feature $f$ with activation values $\{a_i^{\text{r}}\}_{i=1}^{n_\text{r}}$ on reasoning samples and $\{a_j^{\text{nr}}\}_{j=1}^{n_\text{nr}}$ on non-reasoning samples, the ROC-AUC measures the probability that a randomly chosen reasoning sample has higher activation than a randomly chosen non-reasoning sample:

$$\mathrm{AUC} = \mathbb{P}(a^{\text{r}} > a^{\text{nr}}) = \frac{1}{n_\text{r} \cdot n_\text{nr}} \sum_{i=1}^{n_\text{r}} \sum_{j=1}^{n_\text{nr}} \mathbf{1}[a_i^{\text{r}} > a_j^{\text{nr}}].$$

**Threshold:** AUC ≥ 0.6 (10% improvement over random chance)

**Justification:** ROC-AUC is distribution-free and robust to class imbalance. Unlike accuracy, it evaluates discrimination ability across all possible thresholds, making it ideal when we don't know the optimal activation threshold a priori.

#### 2.2.2 Cohen's d (Standardized Effect Size)

**Definition:** Cohen's d measures the standardized difference between group means:

$$d = \frac{\bar{a}^{\text{r}} - \bar{a}^{\text{nr}}}{s_{\text{pooled}}},$$

where the pooled standard deviation is:

$$s_{\text{pooled}} = \sqrt{\frac{(n_\text{r} - 1)s_\text{r}^2 + (n_\text{nr} - 1)s_\text{nr}^2}{n_\text{r} + n_\text{nr} - 2}}.$$

**Threshold:** |d| ≥ 0.3 (small-to-medium effect)

**Justification:** Cohen's d is scale-invariant and provides interpretable effect sizes:
- |d| ≈ 0.2: small effect
- |d| ≈ 0.5: medium effect
- |d| ≈ 0.8: large effect

This allows comparison across features with different activation magnitudes and ensures we detect practically meaningful differences, not just statistically significant ones.

#### 2.2.3 Mann-Whitney U Test (Non-Parametric Hypothesis Test)

**Definition:** The Mann-Whitney U statistic tests whether the distributions of activations differ:

$$U = \sum_{i=1}^{n_\text{r}} \sum_{j=1}^{n_\text{nr}} S(a_i^{\text{r}}, a_j^{\text{nr}}),$$

where $S(x, y) = \mathbf{1}[x > y] + 0.5 \cdot \mathbf{1}[x = y]$.

**Threshold:** p-value ≤ 0.01 after Bonferroni correction (p ≤ 0.01/K for K features)

**Justification:** Unlike t-tests, Mann-Whitney makes no assumptions about normality—critical since SAE activations are typically sparse and heavy-tailed. Bonferroni correction controls family-wise error rate when testing thousands of features simultaneously.

#### 2.2.4 Activation Frequency Ratio

**Definition:** The ratio of activation frequencies between groups:

$$\mathrm{FreqRatio} = \frac{P(a^{\text{r}} > \tau) + \epsilon}{P(a^{\text{nr}} > \tau) + \epsilon},$$

where $\tau = 0.01 \cdot \max(a)$ and $\epsilon = 0.01$ prevents division by zero.

**Justification:** This captures whether a feature fires more often in reasoning contexts, which is particularly relevant for sparse features that may have similar activation magnitudes but different activation probabilities.

### 2.3 Composite Reasoning Score

**Definition:** We combine the metrics into a single score:

$$\mathrm{ReasoningScore} = \operatorname{sign}(\bar{a}^{\text{r}} - \bar{a}^{\text{nr}}) \cdot \left( 
    w_1 \cdot \mathrm{AUC}_{\text{contrib}} + 
    w_2 \cdot \mathrm{Effect}_{\text{contrib}} + 
    w_3 \cdot \mathrm{P}_{\text{contrib}} + 
    w_4 \cdot \mathrm{Freq}_{\text{contrib}}
\right),$$

where:
- $\mathrm{AUC}_{\text{contrib}} = 2 \cdot |\mathrm{AUC} - 0.5|$ (scaled to [0, 1])
- $\mathrm{Effect}_{\text{contrib}} = \min(|d|, 3) / 3$ (capped at d=3)
- $\mathrm{P}_{\text{contrib}} = \min(-\log_{10}(p), 50) / 50$ (log-scaled p-value)
- $\mathrm{Freq}_{\text{contrib}} = \min(\log_2(\mathrm{FreqRatio} + 1) / 5, 1)$ (log-scaled ratio)
- Weights: $w_1 = 0.30$, $w_2 = 0.25$, $w_3 = 0.25$, $w_4 = 0.20$

**Justification for Weights:**
1. **AUC (30%):** Receives highest weight as the most robust discrimination metric
2. **Effect size (25%):** Ensures practical significance beyond statistical significance
3. **P-value (25%):** Guards against random fluctuations in small samples
4. **Frequency (20%):** Lower weight because it's partially redundant with AUC

**Why Weighted Sum vs. Other Aggregation Methods:**
- **Multiplicative:** Would require all metrics to be non-zero, overly penalizing features strong on most but not all metrics
- **Voting:** Loses granularity in ranking
- **PCA/Factor analysis:** Would require cross-feature standardization, complicating interpretation
- **Weighted sum:** Provides smooth ranking, interpretable contributions, and allows features to compensate for weaknesses in one metric with strengths in others

---

## 3. Token Dependency Analysis

### 3.1 Motivation

Even if a feature shows strong differential activation on reasoning text, this doesn't mean it captures reasoning per se. The feature may simply respond to specific tokens that happen to appear more frequently in reasoning contexts (e.g., "prove", "therefore", "let").

### 3.2 Metrics for Token-Feature Association

#### 3.2.1 Mean Activation

**Definition:** For token $t$ appearing at positions $\{(i, j)\}$ where $i$ indexes samples and $j$ indexes positions:

$$\bar{a}_t = \frac{1}{|P_t|} \sum_{(i,j) \in P_t} a_{i,j}.$$

where $P_t$ is the set of all positions where token $t$ appears.

**Justification:** Mean activation is the most direct measure of how strongly a token triggers a feature. It's interpretable, stable across sample sizes, and directly relates to the feature's learned representation.

#### 3.2.2 Pointwise Mutual Information (PMI)

**Definition:** PMI measures the co-occurrence strength between token $t$ and feature activation:

$$\mathrm{PMI}(t, f) = \log_2 \frac{P(a > \tau, t)}{P(a > \tau) \cdot P(t)}.$$

where $\tau$ is the activation threshold.

**Justification:** PMI identifies tokens that are unusually associated with feature activation, beyond what would be expected by chance. This helps detect specific triggers rather than common tokens that appear everywhere.

#### 3.2.3 Token Concentration

**Definition:** The fraction of high activations that come from the top-k tokens:

$$\mathrm{Concentration} = \frac{\sum_{t \in \mathrm{Top}_k} \operatorname{Count}(a > \tau \land t)}{\operatorname{Count}(a > \tau)}.$$

**Threshold:** Concentration > 0.5 indicates high token dependency

**Justification:** This directly measures whether a feature's activation is dominated by a small set of tokens. A feature with concentration > 0.5 has more than half of its high activations triggered by just the top-k tokens, suggesting it's a "token detector" rather than a "concept detector."

#### 3.2.4 Normalized Entropy

**Definition:** The entropy of the token distribution weighted by activation:

$$H_{\text{norm}} = \frac{-\sum_t p_t \log_2 p_t}{\log_2 |V|},$$

where $p_t = \frac{\sum_{(i,j): \text{token}_{i,j} = t} a_{i,j}}{\sum_{i,j} a_{i,j}}$ and $|V|$ is vocabulary size.

**Justification:** Low entropy indicates the feature's activation is concentrated on few tokens; high entropy indicates distributed activation across many tokens. Features with higher entropy are more likely to capture semantic concepts rather than specific token patterns.

### 3.3 Why Mean Activation for Ranking

We rank tokens by mean activation rather than PMI or other metrics because:

1. **Interpretability:** "Token X has mean activation 5.2" is directly interpretable in terms of the feature's learned representation
2. **Stability:** Mean activation is less sensitive to rare tokens than PMI
3. **Relevance:** We care about which tokens most strongly drive the feature, not which are most unusually associated
4. **Compatibility:** Mean activation allows comparison across features with different sparsity levels

However, we report PMI and concentration as secondary metrics to provide a complete picture.

---

## 4. Steering Experiment Design

### 4.1 Steering Methods

#### Multiplicative Steering

**Definition:** Modify feature activations by a scalar multiplier:

$$\tilde{f}_i = m \cdot f_i,$$

where $m > 1$ amplifies and $m < 1$ suppresses the feature.

#### Additive Steering

**Definition:** Add a fixed value to feature activations:

$$\tilde{f}_i = f_i + \delta.$$

### 4.2 Multiplicative vs. Additive: Design Choice

**We choose multiplicative steering** for the following reasons:

1. **Respects feature sparsity:** Multiplicative steering only affects positions where the feature is already active. Additive steering would activate the feature everywhere, fundamentally changing its behavior.

2. **Scale-appropriate:** SAE features have varying activation magnitudes. Multiplicative steering scales interventions appropriately (amplifying high activations more than low ones).

3. **Interpretable:** "2× amplification" has a clear meaning regardless of the feature's baseline activation level.

4. **Preserves relative patterns:** If a feature fires more strongly at position A than B, multiplicative steering preserves this relationship.

**When additive might be appropriate:**
- For features that are rarely active, additive steering can "force" activation
- For studying what happens when a feature is injected into contexts where it normally wouldn't fire

### 4.3 Steering One vs. Many Features

**We steer multiple features simultaneously (top-k)** because:

1. **Realistic intervention:** In practice, multiple features may jointly encode a concept. Steering one feature while leaving correlated features unchanged may produce inconsistent representations.

2. **Statistical power:** Individual feature effects may be small; aggregating across multiple related features provides a stronger signal.

3. **Robustness:** If one feature in the set is a false positive, the others may still provide meaningful steering.

**Considerations:**
- We test k ∈ {10, 20, 50, 100} to evaluate sensitivity to this choice
- We also run single-feature ablations for the top features to check for dominant individual effects

### 4.4 Multiplier Selection

We test multipliers {0.0, 0.5, 1.0, 2.0, 4.0} to cover:
- **0.0 (ablation):** Complete removal of the feature
- **0.5 (suppression):** Moderate dampening
- **1.0 (baseline):** No intervention (control)
- **2.0 (mild amplification):** Moderate enhancement
- **4.0 (strong amplification):** Aggressive enhancement

This range allows us to observe both monotonic and non-monotonic effects.

---

## 5. ANOVA Analysis for Disentangling Token vs. Behavior Effects

### 5.1 Motivation

The central question: **Are reasoning features driven by tokens or by reasoning behavior?**

Standard correlation analysis cannot disentangle these because tokens and behavior are confounded in natural data—reasoning text contains both reasoning tokens and reasoning behavior.

### 5.2 The 2×2 Factorial Design

We construct four conditions by crossing two factors:

|   | **Has Reasoning Tokens** | **No Reasoning Tokens** |
|---|--------------------------|-------------------------|
| **Reasoning Chain** | Quadrant A | Quadrant B |
| **General Text** | Quadrant C | Quadrant D |

#### Quadrant Definitions:

**Quadrant A (Reasoning + Reasoning Tokens):** Natural reasoning text (existing data)

**Quadrant B (Reasoning + No Reasoning Tokens):** Reasoning chains with explicit reasoning tokens masked/replaced
- Replace "Let's think step by step" → "Here is the process"
- Replace "Therefore" → neutral connectives
- Remove first-person hedging ("I think", "Maybe")

**Quadrant C (Not Reasoning + Reasoning Tokens):** Non-reasoning text injected with reasoning tokens
- Insert "Let me think..." into factual passages
- Add "Step 1: ..." structure to procedural text without actual reasoning

**Quadrant D (Not Reasoning + No Reasoning Tokens):** Natural non-reasoning text (existing data)

### 5.3 Linear Model Specification

For each feature $f$, we fit a two-way ANOVA model:

$$a_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk},$$

where:
- $a_{ijk}$: Activation of feature $f$ for sample $k$ in condition $(i, j)$
- $\mu$: Grand mean
- $\alpha_i$: Main effect of **Token Factor** (i = has tokens, no tokens)
- $\beta_j$: Main effect of **Behavior Factor** (j = reasoning, not reasoning)
- $(\alpha\beta)_{ij}$: Interaction effect
- $\epsilon_{ijk} \sim N(0, \sigma^2)$: Residual error

### 5.4 Sum of Squares Decomposition

The total sum of squares decomposes as:

$$\mathit{SS}_{\text{Total}} = \mathit{SS}_{\text{Token}} + \mathit{SS}_{\text{Behavior}} + \mathit{SS}_{\text{Interaction}} + \mathit{SS}_{\text{Error}},$$

where:

$$\mathit{SS}_{\text{Token}} = n_j n_k \sum_i (\bar{a}_{i\cdot\cdot} - \bar{a}_{\cdot\cdot\cdot})^2,$$

$$\mathit{SS}_{\text{Behavior}} = n_i n_k \sum_j (\bar{a}_{\cdot j\cdot} - \bar{a}_{\cdot\cdot\cdot})^2,$$

$$\mathit{SS}_{\text{Interaction}} = n_k \sum_{i,j} (\bar{a}_{ij\cdot} - \bar{a}_{i\cdot\cdot} - \bar{a}_{\cdot j\cdot} + \bar{a}_{\cdot\cdot\cdot})^2.$$

### 5.5 Variance Explained Metrics

We compute the **eta-squared** (η²) effect sizes:

$$\eta^2_{\text{Token}} = \frac{\mathit{SS}_{\text{Token}}}{\mathit{SS}_{\text{Total}}},$$

$$\eta^2_{\text{Behavior}} = \frac{\mathit{SS}_{\text{Behavior}}}{\mathit{SS}_{\text{Total}}}.$$

### 5.6 Decision Rule

A feature is **dominated by token patterns** if:

$$\eta^2_{\text{Token}} > 2 \cdot \eta^2_{\text{Behavior}} \quad \text{AND} \quad \eta^2_{\text{Token}} > 0.1.$$

This requires:
1. Token factor explains at least twice as much variance as behavior factor
2. Token factor has at least a "small-to-medium" effect size (η² > 0.1)

### 5.7 Expected Outcomes

**If our hypothesis is correct:**
- Most "reasoning features" will satisfy the domination criterion
- $\eta^2_{\text{Token}} \gg \eta^2_{\text{Behavior}}$ across the population of reasoning features
- Interaction effects will be small (token and behavior effects are additive)

**Alternative outcomes:**
- If $\eta^2_{\text{Behavior}} > \eta^2_{\text{Token}}$: Features capture genuine reasoning (surprising)
- If both effects are large with significant interaction: Features respond to token-behavior combinations

### 5.8 Token-Level vs. Text-Level ANOVA

#### 5.8.1 The Problem with Text-Level ANOVA

The original text-level ANOVA design has a critical limitation: it uses **max activation across the sequence** for each text. This aggregation has two problems:

1. **Information loss:** The max operation discards information about which specific tokens drove the activation
2. **Token coverage:** Texts classified as "not having reasoning tokens" may still contain many OTHER tokens that activate the feature

**Empirical Evidence:** In our experiments, text-level ANOVA often shows "behavior-dominated" results even for features that respond strongly to specific tokens. For example:

| Feature | Text-Level η²_behavior | Text-Level η²_token | Result |
|---------|------------------------|---------------------|--------|
| 8684    | 0.54                   | 0.015               | Behavior-dominated |
| 12481   | 0.20                   | 0.14                | Mixed |

#### 5.8.2 Token-Level ANOVA Design

We developed a refined approach that operates at the **token level**:

**Unit of analysis:** Each token position (not each text)

**Factors:**
- **Token Factor:** Is this specific token in the feature's top-k token set?
- **Context Factor:** Is this text from the reasoning corpus?

**Dependent variable:** Feature activation at THIS token position

**Key advantage:** This directly tests whether individual tokens drive activation, regardless of surrounding context.

#### 5.8.3 Token-Level Results

Token-level ANOVA reveals a more nuanced picture:

| Feature | η²_token (Token-Level) | η²_context (Token-Level) | Token Effect in R | Token Effect in NR |
|---------|------------------------|--------------------------|-------------------|-------------------|
| 8684    | 0.0005                 | 0.24                     | 0.63              | -0.01             |
| 12481   | 0.087                  | 0.023                    | 14.09             | 1.70              |

**Key Insight:** Feature 12481 IS token-dominated at the token level, with a strong token effect (14.09) that partially transfers to non-reasoning context (1.70). Feature 8684's tokens only work in reasoning context.

#### 5.8.4 Token Consistency Metric

We introduce a new metric to measure whether tokens work consistently across contexts:

$$\text{TokenConsistency} = 1 - \frac{|\text{TokenEffect}_R - \text{TokenEffect}_{NR}|}{\max(|\text{TokenEffect}_R|, |\text{TokenEffect}_{NR}|) + \epsilon}$$

- **High consistency (>0.5):** Token effect is stable across contexts → genuine token-driven feature
- **Low consistency (<0.5):** Token effect depends on context → feature learns token+context interactions

#### 5.8.5 Context-Agnostic Token Analysis

To control for confounding, we analyze only tokens that appear in BOTH reasoning and non-reasoning text:

1. Filter feature's top tokens to those with occurrences in both contexts
2. Run token-level ANOVA using only these "context-agnostic" tokens
3. Check if same tokens produce similar activations regardless of context

**Finding:** Even "token-dominated" features show context modulation. Feature 12481's tokens activate 8x more in reasoning context (14.5 vs 1.7), suggesting features learn **token+context interactions**, not pure token patterns.

#### 5.8.6 Classification of Features

Based on token-level analysis, features fall into three categories:

| Category | Criteria | Interpretation |
|----------|----------|----------------|
| **Token-Dominated** | η²_token > 2×η²_context, TokenConsistency > 0.5 | Feature responds to specific tokens regardless of context |
| **Context-Dominated** | η²_context > 2×η²_token, TokenConsistency < 0.2 | Same tokens behave differently based on surrounding context |
| **Confounded** | TokenConsistency < 0.3, high context effect | Tokens only work in reasoning context; cannot separate factors |

### 5.9 Implementation Details

#### 5.9.1 Supported Datasets

The ANOVA experiment supports three reasoning dataset configurations:

| Dataset | Description | Source Column |
|---------|-------------|---------------|
| `s1k` | s1K-1.1 reasoning trajectories | `gemini_thinking_trajectory`, `deepseek_thinking_trajectory` |
| `general_inquiry_cot` | General Inquiry Chain-of-Thought | `metadata.reasoning` (extracted from `<think>` tags) |
| `combined` | Both datasets merged | All of the above |

The non-reasoning baseline is always drawn from `monology/pile-uncopyrighted`.

#### 5.9.2 Text Splitting Strategy

Long reasoning chains are split into sentence-level chunks to increase sample size for the 2×2 design:

1. **Paragraph splitting:** Text is first split by double newlines
2. **Sentence splitting:** Paragraphs exceeding `max_length` are split at sentence boundaries (`.!?`)
3. **Chunk aggregation:** Short sentences are merged until they reach `min_length`
4. **Fallback:** If insufficient chunks, split by single newlines

Parameters:
- `min_length = 50` characters
- `max_length = 500` characters

This typically yields 10-50 chunks per reasoning chain, significantly expanding the sample pool.

#### 5.9.3 Token-Based Classification

For each feature, texts are classified based on presence of the feature's top-k tokens:

```python
def text_contains_tokens(text, tokens, threshold=1):
    """Returns True if text contains at least `threshold` tokens."""
    matches = sum(1 for t in tokens if t.lower() in text.lower())
    return matches >= threshold
```

The top tokens are loaded from the token analysis results produced by `find_reasoning_features.py`.

#### 5.9.4 Activation Collection

For each text in each condition:
1. Tokenize with model's tokenizer (padding, truncation to `max_length`)
2. Run forward pass through the model
3. Extract residual stream activations at the target layer
4. Encode through SAE to get feature activations
5. Take maximum activation across sequence positions

#### 5.9.5 ANOVA Computation

For balanced design with $n$ samples per cell:

**Cell means:**
- $\bar{a}_A$: Mean activation for Reasoning + Has Tokens
- $\bar{a}_B$: Mean activation for Reasoning + No Tokens
- $\bar{a}_C$: Mean activation for Non-reasoning + Has Tokens
- $\bar{a}_D$: Mean activation for Non-reasoning + No Tokens

**Marginal means:**
- $\bar{a}_{\text{has\_tokens}} = (\bar{a}_A + \bar{a}_C) / 2$
- $\bar{a}_{\text{no\_tokens}} = (\bar{a}_B + \bar{a}_D) / 2$
- $\bar{a}_{\text{reasoning}} = (\bar{a}_A + \bar{a}_B) / 2$
- $\bar{a}_{\text{nonreasoning}} = (\bar{a}_C + \bar{a}_D) / 2$

**Sum of squares (balanced design):**

$$SS_{\text{token}} = 2n \cdot [(\bar{a}_{\text{has\_tokens}} - \bar{a}_{\cdot\cdot})^2 + (\bar{a}_{\text{no\_tokens}} - \bar{a}_{\cdot\cdot})^2]$$

$$SS_{\text{behavior}} = 2n \cdot [(\bar{a}_{\text{reasoning}} - \bar{a}_{\cdot\cdot})^2 + (\bar{a}_{\text{nonreasoning}} - \bar{a}_{\cdot\cdot})^2]$$

**F-statistics and p-values:** Computed using the F-distribution with appropriate degrees of freedom.

#### 5.9.6 Decision Criteria

A feature is classified as:

| Classification | Criteria |
|---------------|----------|
| **Token-dominated** | $\eta^2_{\text{token}} > 2 \cdot \eta^2_{\text{behavior}}$ AND $\eta^2_{\text{token}} > 0.06$ |
| **Behavior-dominated** | $\eta^2_{\text{behavior}} > 2 \cdot \eta^2_{\text{token}}$ AND $\eta^2_{\text{behavior}} > 0.06$ |
| **Interaction-dominated** | $\eta^2_{\text{interaction}} > \max(\eta^2_{\text{token}}, \eta^2_{\text{behavior}})$ AND $\eta^2_{\text{interaction}} > 0.06$ |
| **Mixed** | No single factor dominates but effects are non-negligible |
| **None** | All effects are below 0.01 (negligible) |

The threshold of 0.06 corresponds to a "medium" effect size in the η² scale.

#### 5.9.7 Usage Examples

**Text-Level ANOVA (Original Approach):**
```bash
# Run ANOVA experiment for layer 8 features using s1K dataset
python reasoning_features/scripts/run_anova_experiment.py \
    --token-analysis results/layer8/token_analysis.json \
    --layer 8 \
    --reasoning-dataset s1k \
    --top-k-features 50 \
    --top-k-tokens 10 \
    --n-per-condition 200 \
    --save-dir results/anova/layer8
```

**Token-Level ANOVA (Recommended):**
```bash
# Run token-level ANOVA with feature-specific tokens
python reasoning_features/scripts/run_anova_tokenwise.py \
    --token-analysis results/layer8/token_analysis.json \
    --layer 8 \
    --top-k-features 10 \
    --top-k-tokens 50 \
    --token-strategy feature_specific \
    --save-dir results/anova/layer8/tokenwise

# Run with global reasoning tokens (tokens overrepresented in reasoning text)
python reasoning_features/scripts/run_anova_tokenwise.py \
    --token-analysis results/layer8/token_analysis.json \
    --layer 8 \
    --token-strategy global_reasoning \
    --global-top-k 300 \
    --save-dir results/anova/layer8/tokenwise_global
```

**Refined ANOVA (Context-Agnostic Tokens Only):**
```bash
# Analyze only tokens appearing in both contexts
python reasoning_features/scripts/run_anova_refined.py \
    --token-analysis results/layer8/token_analysis.json \
    --layer 8 \
    --top-k-features 10 \
    --min-occurrences-both 5 \
    --save-dir results/anova/layer8/refined
```

#### 5.9.8 Output Format

The experiment produces `anova_results.json` with the following structure:

```json
{
  "config": {
    "token_analysis": "path/to/token_analysis.json",
    "reasoning_dataset": "s1k",
    "layer": 8,
    "top_k_features": 50,
    "top_k_tokens": 10,
    "n_per_condition": 200
  },
  "summary": {
    "n_features_analyzed": 45,
    "n_token_dominated": 32,
    "n_behavior_dominated": 5,
    "pct_token_dominated": 71.1,
    "pct_behavior_dominated": 11.1,
    "mean_eta_sq_token": 0.142,
    "mean_eta_sq_behavior": 0.038,
    "dominant_factor_distribution": {
      "token": 32,
      "behavior": 5,
      "mixed": 6,
      "none": 2
    }
  },
  "features": [
    {
      "feature_index": 42,
      "eta_sq_token": 0.23,
      "eta_sq_behavior": 0.05,
      "eta_sq_interaction": 0.02,
      "f_token": 89.5,
      "f_behavior": 18.3,
      "p_token": 1e-15,
      "p_behavior": 2e-5,
      "mean_A": 3.21,
      "mean_B": 1.45,
      "mean_C": 2.89,
      "mean_D": 0.98,
      "is_token_dominated": true,
      "dominant_factor": "token"
    }
  ]
}
```

---

## 6. Empirical Validation

### 6.1 Preliminary Results

Analysis of our initial experiments on Gemma-2-2B with gemma-scope SAEs reveals:

#### Token Dependency by Layer

| Dataset | Layer 0 | Layer 8 | Layer 16 | Layer 24 |
|---------|---------|---------|----------|----------|
| s1k | 80.0% | 0.0% | 5.0% | 15.0% |
| General Inquiry CoT | 95.0% | 35.0% | 75.0% | 60.0% |

**Observation:** Early layers show high token dependency, which decreases in middle layers but partially recovers in later layers for certain datasets.

#### Top Tokens Driving Reasoning Features

**s1k dataset:** "asks", "Prove", "Solve", "Problem", "Let", "Consider"

**General Inquiry CoT:** "need", "should", "I", "First", "Next", "must"

These are precisely the shallow lexical cues we hypothesized would dominate.

#### Correlation Analysis

| Metric Pair | s1k | General Inquiry CoT |
|-------------|-----|---------------------|
| Reasoning Score vs. Token Concentration | -0.557 | -0.035 |
| ROC-AUC vs. Token Concentration | -0.701 | -0.317 |
| Cohen's d vs. Token Concentration | -0.376 | +0.013 |

**Interpretation:** 
- In s1k, features with higher reasoning scores tend to have *lower* token concentration—these may be more genuine reasoning features
- In General Inquiry CoT, no clear relationship—token dependency is consistently high regardless of reasoning score

### 6.2 Implications for Experimental Design

Based on these preliminary findings, we refine our approach:

1. **Layer selection:** Focus analysis on middle layers (8-16) where token dependency is lowest but reasoning signals remain strong

2. **Dataset selection:** Use s1k for primary analysis due to lower baseline token dependency; use General Inquiry CoT as a "hard case" where token cues dominate

3. **Feature selection for steering:** Prioritize features with high reasoning score AND low token concentration for the most informative steering experiments

---

## 7. Improved Feature Selection for Steering

### 7.1 The Problem with Naive Selection

Simply selecting features with the highest reasoning score for steering experiments is problematic because many high-scoring features may be dominated by token-level correlations. Our empirical analysis reveals:

- **s1k dataset**: Strong negative correlation (r = -0.83) between reasoning score and token concentration
- **General Inquiry CoT**: Moderate negative correlation (r = -0.72) but only 13% of top features have low token dependency

### 7.2 Combined Selection Score

We propose a combined score that balances reasoning strength with token independence:

$$\mathrm{CombinedScore} = w_r \cdot \frac{\mathrm{ReasoningScore}}{\max(\mathrm{ReasoningScore})} + w_t \cdot \mathrm{TokenIndependence},$$

where:

$$\mathrm{TokenIndependence} = (1 - \mathrm{TokenConcentration}) \cdot (\mathrm{NormalizedEntropy} + 0.1).$$

Default weights: $w_r = 0.6$, $w_t = 0.4$.

### 7.3 Implementation

The `FeatureSelector` class (see `reasoning_features/features/selection.py`) provides:

1. `get_top_features(k, require_low_token_dependency)`: Combined score ranking with optional filtering
2. `get_features_by_reasoning_only(k)`: Pure reasoning score ranking
3. `get_features_by_token_independence(k)`: Pure token independence ranking
4. `compare_selection_strategies(k)`: Quantitative comparison of methods

### 7.4 Recommended Usage

For steering experiments, we recommend:

```python
from reasoning_features.features import load_and_select_features

# Select features with both high reasoning scores AND low token dependency
indices = load_and_select_features(
    "results/layer8/reasoning_features.json",
    "results/layer8/token_analysis.json",
    top_k=20,
)
```

This ensures we steer features that are more likely to capture genuine reasoning patterns rather than surface-level token correlations.

---

## 8. Limitations and Future Work

### 8.1 Known Limitations

1. **Token set definition:** The "reasoning tokens" we identify are based on frequency analysis; a more principled approach would use linguistic annotation

2. **Binary behavior classification:** Reasoning exists on a spectrum; binary classification may miss nuances

3. **Causal confounds:** Even with 2×2 ANOVA, we cannot fully rule out unmeasured confounds

### 8.2 Future Directions

1. **Gradient-based attribution:** Use input gradients to identify which tokens causally drive feature activations

2. **Intervention studies:** Systematically ablate tokens in context and measure feature response

3. **Cross-model validation:** Replicate findings across different model families and sizes

---

## 9. Summary

This methodology provides a rigorous framework for investigating whether SAE "reasoning features" capture genuine reasoning ability or merely token-level correlations. Our approach includes:

1. **Multi-metric detection** using ROC-AUC, Cohen's d, Mann-Whitney U, and a composite score
2. **Token dependency analysis** via concentration, entropy, and PMI metrics
3. **Multiplicative steering** of multiple features with evaluation on reasoning benchmarks
4. **2×2 ANOVA** to disentangle token and behavior effects
5. **Improved feature selection** that balances reasoning strength with token independence

Preliminary results strongly suggest that many "reasoning features" are dominated by shallow token cues (e.g., "Let", "Prove", "First"), supporting our central hypothesis. The ANOVA analysis (upcoming) will provide definitive evidence for or against this hypothesis.

---

## References

1. Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. 2nd ed. Hillsdale, NJ: Lawrence Erlbaum Associates.
2. Mann, H. B., & Whitney, D. R. (1947). On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other. *Annals of Mathematical Statistics*, 18(1), 50-60.
3. Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. *arXiv:2309.08600*.
4. Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Anthropic Research.
5. Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Anthropic Research.
