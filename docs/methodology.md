# Methodology and Experimental Design

## 1. Defining Reasoning Features in Sparse Autoencoders

Sparse Autoencoders (SAEs) have emerged as a promising technique for decomposing neural network activations into interpretable feature directions. However, the extent to which these features capture high-level cognitive processes, such as reasoning, remains an open question. We define **genuine reasoning features** as SAE feature directions that respond specifically to the presence of abstract reasoning processes—including but not limited to logical deduction, multi-step problem-solving, counterfactual analysis, and deliberative planning—rather than surface-level lexical or syntactic correlates of reasoning text.

Formally, let $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ denote a residual stream activation at layer $\ell$ of a transformer language model, and let $\mathbf{f} = \text{SAE}_{\text{enc}}(\mathbf{x}) \in \mathbb{R}^{d_{\text{SAE}}}$ represent the corresponding sparse feature activations. A feature $f_i$ is considered a **genuine reasoning feature** if and only if:

1. **Reasoning Specificity**: The feature activates systematically on text exhibiting reasoning behavior across diverse contexts, independent of superficial lexical patterns.
2. **Non-spurious Correlation**: The feature does not activate on non-reasoning text that contains reasoning-associated tokens (e.g., "therefore," "let's consider," "step") or syntactic structures common in reasoning corpora.
3. **Semantic Invariance**: The feature remains robust to paraphrases and stylistic variations that preserve the underlying reasoning structure.

This definition deliberately excludes features that activate on shallow token-level cues. For instance, a feature that responds to the token "wait" or the phrase "Let me think" may correlate with reasoning text but does not capture the cognitive process of reasoning itself. Such features represent spurious correlates rather than genuine reasoning mechanisms. This distinction is critical: prior work has demonstrated that simple token-level interventions—such as suppressing end-of-thinking tokens and replacing them with "wait"—can yield substantial performance gains on reasoning benchmarks without engaging SAEs whatsoever (e.g., improvements from 0% to nearly 60% on AIME 2024 for 32B models). Our investigation therefore seeks features that capture reasoning *processes*, not merely reasoning *vocabulary*.

## 2. Identifying Candidate Reasoning Features

### 2.1 Statistical Framework

We adopt a rigorous statistical approach to identify SAE features that exhibit differential activation between reasoning and non-reasoning text. Let $\mathcal{D}_{\text{R}}$ denote a corpus of reasoning text and $\mathcal{D}_{\text{NR}}$ denote a corpus of non-reasoning text. For each feature $f_i$, we compute the maximum activation across all tokens in each text sample, yielding distributions $\{a_{i,j}^{\text{R}}\}_{j=1}^{n_{\text{R}}}$ and $\{a_{i,k}^{\text{NR}}\}_{k=1}^{n_{\text{NR}}}$ for reasoning and non-reasoning samples, respectively.

### 2.2 Cohen's d Effect Size

Our primary metric for feature selection is **Cohen's d**, which quantifies the standardized mean difference between the two distributions:

$$
d_i = \frac{\bar{a}_i^{\text{R}} - \bar{a}_i^{\text{NR}}}{s_{\text{pooled}}}, \quad \text{where} \quad s_{\text{pooled}} = \sqrt{\frac{(n_{\text{R}} - 1)(s_i^{\text{R}})^2 + (n_{\text{NR}} - 1)(s_i^{\text{NR}})^2}{n_{\text{R}} + n_{\text{NR}} - 2}}.
$$

Here, $\bar{a}_i^{\text{R}}$ and $\bar{a}_i^{\text{NR}}$ are the sample means, and $s_i^{\text{R}}$, $s_i^{\text{NR}}$ are the sample standard deviations. Cohen's d provides an effect size measure that is independent of sample size, making it suitable for comparing features across different experimental conditions.

We rank all SAE features by their Cohen's d values and select the top $k$ features for further analysis. We impose minimum thresholds to ensure statistical rigor:

- **Minimum effect size**: $|d_i| \geq 0.3$ (small-to-medium effect per Cohen's conventions)
- **Statistical significance**: $p \leq 0.01$ after Bonferroni correction (Mann-Whitney U test)
- **Classification performance**: ROC-AUC $\geq 0.6$ (better than random)

These thresholds ensure that selected features demonstrate both practical significance (effect size) and statistical reliability (p-value), while maintaining discriminative power between reasoning and non-reasoning text.

### 2.3 Auxiliary Metrics (Appendix)

While Cohen's d serves as our primary ranking criterion, we also compute complementary metrics for robustness analysis reported in the appendix:

**ROC-AUC** (Receiver Operating Characteristic - Area Under Curve):
$$
\mathrm{AUC}_i = \mathbb{P}(a_{i,j}^{\text{R}} > a_{i,k}^{\text{NR}}) = \frac{1}{n_{\text{R}} \cdot n_{\text{NR}}} \sum_{j=1}^{n_{\text{R}}} \sum_{k=1}^{n_{\text{NR}}} \mathbb{1}[a_{i,j}^{\text{R}} > a_{i,k}^{\text{NR}}].
$$

**Frequency Ratio**:
$$
\mathrm{FreqRatio}_i = \frac{\mathrm{freq}_i^{\text{R}} + \epsilon}{\mathrm{freq}_i^{\text{NR}} + \epsilon},
$$
where $\text{freq}_i^{\text{R}}$ and $\text{freq}_i^{\text{NR}}$ denote the proportion of samples with non-zero activation, and $\epsilon = 0.01$ is a smoothing constant.

## 3. Token Dependency Analysis and Injection Experiments

### 3.1 Identifying Top-Activating Tokens

A critical step in our methodology is determining whether features respond to specific lexical items or n-gram patterns rather than abstract reasoning processes. For each selected feature $f_i$, we identify the top-$k$ tokens, bigrams, and trigrams that most strongly activate the feature.

**Token Ranking**: For each unique token $t$ in the reasoning corpus, we compute its mean activation:
$$
\bar{a}_{i,t} = \frac{1}{|\{j : \text{token}_j = t\}|} \sum_{j : \text{token}_j = t} a_{i,j},
$$
where $a_{i,j}$ denotes the feature activation at position $j$. Tokens are ranked by $\bar{a}_{i,t}$ in descending order.

**Bigram and Trigram Ranking**: We extend this analysis to consecutive token sequences, computing mean activations for all bigrams and trigrams that appear at least $m$ times in the corpus (we use $m = 3$ for bigrams and $m = 2$ for trigrams to balance statistical reliability with coverage).

This analysis serves two purposes: (1) it provides interpretable proxies for what patterns drive feature activation, and (2) it enables the construction of targeted interventions to test whether these patterns are sufficient to trigger activation in non-reasoning contexts.

### 3.2 Causal Token Injection Framework

The core insight of our token injection experiment is that **superficial token-level cues do not constitute reasoning**. If a feature captures genuine reasoning, then injecting its top-activating tokens into non-reasoning text should not substantially increase feature activation, as the text remains non-reasoning in nature. Conversely, if the feature is merely a token detector, injection should produce activation levels comparable to authentic reasoning text.

**Experimental Design**: For each feature $f_i$, we construct three conditions:

1. **Baseline**: Non-reasoning samples from $\mathcal{D}_{\text{NR}}$ (each truncated to 64 tokens)
2. **Target**: Reasoning samples from $\mathcal{D}_{\text{R}}$ (truncated to 64 tokens)
3. **Injected**: Non-reasoning samples with top-$k$ tokens/n-grams injected

We employ a diverse set of injection strategies to test different hypotheses about feature sensitivity:

**Simple Token Injection** (inject 3 tokens):
- **Prepend**: Insert tokens at the beginning of the text
- **Append**: Insert tokens at the end of the text
- **Intersperse**: Distribute tokens uniformly throughout the text
- **Replace**: Substitute random words with injected tokens

**N-gram Injection** (inject 2 bigrams or 1 trigram):
- **Inject Bigram**: Insert top-activating bigrams identified from token analysis
- **Inject Trigram**: Insert top-activating trigrams

**Contextual Injection**:
- **Bigram Before**: Insert `[context, token]` pairs where `context` is a frequently preceding word
- **Bigram After**: Insert `[token, context]` pairs where `context` is a frequently following word
- **Trigram**: Insert `[before, token, after]` triplets based on context analysis
- **Comma List**: Insert tokens as a comma-separated enumeration

The rationale for testing multiple strategies is that features may exhibit varying degrees of context-sensitivity. Simple prepending may fail to activate context-dependent features, while more sophisticated contextual injection strategies preserve local co-occurrence patterns.

### 3.3 Effect Size Quantification and Classification

We measure the activation increase induced by token injection using Cohen's d:

$$
d_{i}^{\text{inject}} = \frac{\bar{a}_i^{\text{inject}} - \bar{a}_i^{\text{baseline}}}{s_{\text{pooled}}},
$$

where $\bar{a}_i^{\text{inject}}$ and $\bar{a}_i^{\text{baseline}}$ are the mean activations on injected and baseline non-reasoning samples, respectively.

Following established conventions in psychology and social sciences (Cohen, 1988), we classify features based on their injection effect size:

- **Token-driven** ($d \geq 0.8$, $p < 0.01$): Large effect—79% of injected samples exceed baseline median; tokens strongly activate the feature
- **Partially token-driven** ($0.5 \leq d < 0.8$, $p < 0.01$): Medium effect—69% of injected samples exceed baseline median
- **Weakly token-driven** ($0.2 \leq d < 0.5$, $p < 0.05$): Small effect—58% of injected samples exceed baseline median
- **Context-dependent** ($d < 0.2$ or $p \geq 0.05$): Negligible effect—token injection does not meaningfully increase activation

The threshold $d = 0.2$ is particularly important: it represents the boundary below which the effect is considered negligible in magnitude. Features classified as context-dependent warrant further investigation, as they may capture patterns beyond simple token presence.

**Statistical Testing**: We employ independent t-tests to assess the statistical significance of activation differences between injected and baseline conditions, using $\alpha = 0.01$ for large and medium effects, and $\alpha = 0.05$ for small effects. For each feature, we select the best-performing injection strategy (highest Cohen's d) for classification.

## 4. LLM-Guided Feature Interpretation

### 4.1 Motivation and Limitations of Heuristic Approaches

While token injection experiments provide strong evidence for shallow token-level patterns, they face a fundamental limitation: **the space of possible linguistic confounds is intractably large**. A feature may respond to sophisticated patterns—formal academic discourse markers, complex syntactic structures, abstract vocabulary, prose sophistication—that our predefined injection strategies fail to test. Such features would be classified as "context-dependent" despite not capturing genuine reasoning.

To address this limitation, we employ a large language model (Google Gemini 3 Pro) as an intelligent hypothesis-testing agent capable of systematically exploring the space of linguistic patterns and generating targeted counterexamples.

### 4.2 Iterative Hypothesis Testing Protocol

For each context-dependent feature identified in Section 3, we conduct an iterative analysis consisting of the following phases:

**Phase 1: Hypothesis Generation**

Given:
- Top-$k$ tokens ranked by mean activation (from Section 3.1)
- $N$ high-activation text samples from $\mathcal{D}_{\text{R}}$
- Token-level activation patterns within each sample

The LLM generates an initial hypothesis about what linguistic pattern the feature detects, considering:
- Lexical patterns (vocabulary, word categories, semantic fields)
- Syntactic patterns (clause structures, sentence complexity, grammatical constructions)
- Discourse patterns (hedging, meta-cognition, transition markers, epistemic stance)
- Stylistic patterns (formality, register, verbosity, complexity)

**Phase 2: Counterexample Generation**

The LLM generates two types of counterexamples to falsify the reasoning hypothesis:

1. **False Positives (FP)**: Non-reasoning text predicted to activate the feature
   - Strategy: Generate non-reasoning content (recipes, product reviews, news articles, fiction) that contains the hypothesized linguistic pattern
   - Criterion: $\max(a_i(\text{candidate})) > \tau$, where $\tau = 0.5 \times \max_{j} a_i(x_j^{\text{R}})$

2. **False Negatives (FN)**: Reasoning text predicted NOT to activate the feature
   - Strategy: Generate reasoning content that avoids the hypothesized pattern (casual language, simple syntax, different vocabulary)
   - Criterion: $\max(a_i(\text{candidate})) < 0.1\tau$

In each iteration, the LLM generates 5 candidates per category (10 total). Each candidate is evaluated against the actual model to determine whether it constitutes a valid counterexample.

**Phase 3: Iterative Refinement**

Valid counterexamples from previous iterations inform subsequent generation:
- **Successful patterns** (valid counterexamples) are reinforced and explored further
- **Failed patterns** (invalid counterexamples) are avoided in future iterations

The process continues until either:
- Sufficient counterexamples are found ($\geq 3$ false positives AND $\geq 3$ false negatives), OR
- Maximum iterations $T = 10$ is reached

**Phase 4: Final Classification**

After all iterations, the LLM provides:
1. **Refined interpretation**: What the feature actually detects
2. **Activation conditions**: Specific content types and structures that activate the feature
3. **Non-activation conditions**: Content types that do not activate the feature
4. **Confidence level**: HIGH/MEDIUM/LOW based on consistency of evidence
5. **Binary classification**: "Genuine reasoning feature" (true) or "Confound" (false)

A feature is classified as **genuine reasoning** only if:
- It activates specifically on reasoning/thinking/deliberation processes
- It does NOT activate on non-reasoning content (few or no false positives)
- It activates on diverse types of reasoning (few or no false negatives)

Otherwise, the feature is classified as a **confound**—a linguistic pattern that correlates with reasoning in training data but does not capture the reasoning process itself.

### 4.3 Advantages of LLM-Guided Analysis

This automated approach provides several advantages over manual analysis:

1. **Systematic coverage**: Tests diverse linguistic patterns beyond manually-designed heuristics
2. **Scalability**: Can analyze hundreds of features without human intervention
3. **Consistency**: Applies uniform evaluation criteria across all features
4. **Iterative refinement**: Learns from empirical feedback to improve hypothesis testing
5. **Explainability**: Generates human-interpretable descriptions with concrete counterexamples

The LLM's natural language generation capabilities enable it to propose creative variations that would be difficult to enumerate manually, while the iterative empirical validation ensures that classifications are grounded in the model's actual behavior rather than the LLM's priors.

## 5. Steering Experiments

To further validate our findings, we conduct a supplementary steering experiment on a subset of top-ranked features. While this is not a primary component of our methodology (as our central claim concerns the *absence* of genuine reasoning features), it provides additional evidence regarding the nature of the features identified in Section 2.

**Steering Formula**: For a feature $f_i$ with decoder direction $\mathbf{W}_{\text{dec},i} \in \mathbb{R}^{d_{\text{model}}}$, we modify the residual stream at layer $\ell$ using:

$$
\mathbf{x}' = \mathbf{x} + \gamma \cdot f_i^{\max} \cdot \mathbf{W}_{\text{dec},i},
$$

where $\gamma \in \mathbb{R}$ is the steering strength and $f_i^{\max}$ is the maximum activation of feature $i$ observed in the reasoning corpus. Positive $\gamma$ amplifies the feature, while negative $\gamma$ suppresses it.

**Evaluation**: We steer the top 3 features (ranked by Cohen's d) on layer 22 of Gemma-3-12B-Instruct and evaluate performance on two challenging reasoning benchmarks:
- **AIME 2024**: Advanced mathematics competition problems (30 questions, numerical answers)
- **GPQA Diamond**: Graduate-level science questions (198 questions, multiple choice)

We test $\gamma \in \{-2, -1, 0, 1, 2\}$ and report one-shot accuracy with chain-of-thought prompting.

**Interpretation**: If the identified features captured genuine reasoning, we would expect positive $\gamma$ to improve accuracy. However, it is critical to note that **performance improvement does not imply genuine reasoning capture**. As demonstrated in recent work on test-time scaling (e.g., Simple Scaling, 2024), superficial token-level interventions can yield substantial gains. For instance, merely suppressing end-of-thinking tokens and replacing them with "wait" improves AIME 2024 accuracy from 0% to nearly 60% on a 32B model—without using SAEs at all. Thus, our steering experiment serves as a sanity check but not as definitive evidence of reasoning feature existence.

## 6. Experimental Setup and Results

### 6.1 Models and Architectures

We conduct experiments on three state-of-the-art open-weight language models:

1. **Gemma-3-12B-Instruct** (12.4B parameters, 42 layers)
   - Layer analyzed: 22 (middle layer)
   - SAE: `gemma-scope-2-12b-it-res-all` (width: 16,384, L0: small)

2. **Gemma-3-4B-Instruct** (4.5B parameters, 28 layers)
   - Layer analyzed: 17 (middle layer)
   - SAE: `gemma-scope-2-4b-it-res-all` (width: 16,384, L0: small)

3. **DeepSeek-R1-Distill-Llama-8B** (8B parameters, 32 layers)
   - Layer analyzed: 19 (middle layer)
   - SAE: `gemma-scope-2-8b-it-res-all` (width: 16,384, L0: small)

All models are instruction-tuned variants with established proficiency on reasoning tasks. We focus on middle layers (approximately 50-70% depth) where prior work suggests reasoning-related activations may be most prominent. All SAEs are obtained from the GemmaScope release and use residual stream activations with small L0 regularization for enhanced sparsity.

For comparison, we also report results on Gemma-2-9B (layer 21) and Gemma-2-2B (layer 13) in the appendix.

**Hardware**: All experiments were conducted on a single NVIDIA A100 80GB GPU. Feature detection and token injection experiments required approximately 2-4 hours per model-layer-dataset configuration, while LLM interpretation required 8-12 hours per configuration due to iterative API calls.

### 6.2 Datasets

We curate three datasets to distinguish reasoning from non-reasoning text:

**Reasoning Datasets** ($\mathcal{D}_{\text{R}}$):

1. **s1K-1.1** (HuggingFace: `simplescaling/s1K-1.1`):
   - 1,000 challenging mathematics problems with reasoning traces generated by DeepSeek-R1
   - Domains: algebra, geometry, number theory, combinatorics, probability
   - We extract reasoning traces from the `deepseek_thinking_trajectory` and `gemini_thinking_trajectory` fields
   - Samples: 500 texts, each truncated to 64 tokens

2. **General Inquiry Thinking Chain-of-Thought** (HuggingFace: `moremilk/General_Inquiry_Thinking-Chain-Of-Thought`):
   - 6,000 question-answer pairs spanning diverse domains (science, logic, philosophy, everyday reasoning)
   - Each entry includes explicit chain-of-thought reasoning in `<think>` tags
   - Samples: 500 texts (with `<think>` tags removed), each truncated to 64 tokens

**Non-Reasoning Dataset** ($\mathcal{D}_{\text{NR}}$):

3. **Pile (Uncopyrighted)** (HuggingFace: `monology/pile-uncopyrighted`):
   - General web text with all copyrighted content removed
   - Includes Wikipedia, books, academic papers, web crawl data
   - Samples: 500 texts, each truncated to 64 tokens

**Rationale for Truncation**: We truncate all samples to 64 tokens to ensure: (1) computational efficiency during activation collection, (2) uniform sequence length for fair comparison, and (3) sufficient context for SAE features to activate while avoiding confounds from document-level statistics.

For each model and layer, we conduct experiments with both reasoning datasets separately, yielding two independent sets of features that may overlap. This dual-dataset approach provides robustness against dataset-specific biases.

### 6.3 Feature Detection Results

**Hyperparameters**:
- Minimum Cohen's d: 0.3
- Maximum p-value (Bonferroni-corrected): 0.01
- Minimum ROC-AUC: 0.6
- Top-k features selected: 100 (ranked by Cohen's d)
- Samples per dataset: 500 reasoning, 500 non-reasoning
- Sequence length: 64 tokens (truncated)
- Aggregation: Maximum activation across all tokens in each sequence

**Table 1**: Number of features meeting statistical thresholds and mean Cohen's d values.

| Model | Layer | Dataset | Features Found | Mean Cohen's d | Mean ROC-AUC |
|-------|-------|---------|----------------|----------------|--------------|
| Gemma-3-12B-IT | 22 | s1K | 99 | 0.859 | 0.691 |
| Gemma-3-12B-IT | 22 | General Inquiry CoT | 100 | 0.836 | 0.694 |
| Gemma-3-4B-IT | 17 | s1K | 100 | 0.804 | 0.669 |
| Gemma-3-4B-IT | 17 | General Inquiry CoT | 100 | 0.917 | 0.690 |
| DeepSeek-R1-Distill-Llama-8B | 19 | s1K | 100 | 0.675 | 0.670 |
| DeepSeek-R1-Distill-Llama-8B | 19 | General Inquiry CoT | 100 | 0.781 | 0.697 |

**Figure 1 (Placeholder)**: Distribution of Cohen's d values across all 16,384 SAE features for Gemma-3-12B-IT layer 22 on s1K dataset. The plot shows a long tail distribution with the top 100 features (shaded) having $d \geq 0.3$.

**Figure 2 (Placeholder)**: Scatter plot of Cohen's d vs. ROC-AUC for top 100 features across all models and datasets. Points cluster in the upper-right quadrant, indicating features with both large effect sizes and strong discriminative power.

All configurations successfully identified 99-100 features exceeding our statistical thresholds, demonstrating that SAE features can reliably distinguish between reasoning and non-reasoning text in a statistical sense. The mean Cohen's d values range from 0.675 to 0.917, indicating medium-to-large effect sizes. These features serve as candidate "reasoning features" for subsequent causal investigation.

### 6.4 Token Injection Experimental Results

For each configuration, we analyze the top 100 features identified in Section 6.3. We extract the top 10 tokens, top 20 bigrams, and top 10 trigrams for each feature based on mean activation. We then conduct injection experiments using 500 non-reasoning samples (baseline) and 500 reasoning samples (target).

**Hyperparameters**:
- Top-k tokens per feature: 10
- Tokens injected (simple strategies): 3
- Bigrams injected: 2
- Trigrams injected: 1
- Injection strategies: prepend, append, intersperse, replace, inject_bigram, inject_trigram, bigram_before, bigram_after, trigram, comma_list
- Classification thresholds:
  - Token-driven: $d \geq 0.8$, $p < 0.01$
  - Partially token-driven: $0.5 \leq d < 0.8$, $p < 0.01$
  - Weakly token-driven: $0.2 \leq d < 0.5$, $p < 0.05$
  - Context-dependent: $d < 0.2$ or $p \geq 0.05$

**Table 2**: Token injection classification results. Each entry shows the number (percentage) of features in each category.

| Model | Layer | Dataset | Token-Driven | Partially TD | Weakly TD | Context-Dep | Avg Cohen's d |
|-------|-------|---------|--------------|--------------|-----------|-------------|---------------|
| Gemma-3-12B-IT | 22 | s1K | 36 (36.4%) | 20 (20.2%) | 24 (24.2%) | 19 (19.2%) | 0.871 |
| Gemma-3-12B-IT | 22 | Gen. Inq. | 46 (46.0%) | 20 (20.0%) | 26 (26.0%) | 8 (8.0%) | 0.997 |
| Gemma-3-4B-IT | 17 | s1K | 46 (46.0%) | 23 (23.0%) | 18 (18.0%) | 13 (13.0%) | 0.922 |
| Gemma-3-4B-IT | 17 | Gen. Inq. | 39 (39.0%) | 26 (26.0%) | 18 (18.0%) | 17 (17.0%) | 0.946 |
| DeepSeek-R1 | 19 | s1K | 46 (46.0%) | 12 (12.0%) | 19 (19.0%) | 23 (23.0%) | 0.899 |
| DeepSeek-R1 | 19 | Gen. Inq. | 25 (25.0%) | 14 (14.0%) | 20 (20.0%) | 41 (41.0%) | 0.521 |

**Key Findings**:

1. **Majority are token-driven**: Across all configurations, 25-46% of features are classified as strongly token-driven (large effect), with an additional 12-26% being partially token-driven and 18-26% weakly token-driven. This indicates that **60-92% of putative "reasoning features" can be substantially activated by injecting a small number of tokens into non-reasoning text**.

2. **Context-dependent minority**: Only 8-41% of features (13-41 out of 100) are classified as context-dependent, suggesting that the majority of features respond to token-level patterns that can be captured by our injection strategies.

3. **Dataset variation**: The General Inquiry CoT dataset on DeepSeek-R1 shows the highest proportion of context-dependent features (41%), while the same dataset on Gemma-3-12B shows the lowest (8%), suggesting model-specific differences in feature composition.

4. **High average effect sizes**: The average Cohen's d values (0.521-0.997) indicate that token injection produces large activation increases even when injecting only 3 tokens into 64-token sequences (4.7% of tokens).

**Figure 3 (Placeholder)**: Bar chart showing the distribution of feature classifications across all six configurations. Each bar is segmented by category (token-driven, partially token-driven, weakly token-driven, context-dependent) with different colors.

**Figure 4 (Placeholder)**: Box plots comparing baseline activation, injected activation, and reasoning activation for a representative feature (e.g., Feature 3420 from Gemma-3-4B-IT layer 17 s1K). The injected and reasoning distributions should substantially overlap for token-driven features.

These results provide strong evidence that the majority of statistically-identified "reasoning features" are in fact shallow token detectors. Even features with large Cohen's d values (high statistical correlation with reasoning text) can be substantially activated by simple lexical interventions that do not introduce genuine reasoning.

### 6.5 LLM-Guided Interpretation Results

For each configuration, we randomly sample up to 20 context-dependent features from the token injection experiment and subject them to LLM-guided analysis. We use Google Gemini 3 Pro (via OpenRouter API) with the iterative protocol described in Section 4.2.

**Hyperparameters**:
- LLM model: `google/gemini-3-pro-preview`
- Maximum iterations: 10
- Minimum false positives required: 3
- Minimum false negatives required: 3
- Activation threshold: $\tau = 0.5 \times \max(\text{reasoning activations})$
- Temperature (generation): 0.8 (for diversity in counterexample generation)
- Temperature (interpretation): 0.3 (for consistency in final classification)

**Table 3**: LLM interpretation results for context-dependent features.

| Model | Layer | Dataset | Features Analyzed | Genuine Reasoning | Confounds | High Conf | Med Conf | Low Conf |
|-------|-------|---------|-------------------|-------------------|-----------|-----------|----------|----------|
| Gemma-3-12B-IT | 22 | s1K | 18 | 0 | 18 | 18 | 0 | 0 |
| Gemma-3-12B-IT | 22 | Gen. Inq. | 7 | 0 | 7 | 5 | 0 | 2 |
| Gemma-3-4B-IT | 17 | s1K | 13 | 0 | 13 | 10 | 0 | 3 |
| Gemma-3-4B-IT | 17 | Gen. Inq. | 17 | 0 | 17 | 16 | 0 | 1 |
| DeepSeek-R1 | 19 | s1K | 20 | 0 | 20 | 14 | 0 | 6 |
| DeepSeek-R1 | 19 | Gen. Inq. | 19 | 0 | 19 | 18 | 0 | 1 |

**Main Finding**: **Across all 94 context-dependent features analyzed, zero were classified as genuine reasoning features**. Every feature was identified as a confound—a linguistic pattern that correlates with reasoning text but does not capture the reasoning process itself.

**Common Confounds Discovered**:

The LLM analysis revealed several recurring patterns among the purported "reasoning features":

1. **Conversational introductory sequences** (e.g., "Okay, let's see," "So," "Now let's tackle") combined with technical vocabulary. These features activate on both problem-solving text and static descriptions with similar phrasing.

2. **Formal academic discourse markers** (e.g., "Furthermore," "Consequently," "In light of," "Given that") that appear frequently in reasoning text but also in expository and descriptive writing.

3. **Meta-cognitive planning phrases** (e.g., "First, we need to," "The next step is," "Before we proceed") that indicate procedural organization but not necessarily reasoning.

4. **Complex syntactic structures** (e.g., subordinate clauses, relative clauses, conditional constructions) that correlate with prose sophistication rather than reasoning per se.

5. **Abstract vocabulary and technical terminology** from mathematical, scientific, or philosophical domains that co-occur with reasoning but do not capture the reasoning process.

**Confidence Distribution**: The LLM expressed high confidence (based on consistency of counterexamples) for 81 out of 94 features (86.2%), indicating that the confounds are typically clear and robust. Low confidence cases often involved features with complex, multi-faceted activation patterns that proved difficult to characterize succinctly.

**Example Interpretation** (Feature 3420, Gemma-3-4B-IT layer 17 s1K):
- **Classification**: Confound (HIGH confidence)
- **Refined interpretation**: "This feature detects a specific lexical pattern combining conversational introductory fillers (specifically 'Okay, let's see,' 'So,' 'tackle') with technical, scientific, or structural vocabulary."
- **False positives found**: 5 (e.g., unboxing videos, product descriptions with conversational tone and technical terms)
- **False negatives found**: 5 (e.g., formal mathematical reasoning without conversational markers)
- **Iterations used**: 2

**Figure 5 (Placeholder)**: Pie chart showing the distribution of confound types across all 94 analyzed features. Major categories include: conversational markers (28%), formal discourse (22%), meta-cognitive phrases (19%), syntactic complexity (16%), abstract vocabulary (15%).

**Figure 6 (Placeholder)**: Example false positive and false negative sentences for representative features, with activation values annotated. This visualization demonstrates how features respond to surface patterns rather than reasoning content.

These results strongly corroborate our central hypothesis: **even features that survive token injection testing (i.e., exhibit context-dependence to simple lexical interventions) are not genuine reasoning features**. They are sophisticated confounds—linguistic patterns that correlate with reasoning in naturalistic data but can be decoupled from the reasoning process through careful counterexample generation.

### 6.6 Steering Experimental Results (Preliminary)

We conduct steering experiments on Gemma-3-12B-IT layer 22, using the top 3 features ranked by Cohen's d from the s1K dataset (Features: [TBD], [TBD], [TBD]).

**Hyperparameters**:
- Steering formula: $\mathbf{x}' = \mathbf{x} + \gamma \cdot f_i^{\max} \cdot \mathbf{W}_{\text{dec},i}$
- Gamma values: $\gamma \in \{-2, -1, 0, 1, 2\}$
- Benchmarks: AIME 2024 (30 problems), GPQA Diamond (198 problems)
- Generation: Maximum 32,768 tokens, temperature 0.6, top-p 0.95
- Prompt: One-shot chain-of-thought with exemplar
- Metric: Exact match accuracy (AIME), multiple-choice accuracy (GPQA)

**Table 4 (Placeholder)**: Steering results for top 3 features on AIME 2024 and GPQA Diamond.

| Feature | $\gamma$ | AIME Accuracy | GPQA Accuracy | Notes |
|---------|----------|---------------|---------------|-------|
| [TBD] | -2 | [TBD]% | [TBD]% | Strong suppression |
| [TBD] | -1 | [TBD]% | [TBD]% | Mild suppression |
| [TBD] | 0 | [TBD]% | [TBD]% | Baseline |
| [TBD] | 1 | [TBD]% | [TBD]% | Mild amplification |
| [TBD] | 2 | [TBD]% | [TBD]% | Strong amplification |

**Figure 7 (Placeholder)**: Line plot showing accuracy as a function of gamma for each feature on both benchmarks. Expected pattern: no systematic improvement with positive gamma, possibly slight degradation or no change.

**Interpretation**: As discussed in Section 5, performance improvement (if observed) would not constitute evidence of genuine reasoning capture, as superficial interventions can yield gains. Our results are expected to show [TBD: minimal change or slight degradation], consistent with features capturing spurious correlates rather than causal reasoning mechanisms.

## 7. Limitations and Future Directions

### 7.1 Dataset and Sampling Limitations

1. **Reasoning Dataset Coverage**: Our analysis focuses on two reasoning datasets (mathematical problem-solving and general inquiry). While these span important domains, they may not exhaust all forms of reasoning (e.g., commonsense reasoning, strategic planning, creative problem-solving). Future work should extend to additional reasoning paradigms.

2. **Sample Truncation**: We truncate all sequences to 64 tokens for computational efficiency. While this captures local reasoning patterns, it may miss long-range dependencies and document-level reasoning structures. Longer sequences may exhibit different feature activation patterns.

3. **Non-Reasoning Baseline**: We use the Pile as our non-reasoning corpus, which includes some expository and analytical text that may overlap with reasoning-adjacent patterns. A more stringent baseline (e.g., purely descriptive or narrative text) might yield different results.

### 7.2 SAE and Architecture Constraints

1. **SAE Variants**: We analyze SAEs trained with L0 regularization on residual stream activations. Other SAE architectures (e.g., attention-based, MLP-based, different sparsity penalties) may capture different feature types.

2. **Layer Selection**: We focus on middle layers (50-70% depth). Reasoning-related features might be more prominent in earlier layers (where input patterns are detected) or later layers (where output is shaped). A comprehensive layer-wise analysis would be valuable.

3. **Feature Capacity**: We analyze SAEs with 16,384 features. Higher-capacity SAEs may discover more fine-grained features, though our results suggest that capacity is not the primary bottleneck.

### 7.3 Methodological Limitations

1. **Token Injection Strategies**: While we test 10 diverse injection strategies, we cannot exhaustively cover all possible contextual patterns. However, our LLM-guided analysis partially addresses this limitation through systematic exploration of the pattern space.

2. **LLM Interpretation Validity**: Our LLM-guided analysis relies on Gemini 3 Pro's ability to generate valid counterexamples. While we empirically validate all counterexamples, the LLM's own biases may influence which patterns are explored. Human expert validation of a subset of interpretations would strengthen confidence.

3. **Causal vs. Correlational Claims**: Our token injection experiments demonstrate that features correlate with token patterns, but establishing that they *only* respond to tokens (and not to latent reasoning) requires stronger causal intervention. Ablation studies or causal mediation analysis could provide additional evidence.

### 7.4 Broader Implications

1. **Mechanistic Interpretability**: Our findings suggest that SAEs may face fundamental limitations in capturing high-level abstractions like reasoning, as these may not correspond to sparse, linear directions in activation space. Alternative decomposition methods (e.g., non-linear, distributed, or hierarchical representations) may be necessary.

2. **Evaluation Frameworks**: The field would benefit from standardized benchmarks and protocols for evaluating claimed "reasoning features," including adversarial testing and out-of-distribution generalization.

3. **Positive Results**: While we find no genuine reasoning features in our analysis, this does not preclude their existence in other layers, models, or SAE architectures. Our methodology provides a rigorous template for future positive claims, which should include evidence of resistance to token injection and successful validation of counterexample testing.

### 7.5 Future Work

1. **Hierarchical and Compositional Features**: Investigating whether combinations or interactions of multiple features capture reasoning more robustly than individual features.

2. **Mechanistic Circuit Analysis**: Tracing how putative reasoning features connect to attention heads and MLP neurons to understand their computational role.

3. **Intervention Studies**: Directly manipulating feature activations during inference on reasoning tasks to establish causal necessity and sufficiency.

4. **Cross-Model Generalization**: Testing whether features identified in one model transfer to other models or architectures.

---

## References

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50-60.

[Additional references to be added based on related work section]
