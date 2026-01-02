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
- **Intersperse**: Distribute tokens uniformly throughout the text
- **Replace**: Substitute random words with injected tokens

**N-gram Injection** (inject 2 bigrams or 1 trigram):
- **Inject Bigram**: Insert top-activating bigrams identified from token analysis
- **Inject Trigram**: Insert top-activating trigrams

**Contextual Injection**:
- **Bigram Before**: Insert `[context, token]` pairs where `context` is a frequently preceding word
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

The LLM-guided analysis enables systematic exploration of diverse linguistic patterns that would be difficult to enumerate manually through predefined heuristics. The iterative empirical validation ensures that classifications are grounded in the model's actual behavior rather than the LLM's priors. To ensure reliability, we include all LLM-generated interpretations and counterexamples in the appendix and manually verify that none of the analyzed features constitute genuine reasoning features. This dual validation—automated exploration combined with human verification—provides confidence in our negative findings.

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

We test $\gamma \in \{0, 2\}$ and report one-shot accuracy with chain-of-thought prompting.

**Interpretation**: If the identified features captured genuine reasoning, we would expect positive $\gamma$ to improve accuracy. However, it is critical to note that **performance improvement does not imply genuine reasoning capture**. As demonstrated in recent work on test-time scaling (e.g., s1: Simple test-time scaling), superficial token-level interventions can yield substantial gains. For instance, merely suppressing end-of-thinking tokens and replacing them with "wait" improves AIME 2024 accuracy from 0% to nearly 60% on a 32B model—without using SAEs at all. Thus, our steering experiment serves as a sanity check but not as definitive evidence of reasoning feature existence.

## 6. Experimental Setup and Results

### 6.1 Models and Architectures

We conduct experiments on three state-of-the-art open-weight language models with established proficiency on reasoning tasks:

**Gemma-3-12B-Instruct** (12B parameters, 48 layers): We analyze layers 17, 22, and 27, corresponding to approximately 40%, 52%, and 64% network depth. SAEs are from the GemmaScope-2 release with 16,384 features per layer using residual stream activations and small L0 regularization.

**Gemma-3-4B-Instruct** (4B parameters, 34 layers): We analyze layers 17, 22, and 27 (61%, 79%, and 96% depth). SAE architecture matches the 12B variant with 16,384 features per layer.

**DeepSeek-R1-Distill-Llama-8B** (8B parameters, 32 layers): We analyze layer 19 (59% depth) using an SAE trained on reasoning datasets (LMSys-Chat-1M and OpenThoughts-114k) with 65,536 features (Galichin et al., 2025).

Our layer selection focuses on middle-to-late network regions for two complementary reasons. First, prior work suggests that reasoning-related activations emerge most strongly in these layers where the model has processed input patterns but has not yet committed to specific output tokens. Second, and more critically, our preliminary analysis across all layers reveals that middle layers exhibit the lowest token concentration ratios (i.e., the most distributed activation patterns), making them the most likely locations for genuine reasoning features if such features exist. Early layers show high concentration on input tokens, while late layers concentrate on output token predictions.

**Figure 1 (Placeholder)**: Token concentration and normalized entropy across all layers for Gemma-3-12B-Instruct on the s1K dataset. Middle layers (17-27) show the lowest concentration values, indicating less reliance on specific tokens.

For reference, we also report results on Gemma-2-9B (layer 21) and Gemma-2-2B (layer 13) in the appendix.

All experiments were conducted on a single NVIDIA A100 80GB GPU. Feature detection and token injection required approximately 1-2 hours per configuration, while LLM interpretation required 2-4 hours per configuration due to iterative API calls.

### 6.2 Datasets

We construct three datasets to distinguish reasoning from non-reasoning text. For reasoning, we use two complementary corpora: **s1K-1.1**, which contains 1,000 challenging mathematics problems with detailed reasoning traces generated by DeepSeek-R1 and Gemini, covering domains including algebra, geometry, and combinatorics; and **General Inquiry Thinking Chain-of-Thought**, which contains 6,000 question-answer pairs spanning diverse domains (science, logic, philosophy, everyday reasoning) with explicit chain-of-thought annotations. For non-reasoning text, we use **Pile (Uncopyrighted)**, a large-scale corpus of general web text including Wikipedia, books, academic papers, and web crawl data with all copyrighted content removed.

From each corpus, we sample 1000 texts truncated to 64 tokens. For each model and layer, we conduct experiments with both reasoning datasets separately, yielding two independent sets of features that may overlap. This dual-dataset approach provides robustness against dataset-specific biases.

### 6.3 Feature Detection Results

For each model, layer, and reasoning dataset combination, we identify the top 100 features ranked by Cohen's d that meet our statistical thresholds: minimum effect size $|d| \geq 0.3$, Bonferroni-corrected p-value $\leq 0.01$, and ROC-AUC $\geq 0.6$. We use 1000 samples per dataset (reasoning and non-reasoning), each truncated to 64 tokens, with maximum activation across tokens as our aggregation function.

**Table 1**: Mean Cohen's d values for the top 100 features per configuration.

| Model | Layer | s1K | General Inquiry CoT |
|-------|-------|-----|---------------------|
| Gemma-3-12B-IT | 17 | 0.764 | 0.822 |
| Gemma-3-12B-IT | 22 | 0.859 | 0.836 |
| Gemma-3-12B-IT | 27 | 0.848 | 0.820 |
| Gemma-3-4B-IT | 17 | 0.804 | 0.917 |
| Gemma-3-4B-IT | 22 | 0.818 | 1.016 |
| Gemma-3-4B-IT | 27 | 0.733 | 1.012 |
| DeepSeek-R1-Distill-Llama-8B | 19 | 0.675 | 0.781 |

All configurations successfully identified at least 76 features (in most cases 99-100) exceeding our statistical thresholds, demonstrating that SAE features can reliably distinguish between reasoning and non-reasoning text in a statistical sense. The mean Cohen's d values range from 0.675 to 1.016, indicating medium-to-large effect sizes. These features serve as candidate "reasoning features" for subsequent causal investigation. ROC-AUC values (ranging from 0.658 to 0.701) are reported in the appendix.

**Figure 2 (Placeholder)**: Distribution of Cohen's d values across all SAE features for Gemma-3-12B-IT layer 22 on s1K dataset. The plot shows a long-tail distribution with the top 100 features (shaded) clearly separated from the background.

### 6.4 Token Injection Experimental Results

For each configuration, we analyze the top 100 features identified in Section 6.3. We extract the top 10 tokens, top 20 bigrams, and top 10 trigrams for each feature based on mean activation, then conduct injection experiments using 500 non-reasoning samples (baseline) and 500 reasoning samples (target). We test ten injection strategies: prepend, append, intersperse, replace, inject_bigram, inject_trigram, bigram_before, bigram_after, trigram, and comma_list. For simple token strategies, we inject 3 tokens; for bigram strategies, 2 bigrams; for trigram strategies, 1 trigram.

We classify features based on their best-performing injection strategy (highest Cohen's d): token-driven ($d \geq 0.8$, $p < 0.01$), partially token-driven ($0.5 \leq d < 0.8$, $p < 0.01$), weakly token-driven ($0.2 \leq d < 0.5$, $p < 0.05$), or context-dependent ($d < 0.2$ or $p \geq 0.05$).

**Table 2**: Token injection classification results showing number (percentage) of features in each category.

| Model | Layer | s1K TD/PTD/WTD/CD | Gen. Inq. CoT TD/PTD/WTD/CD |
|-------|-------|-------------------|------------------------------|
| Gemma-3-12B-IT | 17 | 65 (65%) / 9 (9%) / 15 (15%) / 11 (11%) | 63 (63%) / 13 (13%) / 18 (18%) / 6 (6%) |
| Gemma-3-12B-IT | 22 | 36 (36%) / 20 (20%) / 24 (24%) / 19 (19%) | 46 (46%) / 20 (20%) / 26 (26%) / 8 (8%) |
| Gemma-3-12B-IT | 27 | 44 (58%) / 8 (11%) / 12 (16%) / 12 (16%) | 60 (60%) / 8 (8%) / 15 (15%) / 17 (17%) |
| Gemma-3-4B-IT | 17 | 46 (46%) / 23 (23%) / 18 (18%) / 13 (13%) | 39 (39%) / 26 (26%) / 18 (18%) / 17 (17%) |
| Gemma-3-4B-IT | 22 | 59 (59%) / 16 (16%) / 19 (19%) / 6 (6%) | 55 (55%) / 14 (14%) / 21 (21%) / 10 (10%) |
| Gemma-3-4B-IT | 27 | 37 (37%) / 19 (19%) / 22 (22%) / 22 (22%) | 52 (52%) / 13 (13%) / 18 (18%) / 17 (17%) |
| DeepSeek-R1 | 19 | 46 (46%) / 12 (12%) / 19 (19%) / 23 (23%) | 25 (25%) / 14 (14%) / 20 (20%) / 41 (41%) |

Across all 14 configurations, we observe that the vast majority of putative "reasoning features" exhibit substantial activation increases when their top tokens are injected into non-reasoning text. Combining all three token-driven categories (large, medium, and small effects), 77-95% of features show statistically significant activation from token injection alone, with only 6-41% classified as context-dependent. This pattern holds consistently across different models, layers, and datasets, though we note that DeepSeek-R1 on General Inquiry CoT shows the highest proportion of context-dependent features (41%), while Gemma-3-12B layer 17 and Gemma-3-4B layer 22 on General Inquiry CoT show the lowest (6%).

The average Cohen's d values for token injection range from 0.521 to 1.790 across configurations, indicating that injecting merely 3 tokens into 64-token sequences (4.7% of tokens) produces large activation increases. These findings provide strong evidence that most statistically-identified "reasoning features" are, in fact, shallow token detectors that respond to lexical patterns rather than reasoning processes.

**Figure 3 (Placeholder)**: Stacked bar chart showing the distribution of feature classifications across all configurations. Each bar represents one configuration, with segments colored by category (token-driven in red, partially token-driven in orange, weakly token-driven in yellow, context-dependent in blue).

### 6.5 LLM-Guided Interpretation Results

For each configuration, we randomly sample up to 20 context-dependent features from the token injection experiment and subject them to LLM-guided analysis using Google Gemini 3 Pro. The iterative protocol (described in Section 4.2) runs for up to 10 iterations per feature, generating false positive and false negative counterexamples until reaching 3 valid examples of each type or exhausting the iteration budget. We set the activation threshold at $\tau = 0.5 \times \max(\text{reasoning activations})$ and use temperature 0.8 for counterexample generation (to encourage diversity) and 0.3 for final interpretation (to ensure consistency).

**Table 3**: LLM interpretation results for context-dependent features. All entries show Genuine/Confound counts.

| Model | Layer | s1K (Analyzed / Genuine / High Conf) | Gen. Inq. CoT (Analyzed / Genuine / High Conf) |
|-------|-------|--------------------------------------|------------------------------------------------|
| Gemma-3-12B-IT | 17 | 11 / 0 / 9 | 6 / 0 / 6 |
| Gemma-3-12B-IT | 22 | 18 / 0 / 18 | 7 / 0 / 5 |
| Gemma-3-12B-IT | 27 | 12 / 0 / 12 | 16 / 0 / 14 |
| Gemma-3-4B-IT | 17 | 13 / 0 / 10 | 17 / 0 / 16 |
| Gemma-3-4B-IT | 22 | 6 / 0 / 6 | 10 / 0 / 9 |
| Gemma-3-4B-IT | 27 | 20 / 0 / 19 | 17 / 0 / 17 |
| DeepSeek-R1 | 19 | 20 / 0 / 14 | 19 / 0 / 18 |

The central finding is unequivocal: **across all 153 context-dependent features analyzed, zero were classified as genuine reasoning features**. Every feature was identified as a confound—a linguistic pattern that correlates with reasoning text but does not capture the reasoning process itself. The LLM expressed high confidence for 136 out of 153 features (89%), indicating that the confounds are typically clear and robust based on the success of counterexample generation.

The analysis revealed common categories of confounds. Many features detect conversational introductory sequences (e.g., "Okay, let's see," "So," "Now let's tackle") paired with technical vocabulary, activating on both problem-solving text and descriptive content with similar phrasing. Others respond to formal academic discourse markers ("Furthermore," "Consequently," "Given that") that appear in both reasoning and expository writing, or to meta-cognitive planning phrases ("First, we need to," "The next step is") that signal procedural organization without necessarily indicating reasoning. Some features activate on complex syntactic structures or abstract/technical vocabulary that correlates with prose sophistication and domain expertise rather than the reasoning process itself.

All LLM-generated interpretations and counterexamples are provided in the appendix, where readers can verify that none of the analyzed features constitute genuine reasoning detectors. This comprehensive analysis, combining automated exploration with human verification, strongly supports our central hypothesis: even features that exhibit context-dependence to simple token injection are not genuine reasoning features but rather sophisticated linguistic confounds.

### 6.6 Steering Experimental Results

We conduct steering experiments on Gemma-3-12B-IT layer 22, using the top 3 features ranked by Cohen's d from the s1K dataset. We apply the steering formula $\mathbf{x}' = \mathbf{x} + \gamma \cdot f_i^{\max} \cdot \mathbf{W}_{\text{dec},i}$ with $\gamma \in \{0, 2\}$ to evaluate baseline performance versus strong amplification. We test on AIME 2024 (30 mathematics competition problems) and GPQA Diamond (198 graduate-level science questions) using one-shot chain-of-thought prompting with maximum 32,768 generation tokens, temperature 0.6, and top-p 0.95.

**Table 4 (Placeholder)**: Steering results showing one-shot accuracy for each feature.

| Feature | Baseline ($\gamma = 0$) AIME | Steered ($\gamma = 2$) AIME | Baseline GPQA | Steered GPQA |
|---------|------------------------------|------------------------------|---------------|--------------|
| [TBD] | [TBD]% | [TBD]% | [TBD]% | [TBD]% |
| [TBD] | [TBD]% | [TBD]% | [TBD]% | [TBD]% |
| [TBD] | [TBD]% | [TBD]% | [TBD]% | [TBD]% |

As discussed in Section 5, this steering experiment serves as a supplementary sanity check rather than a primary line of evidence. Performance improvement (if observed) would not constitute definitive evidence of genuine reasoning capture, as superficial token-level interventions have been shown to yield substantial gains on these benchmarks without engaging reasoning mechanisms. Our results are expected to show [TBD: minimal change or slight degradation], consistent with features capturing spurious correlates rather than causal reasoning mechanisms.

## 7. Limitations

Our investigation is subject to several important limitations that should inform interpretation of our findings.

**Dataset and Sampling Constraints**: Our analysis examines two reasoning datasets (mathematical problem-solving via s1K-1.1 and general inquiry via General Inquiry CoT), which, while covering important domains, may not exhaust all forms of reasoning behavior. Additionally, we truncate all sequences to 64 tokens for computational efficiency, potentially missing long-range dependencies in extended reasoning chains. The Pile, used as our non-reasoning baseline, contains some expository and analytical text that may exhibit reasoning-adjacent linguistic patterns, though this conservative choice strengthens rather than weakens our negative findings.

**Architectural Scope**: We analyze SAEs trained with L0 regularization on residual stream activations from specific layers (layers 17, 22, 27 for Gemma models; layer 19 for DeepSeek). Different SAE variants (e.g., attention-based or MLP-based decompositions, alternative sparsity penalties) or different network depths may yield different feature types. However, our selection of middle layers—where token concentration is lowest—maximizes the likelihood of finding genuine reasoning features if they exist.

**Methodological Coverage**: While we test ten diverse token injection strategies, we cannot exhaustively enumerate all possible linguistic patterns. Our LLM-guided analysis partially addresses this limitation through systematic exploration of the pattern space, though it relies on Gemini 3 Pro's ability to generate valid counterexamples. The LLM's biases may influence which patterns are explored, though we mitigate this through empirical validation of all counterexamples and human verification of all interpretations (provided in the appendix).

**Broader Implications**: Our findings suggest potential fundamental limitations of SAE-based decomposition for capturing high-level abstractions like reasoning, which may not correspond to sparse, linear directions in activation space. While we find zero genuine reasoning features across 153 context-dependent features and 14 experimental configurations, this does not preclude their existence in unexplored architectures, layers, or models. Our methodology provides a rigorous template for evaluating future positive claims, which should include demonstrated resistance to token injection and successful counterexample testing.

---

## References

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50-60.

[Additional references to be added based on related work section]
