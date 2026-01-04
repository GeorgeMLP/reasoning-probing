# Appendix: Supplementary Experimental Results

This appendix provides additional experimental results and analyses that complement the main text. Section A reports results on Gemma-2 models (Gemma-2-9B and Gemma-2-2B) to assess architectural generalization. Section B examines alternative feature ranking metrics (ROC-AUC and frequency ratio) to validate robustness of our methodology. Section C presents detailed statistics on token dependency, injection strategies, and dataset overlap from the main experiments. Section D documents all hyperparameters used across the experimental pipeline. Additionally, we provide complete LLM-generated interpretations for all 192 analyzed context-dependent features in a separate LaTeX document (`docs/feature_interpretations.tex`), which includes refined interpretations, classifications, and false positive/negative counterexamples for each feature.

## A. Experimental Results on Additional Models

To validate the generality of our findings across model architectures and sizes, we conducted the same experimental protocol on two additional models from the Gemma-2 family: Gemma-2-9B and Gemma-2-2B. These models use the previous-generation Gemma architecture and provide a comparison point to our main results on Gemma-3 models.

### A.1 Gemma-2-9B Results

For Gemma-2-9B (9B parameters, 42 layers), we analyze layer 21 (50% depth) on both reasoning datasets. This layer was selected based on preliminary token concentration analysis showing minimal reliance on specific tokens. We use SAEs from the GemmaScope release with 16,384 features per layer.

**Table A1**: Feature detection and token injection results for Gemma-2-9B layer 21.

| Dataset | Mean Cohen's d | Token-Driven | Partially TD | Weakly TD | Context-Dep | Avg Injection d |
|---------|----------------|--------------|--------------|-----------|-------------|-----------------|
| s1K | 0.709 | 6 (8%) | 7 (9%) | 23 (29%) | 42 (54%) | 0.285 |
| General Inquiry CoT | 0.764 | 6 (6%) | 4 (4%) | 33 (33%) | 57 (57%) | 0.294 |

Gemma-2-9B exhibits markedly different behavior compared to Gemma-3 models. While features still show medium effect sizes in statistical detection (mean Cohen's d of 0.709-0.764), token injection yields substantially lower effect sizes (average d of 0.285-0.294) and a dramatically higher proportion of context-dependent features (54-57% vs. 6-41% in Gemma-3 models). This suggests that Gemma-2 features may be less reliant on shallow token patterns, though our LLM analysis confirms this does not translate to genuine reasoning capture.

We analyzed 20 randomly sampled context-dependent features from each dataset configuration using the LLM-guided protocol. Across all 40 analyzed features, zero were classified as genuine reasoning features. The LLM identified all 40 as confounds with high confidence for 34 features (85%). Common patterns include formal academic writing style, meta-cognitive discourse markers, and technical vocabulary that appears in both reasoning and non-reasoning contexts.

### A.2 Gemma-2-2B Results

For Gemma-2-2B (2B parameters, 26 layers), we analyze layer 13 (50% depth) on both reasoning datasets. This smaller model provides insight into whether reasoning feature capture scales with model capacity.

**Table A2**: Feature detection and token injection results for Gemma-2-2B layer 13.

| Dataset | Mean Cohen's d | Token-Driven | Partially TD | Weakly TD | Context-Dep | Avg Injection d |
|---------|----------------|--------------|--------------|-----------|-------------|-----------------|
| s1K | 0.636 | 2 (2%) | 18 (18%) | 36 (36%) | 44 (44%) | 0.281 |
| General Inquiry CoT | 0.602 | 1 (1%) | 7 (7%) | 35 (35%) | 57 (57%) | 0.207 |

Gemma-2-2B shows even weaker token injection effects than Gemma-2-9B, with only 1-2% of features classified as strongly token-driven and 44-57% as context-dependent. The average injection effect sizes (0.207-0.281) are the lowest across all models tested, suggesting that smaller models may develop less token-reliant feature representations. However, LLM analysis of 39 context-dependent features (19-20 per configuration) revealed zero genuine reasoning features, with 35 (90%) classified as confounds with high confidence. The confounds identified include procedural discourse markers, formal sentence structure, and domain-specific vocabulary patterns.

## B. Experimental Results with Alternative Ranking Metrics

In the main text, we use Cohen's d as our primary metric for ranking and selecting candidate reasoning features. To assess the robustness of our findings to this methodological choice, we conducted parallel experiments on Gemma-3-4B-Instruct layer 22 using two alternative ranking metrics: ROC-AUC and frequency ratio. All other experimental parameters (datasets, sample sizes, thresholds, injection strategies, LLM analysis) remain identical to the main experiments.

### B.1 ROC-AUC Based Selection

The ROC-AUC (Receiver Operating Characteristic - Area Under Curve) metric quantifies a feature's ability to discriminate between reasoning and non-reasoning samples across all possible classification thresholds:

$$
\text{AUC}_i = \mathbb{P}(a_{i,j}^{\text{R}} > a_{i,k}^{\text{NR}}) = \frac{1}{n_{\text{R}} \cdot n_{\text{NR}}} \sum_{j=1}^{n_{\text{R}}} \sum_{k=1}^{n_{\text{NR}}} \mathbb{1}[a_{i,j}^{\text{R}} > a_{i,k}^{\text{NR}}].
$$

This metric provides a distribution-free measure of discriminative power that is robust to class imbalance and does not assume any particular activation distribution shape. An AUC of 0.5 indicates chance-level performance, while 1.0 indicates perfect separation. We rank all features by AUC and select the top 100 meeting our thresholds (AUC $\geq$ 0.6, p-value $\leq$ 0.01, Cohen's d $\geq$ 0.3).

**Table B1**: Results for top 100 features ranked by ROC-AUC (Gemma-3-4B-Instruct layer 22, s1K).

| Metric | Value |
|--------|-------|
| Mean Cohen's d | 0.830 |
| Mean ROC-AUC | 0.667 |
| Token-driven | 59 (59%) |
| Partially token-driven | 18 (18%) |
| Weakly token-driven | 18 (18%) |
| Context-dependent | 5 (5%) |
| Features analyzed (LLM) | 5 |
| Genuine reasoning features | 0 |

Features selected by ROC-AUC ranking show similar token injection behavior to Cohen's d selected features, with 95% exhibiting significant token-driven activation (combining all three categories) and only 5% context-dependent. The average Cohen's d for these features (0.830) is comparable to the Cohen's d-selected set (0.818), indicating strong correlation between the two metrics. All 5 context-dependent features were classified as confounds by LLM analysis with high confidence.

### B.2 Frequency Ratio Based Selection

The frequency ratio metric measures the relative prevalence of feature activation between reasoning and non-reasoning samples:

$$
\text{FreqRatio}_i = \frac{\text{freq}_i^{\text{R}} + \epsilon}{\text{freq}_i^{\text{NR}} + \epsilon},
$$

where $\text{freq}_i^{\text{R}}$ and $\text{freq}_i^{\text{NR}}$ denote the proportion of samples with activation exceeding a threshold (we use $\max(0.5 \sigma_{\text{baseline}}, 0.01)$ as the threshold), and $\epsilon = 0.01$ is a smoothing constant to handle zero denominators. This metric captures a different aspect of feature selectivity: rather than measuring activation magnitude differences (Cohen's d) or rank-order discrimination (ROC-AUC), it quantifies how much more frequently a feature activates on reasoning text.

**Table B2**: Results for top 100 features ranked by frequency ratio (Gemma-3-4B-Instruct layer 22, s1K).

| Metric | Value |
|--------|-------|
| Mean Cohen's d | 0.830 |
| Mean ROC-AUC | 0.667 |
| Token-driven | 65 (65%) |
| Partially token-driven | 14 (14%) |
| Weakly token-driven | 17 (17%) |
| Context-dependent | 4 (4%) |
| Features analyzed (LLM) | 4 |
| Genuine reasoning features | 0 |

Frequency ratio selection produces nearly identical results to ROC-AUC selection, with 96% of features showing token-driven behavior and only 4% context-dependent. Interestingly, both alternative metrics produce slightly higher proportions of strongly token-driven features (59-65%) compared to Cohen's d selection (59% for the same layer and dataset), suggesting they may prioritize features with more concentrated activation patterns. All 4 context-dependent features were classified as confounds by LLM analysis.

**Figure A1**: Bar chart comparing Jaccard similarities between top-100 feature sets selected by different ranking metrics. Path: `figs/metric_comparison_rankings.pdf`

### B.3 Metric Robustness Analysis

The consistency across three distinct ranking metrics provides strong evidence that our findings are not artifacts of metric choice. The pairwise Jaccard similarities between top-100 feature sets are high: Cohen's d vs. ROC-AUC (0.818), Cohen's d vs. frequency ratio (0.739), and ROC-AUC vs. frequency ratio (0.754). Furthermore, 80 features appear in all three top-100 lists, demonstrating substantial consensus despite different ranking criteria.

All three metrics identify features that show substantial token injection effects: 95% for ROC-AUC selection and 96% for frequency ratio selection are classified as token-driven (combining all three token-driven categories), compared to 94% for Cohen's d selection. The mean Cohen's d values for top-100 features are nearly identical across metrics (0.818 for Cohen's d, 0.830 for both ROC-AUC and frequency ratio), indicating that the different metrics converge on similar feature subsets with similar statistical properties.

More critically, LLM analysis of the 9 total context-dependent features (5 from ROC-AUC, 4 from frequency ratio, with some overlap to Cohen's d selection) yielded zero genuine reasoning features across all metrics. This demonstrates that our central finding is robust to the choice of statistical metric used for feature selection.

## C. Additional Experimental Statistics

This section reports supplementary statistics and analyses from our main experiments that provide additional context for interpreting the results.

### C.1 Token Dependency Statistics Across Configurations

Beyond the binary classification of features via token injection, we analyze the underlying distributions of token concentration and activation entropy across all configurations. Token concentration measures the fraction of total activation attributable to the top-30 tokens, while normalized entropy quantifies the dispersion of activation across the token vocabulary.

**Table C1**: Token dependency statistics for reasoning features across all main configurations.

| Model | Layer | Dataset | Mean Concentration | Median Concentration | High Dependency (>0.5) |
|-------|-------|---------|-------------------|---------------------|------------------------|
| Gemma-3-12B-Instruct | 17 | s1K | 0.686 | 0.904 | 69 (69%) |
| Gemma-3-12B-Instruct | 17 | Gen. Inq. | 0.532 | 0.550 | 51 (51%) |
| Gemma-3-12B-Instruct | 22 | s1K | 0.668 | 0.769 | 65 (66%) |
| Gemma-3-12B-Instruct | 22 | Gen. Inq. | 0.607 | 0.633 | 62 (62%) |
| Gemma-3-12B-Instruct | 27 | s1K | 0.807 | 0.989 | 63 (83%) |
| Gemma-3-12B-Instruct | 27 | Gen. Inq. | 0.684 | 0.875 | 69 (69%) |
| Gemma-3-4B-Instruct | 17 | s1K | 0.675 | 0.873 | 69 (69%) |
| Gemma-3-4B-Instruct | 17 | Gen. Inq. | 0.718 | 0.936 | 71 (71%) |
| Gemma-3-4B-Instruct | 22 | s1K | 0.823 | 0.988 | 81 (81%) |
| Gemma-3-4B-Instruct | 22 | Gen. Inq. | 0.737 | 0.944 | 73 (73%) |
| Gemma-3-4B-Instruct | 27 | s1K | 0.786 | 0.968 | 82 (82%) |
| Gemma-3-4B-Instruct | 27 | Gen. Inq. | 0.742 | 0.952 | 77 (77%) |
| DS-R1-Distill-Llama-8B | 19 | s1K | 0.569 | 0.528 | 52 (52%) |
| DS-R1-Distill-Llama-8B | 19 | Gen. Inq. | 0.687 | 0.826 | 69 (69%) |

Across configurations, 51-83% of reasoning features show high token dependency (concentration >0.5), with mean concentration values ranging from 0.532 to 0.823. The high median values (0.528-0.989) indicate that most features have strongly skewed activation distributions favoring a small subset of tokens. DeepSeek-R1 on s1K shows the lowest token concentration (mean: 0.569, median: 0.528), while Gemma-3-4B layer 27 on s1K shows the highest (mean: 0.823, median: 0.989), suggesting layer depth and architecture influence token dependency.

The discrepancy between median and mean concentration (median typically higher) indicates a right-skewed distribution: most features have very high concentration, with a minority showing more distributed patterns. This aligns with our token injection findings that the majority of features respond to specific lexical cues.

**Figure A2**: Token concentration distributions across layers for Gemma-3-12B-Instruct and Gemma-3-4B-Instruct on s1K dataset, shown as violin plots (showing median with a line). Path: `figs/token_concentration_distributions.pdf`

### C.2 Injection Strategy Performance Comparison

We tested ten distinct injection strategies to account for varying degrees of context sensitivity. While we report best-strategy performance in the main text, here we analyze the relative effectiveness of different strategies across all features.

For Gemma-3-12B-Instruct layer 22 on s1K (representative configuration), we identify which strategy achieves the highest Cohen's d for each feature. The prepend strategy dominates, performing best for 69.7% of features (69 out of 99), followed by inject\_bigram (9.1%), replace (6.1%), and inject\_trigram (5.1%). Contextual strategies (bigram\_before, bigram\_after, trigram) collectively account for only 3% of best performances, indicating that preserving natural token co-occurrence patterns does not substantially mitigate token injection effects.

This distribution demonstrates that the simplest intervention (prepending tokens to the beginning of text) is typically most effective, suggesting that features respond primarily to token presence rather than contextual appropriateness or positional factors. The success of n-gram injection strategies (14.2% combined) indicates that some features benefit from multi-token patterns, though this still represents a minority.

**Figure A3**: Box plots comparing Cohen's d distributions across injection strategies for Gemma-3-12B-Instruct layer 22 s1K. The box extends from the first quartile (Q1) to the third quartile (Q3) of the data, with a line at the median. The whiskers extend from the box to the farthest data point lying within 1.5x the inter-quartile range (IQR) from the box. Flier points are those past the end of the whiskers. Path: `figs/strategy_comparison.pdf`

### C.3 Activation Magnitude Analysis

We analyze the absolute activation magnitudes achieved under different conditions to contextualize our effect size findings. Note that activation magnitudes vary significantly across features and are presented here as feature-level averages (i.e., we first compute the mean activation for each feature across its 500 samples, then average these means across the 99 features).

**Table C2**: Mean activation magnitudes across conditions for Gemma-3-12B-Instruct layer 22 s1K.

| Condition | Mean Activation | Std Dev | Median | 90th Percentile |
|-----------|-----------------|---------|--------|-----------------|
| Non-reasoning (baseline) | 184.5 | 193.3 | 134.0 | 469.4 |
| Non-reasoning (injected, best strategy) | 382.6 | 380.0 | 294.1 | 684.2 |
| Reasoning text | 379.3 | 233.6 | 311.4 | 753.1 |

Token injection increases mean activation by a factor of 2.07 on average (from 184.5 to 382.6), achieving 100.9% of reasoning-level activation. The near-complete overlap between injected and reasoning activation magnitudes demonstrates that token presence alone is sufficient to reproduce the activation patterns observed on genuine reasoning text. The 90th percentile values (684.2 for injected vs. 753.1 for reasoning) indicate that the distributions overlap substantially in their upper tails as well.

### C.4 LLM Interpretation Convergence

We analyze the number of iterations required for the LLM-guided protocol to reach stopping criteria (3 false positives and 3 false negatives) across all 153 context-dependent features from the main experiments.

**Table C3**: LLM iteration statistics across all analyzed features.

| Statistic | Value |
|-----------|-------|
| Mean iterations to convergence | 2.4 |
| Median iterations to convergence | 2.0 |
| Features converged in 1 iteration | 47 (31%) |
| Features converged in 2 iterations | 68 (44%) |
| Features converged in 3+ iterations | 38 (25%) |
| Features reaching max iterations (10) | 3 (2%) |

The rapid convergence (75% of features converge within 2 iterations) indicates that the confounds are typically straightforward to identify and validate. Features requiring more iterations often involve complex multi-modal activation patterns (e.g., responding to both formal academic style and technical vocabulary) that require more extensive counterexample exploration.

**Figure A4**: Distribution of iterations to convergence for LLM interpretation. Path: `figs/llm_iterations_distribution.pdf`

### C.5 Feature Overlap Between Datasets

We investigate whether the same features are identified as "reasoning features" across both reasoning datasets (s1K-1.1 and General Inquiry CoT) for a given model and layer. High overlap would suggest dataset-invariant patterns, while low overlap might indicate dataset-specific spurious correlations.

**Table C4**: Jaccard similarity of top 100 feature sets between s1K and General Inquiry CoT.

| Model | Layer | Intersection Size | Jaccard Similarity |
|-------|-------|------------------|-------------------|
| Gemma-3-12B-Instruct | 17 | 41 | 0.258 |
| Gemma-3-12B-Instruct | 22 | 49 | 0.327 |
| Gemma-3-12B-Instruct | 27 | 22 | 0.143 |
| Gemma-3-4B-Instruct | 17 | 21 | 0.117 |
| Gemma-3-4B-Instruct | 22 | 14 | 0.075 |
| Gemma-3-4B-Instruct | 27 | 11 | 0.058 |

The Jaccard similarities range from 0.058 to 0.327, indicating low to moderate overlap. Only 6-33% of top features are shared between datasets, with 11-49 features appearing in both top-100 lists. Gemma-3-12B shows higher overlap (14.3-32.7%) than Gemma-3-4B (5.8-11.7%), suggesting that larger models may develop more dataset-invariant feature representations. The overlap decreases with layer depth for both models, with layer 27 showing particularly low overlap (14.3% for 12B, 5.8% for 4B).

This low overlap suggests that different reasoning corpora activate largely distinct feature subsets, which may reflect genuine differences in reasoning style (mathematical vs. general inquiry) or, alternatively, dataset-specific spurious correlations (e.g., LaTeX notation in s1K, conversational markers in General Inquiry). To distinguish these interpretations, we examined shared features using LLM analysis. All analyzed shared context-dependent features were classified as confounds, with patterns including mathematical notation detectors, formal academic phrases, and meta-cognitive markers that appear in both mathematical and general reasoning contexts but also in non-reasoning text with similar stylistic characteristics.

**Figure A5**: Stacked bar chart and Jaccard similarity showing feature set overlap between s1K and General Inquiry CoT for each model and layer. Path: `figs/dataset_overlap.pdf`

## D. Hyperparameter Settings

This section provides a comprehensive specification of all hyperparameters used across our experimental pipeline.

### D.1 Feature Detection Stage

For identifying candidate reasoning features via statistical testing, we used 1,000 samples from each corpus (reasoning and non-reasoning), with each sample truncated to a maximum of 64 tokens. Features were ranked by Cohen's d (for the main experiments) or by alternative metrics (for appendix experiments). We imposed three simultaneous thresholds: minimum Cohen's d of 0.3, maximum Bonferroni-corrected p-value of 0.01 (Mann-Whitney U test), and minimum ROC-AUC of 0.6. From features meeting these criteria, we selected the top 100 ranked by the primary metric.

For token dependency analysis within this stage, we extracted the top 30 tokens per feature based on mean activation, requiring a minimum of 5 occurrences for inclusion. We similarly extracted the top 20 bigrams (minimum 3 occurrences) and top 10 trigrams (minimum 2 occurrences) for each feature. Processing used a batch size of 16 for activation collection on all models.

### D.2 Token Injection Stage

Token injection experiments tested the top 100 features from the detection stage using 500 samples per condition (baseline and injected non-reasoning text). We employed eight injection strategies: prepend, intersperse, replace, inject\_bigram, inject\_trigram, bigram\_before, trigram, and comma\_list. For simple token strategies (prepend, intersperse, replace, comma\_list), we injected 3 tokens selected from the top 10 tokens for that feature. For bigram-based strategies (inject\_bigram, bigram\_before), we injected 2 bigrams selected from the top 20 bigrams. For trigram-based strategies (inject\_trigram, trigram), we injected 1 trigram selected from the top 10 trigrams.

Classification thresholds followed Cohen's conventions: token-driven required d greater than or equal to 0.8 with p less than 0.01, partially token-driven required d between 0.5 and 0.8 with p less than 0.01, weakly token-driven required d between 0.2 and 0.5 with p less than 0.05, and context-dependent corresponded to d less than 0.2 or p greater than or equal to 0.05. For each feature, we selected the best-performing strategy (highest Cohen's d) for final classification. All samples were truncated to 64 tokens with batch size 16 for processing.

### D.3 LLM-Guided Interpretation Stage

For context-dependent features identified in the injection stage, we randomly sampled up to 20 features per configuration for LLM-guided analysis using Google Gemini 3 Pro (accessed via OpenRouter API). The iterative protocol ran for a maximum of 10 iterations per feature, with early stopping when reaching 3 valid false positives and 3 valid false negatives.

The activation threshold for counterexample validation was set at 0.5 times the maximum activation observed on reasoning samples for that feature. False positives (non-reasoning text) were considered valid if their maximum activation exceeded this threshold, while false negatives (reasoning text) were valid if their maximum activation fell below 0.1 times the threshold. In each iteration, the LLM generated 5 candidate examples per category (10 total per iteration).

Temperature settings varied by task: hypothesis generation and final interpretation used temperature 0.3 to ensure consistency and reduce variability, while counterexample generation used temperature 0.8 to encourage diversity in proposed patterns.

### D.4 Steering Experiment Stage

Steering experiments were conducted exclusively on Gemma-3-12B-Instruct layer 22, selecting the top 3 features ranked by Cohen's d from the s1K dataset. We tested two gamma values: 0.0 (baseline, no steering) and 2.0 (positive amplification). The steering formula scaled the intervention by the maximum feature activation observed in the reasoning corpus for that feature.

For text generation, we used maximum 16,384 new tokens with sampling temperature 0.6 and nucleus sampling (top-p) threshold 0.95. All prompts used the model's chat template for proper formatting. Evaluation used one-shot chain-of-thought prompting with a single exemplar for each benchmark. For AIME 2024 (30 problems), we evaluated exact numerical answer matching. For GPQA Diamond (198 problems), we evaluated multiple-choice accuracy by extracting the letter corresponding to the selected answer.

### D.5 Model and SAE Configuration

All Gemma-3 models used the GemmaScope-2 SAE release with width 16,384 features per layer and small L0 regularization. DeepSeek-R1-Distill-Llama-8B used an SAE with width 65,536 trained specifically on reasoning datasets. All models were loaded in bfloat16 precision and run on a single NVIDIA A100 80GB GPU.

For Gemma-2 models (reported in Appendix A), we used similar configurations: Gemma-2-9B with GemmaScope release "gemma-scope-2b-pt-res-canonical" and Gemma-2-2B with the same release family, both using SAEs with 16,384 features per layer. Alternative metric experiments (Appendix B) used identical model and SAE configurations to the Cohen's d experiments, differing only in the feature ranking criterion.
