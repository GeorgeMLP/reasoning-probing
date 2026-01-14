# Do Sparse Autoencoders Identify Reasoning Features in Language Models?

This repository contains the code and experimental framework for our paper investigating whether Sparse Autoencoders (SAEs) capture genuine reasoning features in language models, or merely learn spurious correlations with reasoning-associated tokens.

## Overview

We investigate SAE features that show differential activation on reasoning vs. non-reasoning text through a multi-stage experimental pipeline:

1. **Feature Detection**: Identify features with statistical correlation to reasoning text using Cohen's d, ROC-AUC, and frequency ratio metrics
2. **Token Injection**: Test whether features are driven by specific tokens through causal intervention
3. **LLM-Guided Interpretation**: Use LLM-based hypothesis testing to identify linguistic confounds
4. **Steering Experiments**: Evaluate whether amplifying features improves reasoning performance

**Main Finding**: Across 20 configurations on multiple models, layers and datasets, we find zero genuine reasoning features. All features are identified as confounds that respond to shallow linguistic patterns (conversational markers, formal discourse, syntactic complexity) rather than reasoning processes.

## Repository Structure

```
reasoning_features/
├── datasets/          # Dataset loaders (Pile, s1K, General Inquiry CoT)
├── features/          # Feature analysis and detection
│   ├── collector.py   # SAE activation collection
│   ├── detector.py    # Statistical feature detection
│   └── tokens.py      # Token dependency analysis
├── steering/          # Activation steering and evaluation
│   ├── steerer.py     # Feature steering implementation
│   └── evaluator.py   # Benchmark evaluation
├── utils/             # Utility functions (LLM judge, etc.)
├── scripts/           # Main experiment scripts
│   ├── find_reasoning_features.py
│   ├── run_token_injection_experiment.py
│   ├── analyze_feature_interpretation.py
│   ├── run_steering_experiment.py
│   └── plot_results.py
├── bash/              # Shell scripts for running experiments
└── paper_figs/        # Figure generation for paper
```

## Installation

### For Gemma-3 Experiments

This repository uses a modified version of TransformerLens with Gemma-3 support from [huseyincavusbi/TransformerLens](https://github.com/huseyincavusbi/TransformerLens), included in the `TransformerLens/` directory.

```bash
# Create environment
conda create -n probing python=3.11
conda activate probing

# Install main package
pip install -e .

# Install modified TransformerLens
pip uninstall transformer-lens
cd TransformerLens
pip install -e .
cd ..
```

### For DeepSeek-R1 Experiments

DeepSeek-R1 distilled models require a different TransformerLens fork from [AIRI-Institute/SAE-Reasoning](https://github.com/AIRI-Institute/SAE-Reasoning):

```bash
# Clone and install the AIRI fork instead
git clone https://github.com/AIRI-Institute/SAE-Reasoning.git
cd SAE-Reasoning/TransformerLens
pip install -e .
cd ../..
```

## Running Experiments

All experiments are orchestrated through bash scripts in `reasoning_features/bash/`. Edit the scripts to configure model names, layers, and output directories.

### 1. Feature Detection

Identify features with differential activation between reasoning and non-reasoning text:

```bash
bash reasoning_features/bash/find_reasoning_features.sh
```

**Output**: `results/{metric}/{model}/{dataset}/layer{N}/`
- `reasoning_features.json`: Top features ranked by metric
- `token_analysis.json`: Top tokens, bigrams, trigrams per feature

### 2. Token Injection Experiments

Test whether features are driven by specific tokens:

```bash
bash reasoning_features/bash/run_token_injection_experiment.sh
```

**Output**: `injection_results.json` with classification (token-driven, partially token-driven, weakly token-driven, context-dependent) based on Cohen's d effect sizes.

### 3. LLM-Guided Feature Interpretation

Analyze context-dependent features using Gemini 3 Pro via OpenRouter:

```bash
export OPENROUTER_API_KEY=your_key_here
bash reasoning_features/bash/analyze_feature_interpretation.sh
```

**Output**: `feature_interpretations.json` with refined interpretations, false positive/negative examples, and genuine reasoning classification.

### 4. Steering Experiments

Evaluate benchmark performance with feature amplification:

```bash
bash reasoning_features/bash/run_steering_experiment.sh
```

**Output**: Per-feature results for AIME 2024 and GPQA Diamond benchmarks.

## Results Directory Structure

```
results/{metric}/{model}/{dataset}/layer{N}/
├── reasoning_features.json         # Detected features with statistics
├── token_analysis.json             # Token/bigram/trigram analysis
├── injection_results.json          # Token injection classifications
├── feature_interpretations.json    # LLM-guided interpretations
└── {benchmark}/                    # Steering experiment results
    ├── experiment_summary.json
    └── feature_{index}/
        ├── feature_summary.json
        └── result_gamma_{value}.json
```

## Models and Datasets

### Models Analyzed
- Gemma-3-12B-Instruct (layers 17, 22, 27)
- Gemma-3-4B-Instruct (layers 17, 22, 27)
- DeepSeek-R1-Distill-Llama-8B (layer 19)
- Gemma-2-9B (layer 21, appendix)
- Gemma-2-2B (layer 13, appendix)

### Datasets
- **Reasoning**: s1K-1.1, General Inquiry Thinking Chain-of-Thought
- **Non-Reasoning**: Pile (Uncopyrighted)
- **Benchmarks**: AIME 2024, GPQA Diamond

## Hardware Requirements

All experiments were conducted on a single NVIDIA A100 80GB GPU.

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{ma2026sparse,
    title={{Do Sparse Autoencoders Identify Reasoning Features in Language Models?}},
    author={Ma, George and Liang, Zhongyuan and Chen, Irene Y. and Sojoudi, Somayeh},
    journal={arXiv preprint arXiv:2601.05679},
    year={2026}
}
```

## Acknowledgments

This work uses modified versions of TransformerLens:
- Gemma-3 support from [huseyincavusbi/TransformerLens](https://github.com/huseyincavusbi/TransformerLens)
- DeepSeek-R1 support from [AIRI-Institute/SAE-Reasoning](https://github.com/AIRI-Institute/SAE-Reasoning)

We thank the authors of these forks for making their code available.
