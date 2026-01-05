# Sparse Autoencoder Reasoning Feature Analysis

Code for analyzing reasoning features in sparse autoencoders.

## Installation

```bash
conda create -n probing python=3.11
conda activate probing
pip install -e .
```

For Gemma-3 models, install the modified TransformerLens from `TransformerLens/`:

```bash
pip uninstall transformer-lens
cd TransformerLens
pip install -e .
```

For DeepSeek-R1 distilled models, install the modified TransformerLens from the codebase of the "I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders" paper.

## Repository Structure

```
reasoning_features/
├── datasets/          # Dataset loaders
├── features/          # Feature analysis and detection
├── steering/          # Activation steering and evaluation
├── utils/             # Utility functions
├── scripts/           # Main experiment scripts
└── bash/              # Shell scripts for running experiments
```

## Running Experiments

All experiments are run through bash scripts in `reasoning_features/bash/`.

### Step 1: Feature Detection

Identify features with differential activation between datasets:

```bash
bash reasoning_features/bash/find_reasoning_features.sh
```

### Step 2: Token Injection

Test whether features are driven by specific tokens:

```bash
bash reasoning_features/bash/run_token_injection_experiment.sh
```

### Step 3: LLM-Guided Interpretation

Analyze context-dependent features using LLM-guided counterexample generation:

```bash
export OPENROUTER_API_KEY=your_key_here
bash reasoning_features/bash/analyze_feature_interpretation.sh
```

### Step 4: Steering Experiments

Evaluate performance with feature steering:

```bash
bash reasoning_features/bash/run_steering_experiment.sh
```

## Results

Experimental results are saved in `results/` with the following structure:

```
results/{metric}/{model}/{dataset}/layer{N}/
├── reasoning_features.json         # Detected features
├── token_analysis.json             # Token dependency analysis
├── injection_results.json          # Token injection results
├── feature_interpretations.json    # LLM interpretations
└── {benchmark}/                    # Steering results
    ├── experiment_summary.json
    └── feature_{index}/
        └── result_gamma_{value}.json
```
