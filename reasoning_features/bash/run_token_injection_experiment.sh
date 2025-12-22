for layer in 0 4 8 12 16 20 24; do
    python reasoning_features/scripts/run_token_injection_experiment.py \
        --token-analysis results/initial-setting/gemma-2-9b/s1k/layer$layer/token_analysis.json \
        --reasoning-features results/initial-setting/gemma-2-9b/s1k/layer$layer/reasoning_features.json \
        --layer $layer \
        --top-k-features 10 \
        --top-k-tokens 10 \
        --n-inject 3 \
        --n-samples 2000 \
        --reasoning-dataset s1k \
        --save-dir results/initial-setting/gemma-2-9b/s1k/layer$layer
    python reasoning_features/scripts/run_token_injection_experiment.py \
        --token-analysis results/initial-setting/gemma-2-9b/general_inquiry_cot/layer$layer/token_analysis.json \
        --reasoning-features results/initial-setting/gemma-2-9b/general_inquiry_cot/layer$layer/reasoning_features.json \
        --layer $layer \
        --top-k-features 10 \
        --top-k-tokens 10 \
        --n-inject 3 \
        --n-samples 2000 \
        --reasoning-dataset general_inquiry_cot \
        --save-dir results/initial-setting/gemma-2-9b/general_inquiry_cot/layer$layer
done