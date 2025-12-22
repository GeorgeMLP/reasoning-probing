for layer in 0 4 8 12 16 20 24; do
    python reasoning_features/scripts/run_anova_experiment.py \
        --token-analysis results/initial-setting/gemma-2-9b/s1k/layer$layer/token_analysis.json \
        --layer $layer \
        --top-k-features 20 \
        --top-k-tokens 30 \
        --reasoning-dataset s1k \
        --n-reasoning-chains 2000 \
        --n-nonreasoning-samples 10000 \
        --n-per-condition 5000 \
        --max-length 64 \
        --save-dir results/initial-setting/gemma-2-9b/s1k/layer$layer
    python reasoning_features/scripts/run_anova_experiment.py \
        --token-analysis results/initial-setting/gemma-2-9b/s1k/layer$layer/token_analysis.json \
        --layer $layer \
        --top-k-features 20 \
        --top-k-tokens 30 \
        --reasoning-dataset general_inquiry_cot \
        --n-reasoning-chains 2000 \
        --n-nonreasoning-samples 10000 \
        --n-per-condition 5000 \
        --max-length 64 \
        --save-dir results/initial-setting/gemma-2-9b/general_inquiry_cot/layer$layer
done
