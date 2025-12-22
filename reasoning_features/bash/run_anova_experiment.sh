for layer in 0 4 8 12 16 20 24; do
    python reasoning_features/scripts/run_anova_experiment.py \
        --token-analysis results/initial-setting/gemma-2-9b/s1k/layer$layer/token_analysis.json \
        --layer $layer \
        --top-k-features 100 \
        --top-k-tokens 30 \
        --reasoning-dataset s1k \
        --n-reasoning-texts 2000 \
        --n-nonreasoning-texts 10000 \
        --max-length 64 \
        --save-dir results/initial-setting/gemma-2-9b/s1k/layer$layer
    python reasoning_features/scripts/run_anova_experiment.py \
        --token-analysis results/initial-setting/gemma-2-9b/general_inquiry_cot/layer$layer/token_analysis.json \
        --layer $layer \
        --top-k-features 100 \
        --top-k-tokens 30 \
        --reasoning-dataset general_inquiry_cot \
        --n-reasoning-texts 6000 \
        --n-nonreasoning-texts 30000 \
        --max-length 64 \
        --save-dir results/initial-setting/gemma-2-9b/general_inquiry_cot/layer$layer
done
