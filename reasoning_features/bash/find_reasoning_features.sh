for layer in 17 22 27; do
    python reasoning_features/scripts/find_reasoning_features.py \
        --layer $layer \
        --reasoning-dataset s1k \
        --reasoning-samples 2000 \
        --nonreasoning-samples 2000 \
        --max-length 64 \
        --top-k-features 100 \
        --top-k-tokens 30 \
        --score-weight-auc 1.0 \
        --score-weight-effect 0.0 \
        --score-weight-pvalue 0.0 \
        --score-weight-freq 0.0 \
        --save-dir results/roc-auc/gemma-3-4b-it/s1k/layer$layer \
        --batch-size 16
done

for layer in 17 22 27; do
    python reasoning_features/scripts/find_reasoning_features.py \
        --layer $layer \
        --reasoning-dataset general_inquiry_cot \
        --reasoning-samples 6000 \
        --nonreasoning-samples 6000 \
        --max-length 64 \
        --top-k-features 100 \
        --top-k-tokens 30 \
        --score-weight-auc 1.0 \
        --score-weight-effect 0.0 \
        --score-weight-pvalue 0.0 \
        --score-weight-freq 0.0 \
        --save-dir results/roc-auc/gemma-3-4b-it/general_inquiry_cot/layer$layer \
        --batch-size 16
done
