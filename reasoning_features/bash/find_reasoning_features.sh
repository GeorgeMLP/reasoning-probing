for layer in 0 4 8 12 16 20 24; do
    python reasoning_features/scripts/find_reasoning_features.py \
        --layer $layer \
        --reasoning-dataset s1k \
        --reasoning-samples 2000 \
        --nonreasoning-samples 2000 \
        --max-length 64 \
        --top-k-features 100 \
        --top-k-tokens 30 \
        --save-dir results/initial-setting/gemma-2-9b/s1k/layer$layer \
        --batch-size 16
done

for layer in 0 4 8 12 16 20 24; do
    python reasoning_features/scripts/find_reasoning_features.py \
        --layer $layer \
        --reasoning-dataset general_inquiry_cot \
        --reasoning-samples 6000 \
        --nonreasoning-samples 6000 \
        --max-length 64 \
        --top-k-features 100 \
        --top-k-tokens 30 \
        --save-dir results/initial-setting/gemma-2-9b/general_inquiry_cot/layer$layer \
        --batch-size 16
done
