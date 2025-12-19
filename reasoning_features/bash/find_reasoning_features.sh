for layer in 0 4 8 12 16 20 24; do
    python reasoning_features/scripts/find_reasoning_features.py \
        --layer $layer \
        --reasoning-dataset s1k \
        --reasoning-samples 2000 \
        --nonreasoning-samples 2000 \
        --save-dir results/initial-setting/gemma-2-2b/s1k/layer$layer \
        --batch-size 32
done

for layer in 0 4 8 12 16 20 24; do
    python reasoning_features/scripts/find_reasoning_features.py \
        --layer $layer \
        --reasoning-dataset general_inquiry_cot \
        --reasoning-samples 6000 \
        --nonreasoning-samples 6000 \
        --save-dir results/initial-setting/gemma-2-2b/general_inquiry_cot/layer$layer \
        --batch-size 32
done

for layer in 0 4 8 12 16 20 24; do
    python reasoning_features/scripts/find_reasoning_features.py \
        --layer $layer \
        --reasoning-dataset combined \
        --reasoning-samples 8000 \
        --nonreasoning-samples 8000 \
        --save-dir results/initial-setting/gemma-2-2b/combined/layer$layer \
        --batch-size 32
done
