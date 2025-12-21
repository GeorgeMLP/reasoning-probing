for layer in 0 4 8 12 16 20 24; do
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/initial-setting/gemma-2-2b/s1k/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark aime24 \
        --save-dir results/initial-setting/gemma-2-2b/s1k/layer$layer/aime24 \
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/initial-setting/gemma-2-2b/s1k/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark gpqa_diamond \
        --save-dir results/initial-setting/gemma-2-2b/s1k/layer$layer/gpqa_diamond \
done

for layer in 0 4 8 12 16 20 24; do
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/initial-setting/gemma-2-2b/general_inquiry_cot/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark aime24 \
        --save-dir results/initial-setting/gemma-2-2b/general_inquiry_cot/layer$layer/aime24 \
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/initial-setting/gemma-2-2b/general_inquiry_cot/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark gpqa_diamond \
        --save-dir results/initial-setting/gemma-2-2b/general_inquiry_cot/layer$layer/gpqa_diamond \
done
