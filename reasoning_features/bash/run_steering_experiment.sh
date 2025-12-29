for layer in 22; do
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/cohens_d/gemma-3-4b-it/s1k/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark aime24 \
        --gamma-values 0.0 2.0 \
        --max-gen-toks 16384 \
        --save-dir results/cohens_d/gemma-3-4b-it/s1k/layer$layer/aime24
done
for layer in 22; do
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/cohens_d/gemma-3-4b-it/s1k/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark gpqa_diamond \
        --gamma-values 0.0 2.0 \
        --max-gen-toks 16384 \
        --save-dir results/cohens_d/gemma-3-4b-it/s1k/layer$layer/gpqa_diamond
done

for layer in 22; do
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/cohens_d/gemma-3-4b-it/general_inquiry_cot/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark aime24 \
        --gamma-values 0.0 2.0 \
        --max-gen-toks 16384 \
        --save-dir results/cohens_d/gemma-3-4b-it/general_inquiry_cot/layer$layer/aime24
done
for layer in 22; do
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/cohens_d/gemma-3-4b-it/general_inquiry_cot/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark gpqa_diamond \
        --gamma-values 0.0 2.0 \
        --max-gen-toks 16384 \
        --save-dir results/cohens_d/gemma-3-4b-it/general_inquiry_cot/layer$layer/gpqa_diamond
done
