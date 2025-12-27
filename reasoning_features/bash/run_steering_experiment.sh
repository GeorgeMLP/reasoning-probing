for layer in 17 22 27; do
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/cohens_d/gemma-3-4b-it/s1k/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark aime24 \
        --max-gen-toks 32768 \
        --save-dir results/cohens_d/gemma-3-4b-it/s1k/layer$layer/aime24
done
for layer in 17 22 27; do
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/cohens_d/gemma-3-4b-it/s1k/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark gpqa_diamond \
        --max-gen-toks 32768 \
        --save-dir results/cohens_d/gemma-3-4b-it/s1k/layer$layer/gpqa_diamond
done
for layer in 17 22 27; do
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/cohens_d/gemma-3-4b-it/s1k/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark math500 \
        --max-gen-toks 32768 \
        --save-dir results/cohens_d/gemma-3-4b-it/s1k/layer$layer/math500
done

for layer in 17 22 27; do
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/cohens_d/gemma-3-4b-it/general_inquiry_cot/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark aime24 \
        --max-gen-toks 32768 \
        --save-dir results/cohens_d/gemma-3-4b-it/general_inquiry_cot/layer$layer/aime24
done
for layer in 17 22 27; do
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/cohens_d/gemma-3-4b-it/general_inquiry_cot/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark gpqa_diamond \
        --max-gen-toks 32768 \
        --save-dir results/cohens_d/gemma-3-4b-it/general_inquiry_cot/layer$layer/gpqa_diamond
done
for layer in 17 22 27; do
    python reasoning_features/scripts/run_steering_experiment.py \
        --layer $layer \
        --features-file results/cohens_d/gemma-3-4b-it/general_inquiry_cot/layer$layer/reasoning_features.json \
        --top-k-features 10 \
        --benchmark math500 \
        --max-gen-toks 32768 \
        --save-dir results/cohens_d/gemma-3-4b-it/general_inquiry_cot/layer$layer/math500
done
