for layer in 0 4 8 12 16 20 24; do
    python reasoning_features/scripts/run_token_injection_experiment.py \
        --token-analysis results/64maxlen/gemma-2-9b/s1k/layer$layer/token_analysis.json \
        --reasoning-features results/64maxlen/gemma-2-9b/s1k/layer$layer/reasoning_features.json \
        --layer $layer \
        --top-k-features 100 \
        --top-k-tokens 10 \
        --n-inject 3 \
        --n-inject-bigram 2 \
        --n-inject-trigram 1 \
        --active-trigram-threshold 0.5 \
        --n-samples 500 \
        --reasoning-dataset s1k \
        --save-dir results/64maxlen/gemma-2-9b/s1k/layer$layer \
        --batch-size 16 \
        --max-length 64
    python reasoning_features/scripts/run_token_injection_experiment.py \
        --token-analysis results/64maxlen/gemma-2-9b/general_inquiry_cot/layer$layer/token_analysis.json \
        --reasoning-features results/64maxlen/gemma-2-9b/general_inquiry_cot/layer$layer/reasoning_features.json \
        --layer $layer \
        --top-k-features 100 \
        --top-k-tokens 10 \
        --n-inject 3 \
        --n-inject-bigram 2 \
        --n-inject-trigram 1 \
        --active-trigram-threshold 0.5 \
        --n-samples 500 \
        --reasoning-dataset general_inquiry_cot \
        --save-dir results/64maxlen/gemma-2-9b/general_inquiry_cot/layer$layer \
        --batch-size 16 \
        --max-length 64
done