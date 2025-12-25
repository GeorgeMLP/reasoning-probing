for layer in 0 4 8 12 16 20 24; do
    python reasoning_features/scripts/visualize_injection_features.py \
        --injection-results results/64maxlen/gemma-2-9b/s1k/layer$layer/injection_results.json \
        --token-analysis results/64maxlen/gemma-2-9b/s1k/layer$layer/token_analysis.json \
        --layer $layer \
        --n-features 100 \
        --n-examples 10 \
        --max-seq-len 256 \
        --reasoning-dataset s1k \
        --output-dir visualizations/64maxlen/gemma-2-9b/s1k/layer$layer
    python reasoning_features/scripts/visualize_injection_features.py \
        --injection-results results/64maxlen/gemma-2-9b/general_inquiry_cot/layer$layer/injection_results.json \
        --token-analysis results/64maxlen/gemma-2-9b/general_inquiry_cot/layer$layer/token_analysis.json \
        --layer $layer \
        --n-features 100 \
        --n-examples 10 \
        --max-seq-len 256 \
        --reasoning-dataset general_inquiry_cot \
        --output-dir visualizations/64maxlen/gemma-2-9b/general_inquiry_cot/layer$layer
done