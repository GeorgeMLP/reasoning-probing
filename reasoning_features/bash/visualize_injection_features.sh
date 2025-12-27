for layer in 17 22 27; do
    python reasoning_features/scripts/visualize_injection_features.py \
        --injection-results results/roc-auc/gemma-3-4b-it/s1k/layer$layer/injection_results.json \
        --token-analysis results/roc-auc/gemma-3-4b-it/s1k/layer$layer/token_analysis.json \
        --layer $layer \
        --n-features 100 \
        --n-examples 10 \
        --max-seq-len 256 \
        --reasoning-dataset s1k \
        --output-dir visualizations/roc-auc/gemma-3-4b-it/s1k/layer$layer
    python reasoning_features/scripts/visualize_injection_features.py \
        --injection-results results/roc-auc/gemma-3-4b-it/general_inquiry_cot/layer$layer/injection_results.json \
        --token-analysis results/roc-auc/gemma-3-4b-it/general_inquiry_cot/layer$layer/token_analysis.json \
        --layer $layer \
        --n-features 100 \
        --n-examples 10 \
        --max-seq-len 256 \
        --reasoning-dataset general_inquiry_cot \
        --output-dir visualizations/roc-auc/gemma-3-4b-it/general_inquiry_cot/layer$layer
done