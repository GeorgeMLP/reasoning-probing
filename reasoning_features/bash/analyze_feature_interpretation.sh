for layer in 17 22 27; do
    python reasoning_features/scripts/analyze_feature_interpretation.py \
        --injection-results results/cohens_d/gemma-3-4b-it/s1k/layer$layer/injection_results.json \
        --token-analysis results/cohens_d/gemma-3-4b-it/s1k/layer$layer/token_analysis.json \
        --mode context_dependent \
        --layer $layer \
        --llm-model google/gemini-3-pro-preview \
        --reasoning-dataset s1k \
        --max-iterations 10 \
        --min-false-positives 3 \
        --min-false-negatives 3 \
        --threshold-ratio 0.5 \
        --max-features 20 \
        --output results/cohens_d/gemma-3-4b-it/s1k/layer$layer/feature_interpretations.json
    python reasoning_features/scripts/analyze_feature_interpretation.py \
        --injection-results results/cohens_d/gemma-3-4b-it/general_inquiry_cot/layer$layer/injection_results.json \
        --token-analysis results/cohens_d/gemma-3-4b-it/general_inquiry_cot/layer$layer/token_analysis.json \
        --mode context_dependent \
        --layer $layer \
        --llm-model google/gemini-3-pro-preview \
        --reasoning-dataset general_inquiry_cot \
        --max-iterations 10 \
        --min-false-positives 3 \
        --min-false-negatives 3 \
        --threshold-ratio 0.5 \
        --max-features 20 \
        --output results/cohens_d/gemma-3-4b-it/general_inquiry_cot/layer$layer/feature_interpretations.json
done