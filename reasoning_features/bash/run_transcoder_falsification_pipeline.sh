FEATURE_BACKEND=${FEATURE_BACKEND:-clt}
MODEL_NAME=${MODEL_NAME:-google/gemma-2-2b}
LAYER=${LAYER:-17}
REASONING_SAMPLES=${REASONING_SAMPLES:-1000}
NONREASONING_SAMPLES=${NONREASONING_SAMPLES:-1000}
INJECTION_SAMPLES=${INJECTION_SAMPLES:-500}
MAX_LENGTH=${MAX_LENGTH:-64}
TOP_K_FEATURES=${TOP_K_FEATURES:-100}
TOP_K_TOKENS=${TOP_K_TOKENS:-30}

if [ "$FEATURE_BACKEND" = "clt" ]; then
  TRANSCODER_SET=${TRANSCODER_SET:-mntss/clt-gemma-2-2b-426k}
else
  TRANSCODER_SET=${TRANSCODER_SET:-mntss/gemma-scope-transcoders}
fi

for DATASET in s1k general_inquiry_cot; do
  SAVE_DIR="results/cohens_d/gemma-2-2b/${FEATURE_BACKEND}/${DATASET}/layer${LAYER}"

  python reasoning_features/scripts/find_reasoning_features.py \
    --model-name "$MODEL_NAME" \
    --feature-backend "$FEATURE_BACKEND" \
    --transcoder-set "$TRANSCODER_SET" \
    --layer "$LAYER" \
    --reasoning-dataset "$DATASET" \
    --reasoning-samples "$REASONING_SAMPLES" \
    --nonreasoning-samples "$NONREASONING_SAMPLES" \
    --max-length "$MAX_LENGTH" \
    --top-k-features "$TOP_K_FEATURES" \
    --top-k-tokens "$TOP_K_TOKENS" \
    --score-weight-auc 0.0 \
    --score-weight-effect 1.0 \
    --score-weight-pvalue 0.0 \
    --score-weight-freq 0.0 \
    --no-filter \
    --save-dir "$SAVE_DIR"

  python reasoning_features/scripts/run_token_injection_experiment.py \
    --model-name "$MODEL_NAME" \
    --feature-backend "$FEATURE_BACKEND" \
    --transcoder-set "$TRANSCODER_SET" \
    --token-analysis "$SAVE_DIR/token_analysis.json" \
    --reasoning-features "$SAVE_DIR/reasoning_features.json" \
    --layer "$LAYER" \
    --top-k-features "$TOP_K_FEATURES" \
    --top-k-tokens 10 \
    --n-samples "$INJECTION_SAMPLES" \
    --reasoning-dataset "$DATASET" \
    --max-length "$MAX_LENGTH" \
    --save-dir "$SAVE_DIR"

  python reasoning_features/scripts/analyze_feature_interpretation.py \
    --model-name "$MODEL_NAME" \
    --feature-backend "$FEATURE_BACKEND" \
    --transcoder-set "$TRANSCODER_SET" \
    --injection-results "$SAVE_DIR/injection_results.json" \
    --token-analysis "$SAVE_DIR/token_analysis.json" \
    --mode context_dependent \
    --layer "$LAYER" \
    --reasoning-dataset "$DATASET" \
    --max-iterations 20 \
    --min-false-positives 3 \
    --min-false-negatives 3 \
    --threshold-ratio 0.5 \
    --max-features 20 \
    --output "$SAVE_DIR/feature_interpretations.json"
done
