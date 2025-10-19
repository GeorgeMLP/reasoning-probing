#!/bin/bash
# Convenience script to run common probing experiments

set -e

# Default values
MODEL="google/gemma-2-2b"
LAYER=8
DEVICE="cuda"
BASE_DIR="data/probing/experiments"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "SAE Probing Experiment Runner"
echo "=========================================="
echo ""

# Parse command line arguments
EXPERIMENT_TYPE=${1:-"quick"}

case $EXPERIMENT_TYPE in
    "quick")
        echo -e "${BLUE}Running quick test experiment (10 min)${NC}"
        python probing/run_probing_experiment.py \
            --model_name $MODEL \
            --layer_index $LAYER \
            --normal_samples 100 \
            --reasoning_samples 50 \
            --probe_type linear \
            --label_type binary \
            --batch_size 32 \
            --num_epochs 50 \
            --patience 10 \
            --device $DEVICE \
            --save_dir $BASE_DIR/quick_test
        ;;
    
    "small")
        echo -e "${BLUE}Running small experiment (~30 min)${NC}"
        python probing/run_probing_experiment.py \
            --model_name $MODEL \
            --layer_index $LAYER \
            --normal_samples 1000 \
            --reasoning_samples 200 \
            --probe_type linear \
            --label_type binary \
            --batch_size 64 \
            --num_epochs 100 \
            --patience 10 \
            --device $DEVICE \
            --save_dir $BASE_DIR/small_exp
        ;;
    
    "full")
        echo -e "${BLUE}Running full experiment (~2 hours)${NC}"
        python probing/run_probing_experiment.py \
            --model_name $MODEL \
            --layer_index $LAYER \
            --normal_samples 5000 \
            --reasoning_samples 1000 \
            --probe_type linear \
            --label_type binary \
            --batch_size 64 \
            --num_epochs 100 \
            --patience 15 \
            --device $DEVICE \
            --save_dir $BASE_DIR/full_exp
        ;;
    
    "mlp")
        echo -e "${BLUE}Running MLP probe experiment${NC}"
        # First run or reuse activations from previous experiment
        if [ -f "$BASE_DIR/full_exp/activations/activations.pt" ]; then
            echo "Reusing activations from full_exp..."
            python probing/run_probing_experiment.py \
                --load_activations $BASE_DIR/full_exp/activations/activations.pt \
                --probe_type mlp_2 \
                --batch_size 64 \
                --num_epochs 100 \
                --patience 15 \
                --device $DEVICE \
                --save_dir $BASE_DIR/mlp_exp
        else
            echo "Collecting new activations..."
            python probing/run_probing_experiment.py \
                --model_name $MODEL \
                --layer_index $LAYER \
                --normal_samples 5000 \
                --reasoning_samples 500 \
                --probe_type mlp_2 \
                --label_type binary \
                --batch_size 64 \
                --num_epochs 100 \
                --patience 15 \
                --device $DEVICE \
                --save_dir $BASE_DIR/mlp_exp
        fi
        ;;
    
    "fine_grained")
        echo -e "${BLUE}Running fine-grained classification experiment${NC}"
        python probing/run_probing_experiment.py \
            --model_name $MODEL \
            --layer_index $LAYER \
            --normal_samples 5000 \
            --reasoning_samples 500 \
            --probe_type mlp_2 \
            --label_type fine_grained \
            --batch_size 64 \
            --num_epochs 100 \
            --patience 15 \
            --device $DEVICE \
            --save_dir $BASE_DIR/fine_grained_exp
        ;;
    
    "multi_layer")
        echo -e "${BLUE}Running multi-layer comparison experiment${NC}"
        for LAYER in 4 8 12 16; do
            echo "Processing layer $LAYER..."
            python probing/run_probing_experiment.py \
                --model_name $MODEL \
                --layer_index $LAYER \
                --normal_samples 1000 \
                --reasoning_samples 200 \
                --probe_type linear \
                --label_type binary \
                --batch_size 64 \
                --num_epochs 100 \
                --patience 10 \
                --device $DEVICE \
                --save_dir $BASE_DIR/layer_${LAYER}_exp
        done
        
        echo -e "${GREEN}Generating comparison plots...${NC}"
        python probing/analyze_results.py \
            $BASE_DIR/layer_*_exp \
            --plot --compare \
            --save_plots $BASE_DIR/multi_layer_comparison
        ;;
    
    "validate")
        echo -e "${BLUE}Running validation tests${NC}"
        python probing/test_implementation.py
        ;;
    
    *)
        echo "Unknown experiment type: $EXPERIMENT_TYPE"
        echo ""
        echo "Usage: $0 [EXPERIMENT_TYPE]"
        echo ""
        echo "Available experiment types:"
        echo "  quick         - Quick test (100 normal, 50 reasoning samples, ~10 min)"
        echo "  small         - Small experiment (1000 normal, 200 reasoning, ~30 min)"
        echo "  full          - Full experiment (5000 normal, 500 reasoning, ~2 hours)"
        echo "  mlp           - MLP probe experiment (reuses activations if available)"
        echo "  fine_grained  - Fine-grained reasoning classification"
        echo "  multi_layer   - Compare multiple layers (4, 8, 12, 16)"
        echo "  validate      - Run validation tests"
        echo ""
        echo "Examples:"
        echo "  $0 quick          # Run quick test"
        echo "  $0 full           # Run full experiment"
        echo "  $0 multi_layer    # Compare layers"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Experiment completed!${NC}"
echo "Results saved to: $BASE_DIR"
echo ""
echo "To analyze results, run:"
echo "  python probing/analyze_results.py $BASE_DIR/[exp_name] --plot"

