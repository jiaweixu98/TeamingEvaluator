#!/bin/bash

# Team Sensitivity Analysis Runner
# Usage: ./run_all_sensitivity_tests.sh path/to/full_model_checkpoint.pt [gpu_id]
# runs/20250731_175915_team_all_features/evaluated_model_epoch030_male0_0.657_male1_0.685_male2_0.700_male3_0.720_male4_0.718_team.pt

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [gpu_id]"
    echo "Example: $0 runs/20250731_175915_team_all_features/best_model_epoch070.pt 0"
    exit 1
fi

CHECKPOINT_PATH="$1"
GPU_ID="${2:-0}"  # Default to GPU 0 if not specified

echo "=================================================="
echo "RUNNING TEAM SENSITIVITY ANALYSIS"
echo "=================================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "GPU ID: $GPU_ID"
echo "=================================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Create directories
mkdir -p sensitivity_results
mkdir -p sensitivity_plots
mkdir -p figs

echo "🚀 Running systematic sensitivity analysis..."

# Run the comprehensive sensitivity analysis
echo "🔬 Running all team sensitivity experiments..."

# Base command template
BASE_CMD="python train.py --training_off 1 --load_checkpoint $CHECKPOINT_PATH --train_years 2006 2014 --test_years 2015 2018"

echo "🧪 Experiment 1: All Authors + Mean Topic (missing experiment)"
$BASE_CMD --input_feature_model 'all features' --inference_time_author_dropping 'all authors' --inference_time_topic_dropping 'drop topic'

echo "🧪 Experiment 2: Mean Authors + Real Topic (missing experiment)" 
$BASE_CMD --input_feature_model 'drop authors' --inference_time_author_dropping 'all authors'

echo "✅ All missing experiments completed! Running analysis..."
python team_sensitivity_analysis.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SENSITIVITY ANALYSIS COMPLETE!"
    echo ""
    echo "📊 Results saved to:"
    echo "   - sensitivity_results/team_sensitivity_analysis.json"
    echo "   - Individual experiment files in sensitivity_results/"
    echo ""
    echo "🔍 Complete sensitivity matrix now includes:"
    echo "   ✅ All Authors + Real Topic (baseline)"
    echo "   ✅ All Authors + Mean Topic (topic ablation)"  
    echo "   ✅ Mean Authors + Real Topic (author ablation)"
    echo "   ✅ No Authors + Real Topic (pure topic)"
    echo "   ✅ Various author modifications + Real Topic"
    echo ""
    echo "🎯 INTERPRETATION GUIDE:"
    echo "   HIGH sensitivity = GOOD for counterfactual prediction"
    echo "   LOW sensitivity = Model ignores team composition (BAD)"
    echo ""
else
    echo "❌ Error running sensitivity analysis"
    exit 1
fi