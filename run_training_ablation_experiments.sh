#!/bin/bash

# Ablation Study: Run all 5 experiments in parallel on different GPUs
# Each experiment will save results to a clearly labeled directory

echo "🚀 Starting Ablation Study - Running 5 experiments in parallel..."
echo "📁 Results will be saved in runs/ directory with clear labels"
echo ""

# Base command arguments
BASE_CMD="python train.py --train_years 2006 2014 --test_years 2015 2018 --hidden_dim 32 --epochs 800 --cold_start_prob 0.5 --beta 0 --eval_mode team"


# Start all experiments in parallel
echo "⚡ Launching experiments..."
echo ""

# Experiment 1: Full model (cuda:0)
echo "🔬 Starting Experiment 1_full_model on GPU 0: all features"
log_file="experiment_1_full_model_gpu0.log"
$BASE_CMD --device cuda:0 --input_feature_model "all features" > "$log_file" 2>&1 &
PID1=$!
echo "   📝 PID: $PID1 | Log: $log_file"

# Experiment 2: Mean paper topic (cuda:1)  
echo "🔬 Starting Experiment 2_mean_paper_topic on GPU 1: drop topic"
log_file="experiment_2_mean_paper_topic_gpu1.log"
$BASE_CMD --device cuda:1 --input_feature_model "drop topic" > "$log_file" 2>&1 &
PID2=$!
echo "   📝 PID: $PID2 | Log: $log_file"

# Experiment 3: Mean authors topic (cuda:2)
echo "🔬 Starting Experiment 3_mean_author_topic on GPU 2: drop author topic"
log_file="experiment_3_mean_author_topic_gpu2.log"
$BASE_CMD --device cuda:2 --input_feature_model "drop author topic" > "$log_file" 2>&1 &
PID3=$!
echo "   📝 PID: $PID3 | Log: $log_file"

# Experiment 4: Mean authors social (cuda:3)
echo "🔬 Starting Experiment 4_mean_author_social on GPU 3: drop social"
log_file="experiment_4_mean_author_social_gpu3.log"
$BASE_CMD --device cuda:3 --input_feature_model "drop social" > "$log_file" 2>&1 &
PID4=$!
echo "   📝 PID: $PID4 | Log: $log_file"

# Experiment 5: Mean both author features (cuda:4)
echo "🔬 Starting Experiment 5_mean_both_author on GPU 4: drop authors"
log_file="experiment_5_mean_both_author_gpu4.log"
$BASE_CMD --device cuda:4 --input_feature_model "drop authors" > "$log_file" 2>&1 &
PID5=$!
echo "   📝 PID: $PID5 | Log: $log_file"

echo ""
echo "✅ All experiments launched!"
echo ""
echo "📊 Experiment Summary:"
echo "  1. Full Model              → GPU 0 (PID: $PID1)"
echo "  2. Mean Paper Topic        → GPU 1 (PID: $PID2)"  
echo "  3. Mean Author Topic       → GPU 2 (PID: $PID3)"
echo "  4. Mean Author Social      → GPU 3 (PID: $PID4)"
echo "  5. Mean Both Author        → GPU 4 (PID: $PID5)"
echo ""
echo "📁 Results will be in runs/ directory with names like:"
echo "   runs/YYYYMMDD_HHMMSS_team_all_features/"
echo "   runs/YYYYMMDD_HHMMSS_team_drop_topic/"
echo "   runs/YYYYMMDD_HHMMSS_team_drop_author_topic/"
echo "   runs/YYYYMMDD_HHMMSS_team_drop_social/"
echo "   runs/YYYYMMDD_HHMMSS_team_drop_authors/"
echo ""

# Function to check if process is still running
check_status() {
    local pid=$1
    local name=$2
    if kill -0 $pid 2>/dev/null; then
        echo "🟡 $name (PID: $pid) - Running"
        return 0
    else
        echo "🟢 $name (PID: $pid) - Completed"
        return 1
    fi
}

# Monitor progress
echo "🔍 Monitoring experiments (press Ctrl+C to stop monitoring, experiments will continue)..."
echo ""

while true; do
    running_count=0
    
    echo "⏰ Status at $(date '+%H:%M:%S'):"
    check_status $PID1 "Exp 1 (Full Model)" && ((running_count++))
    check_status $PID2 "Exp 2 (Mean Paper Topic)" && ((running_count++))
    check_status $PID3 "Exp 3 (Mean Author Topic)" && ((running_count++))
    check_status $PID4 "Exp 4 (Mean Author Social)" && ((running_count++))
    check_status $PID5 "Exp 5 (Mean Both Author)" && ((running_count++))
    
    echo "   📈 $running_count/5 experiments still running"
    echo ""
    
    if [ $running_count -eq 0 ]; then
        echo "🎉 All experiments completed!"
        break
    fi
    
    sleep 30  # Check every 30 seconds
done

echo ""
echo "📋 Final Results Summary:"
echo "  Check the runs/ directory for detailed results from each experiment"
echo "  Each experiment has its own clearly labeled directory"
echo "  Training logs are also available in experiment_*_gpu*.log files"
echo ""
echo "✨ Ablation study complete!"