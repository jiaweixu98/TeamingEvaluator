#!/usr/bin/env python3
"""
SIMPLE TEAM SENSITIVITY ANALYSIS

This script provides a clear, easy-to-understand analysis of how your model's 
citation predictions change when you modify team composition.

STRAIGHTFORWARD APPROACH:
1. Load full model predictions (baseline)
2. Load different team ablation predictions  
3. Compare average citation counts
4. Show which team changes matter most
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_experiment_results(pred_file):
    """Load prediction results and metrics from files."""
    try:
        metrics_file = pred_file.replace('_predictions.npy', '_metrics.json')
        
        if os.path.exists(pred_file) and os.path.exists(metrics_file):
            predictions = np.load(pred_file)
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return predictions, metrics
        else:
            return None, None
    except Exception as e:
        print(f"Error loading {pred_file}: {e}")
        return None, None

def compare_predictions(baseline_predictions, experiment_predictions, exp_name):
    """
    Compare citation predictions between baseline and experiment in simple terms.
    
    Parameters:
    -----------
    baseline_predictions : np.array of shape (n_papers, n_years)
        Predictions from the full model (all authors)
    experiment_predictions : np.array of shape (n_papers, n_years)  
        Predictions from the ablation experiment
    exp_name : str
        Name of the experiment for reporting
        
    Returns:
    --------
    dict : Simple metrics showing how predictions changed
    """
    
    # Convert to total citations per paper (sum across 5 years)
    baseline_total = baseline_predictions.sum(axis=1)
    experiment_total = experiment_predictions.sum(axis=1)
    
    # Simple metrics that are easy to understand
    baseline_mean = baseline_total.mean()
    experiment_mean = experiment_total.mean()
    mean_change = experiment_mean - baseline_mean  # Positive = increase, Negative = decrease
    
    # How much do individual papers change on average?
    per_paper_changes = experiment_total - baseline_total
    avg_absolute_change = np.abs(per_paper_changes).mean()
    
    # What percentage of papers see big changes? (>5 citations different)
    big_changes = np.abs(per_paper_changes) > 5
    percent_big_changes = (big_changes.sum() / len(big_changes)) * 100
    
    return {
        'experiment': exp_name,
        'baseline_mean_citations': round(float(baseline_mean), 1),
        'experiment_mean_citations': round(float(experiment_mean), 1), 
        'mean_change': round(float(mean_change), 1),
        'avg_absolute_change_per_paper': round(float(avg_absolute_change), 1),
        'percent_papers_with_big_changes': round(float(percent_big_changes), 1)
    }

def interpret_change(mean_change, avg_absolute_change, percent_big_changes):
    """
    Give a simple interpretation of how much the predictions changed.
    
    Parameters:
    -----------
    mean_change : float
        Average change in citations (can be positive or negative)
    avg_absolute_change : float  
        Average absolute change per paper
    percent_big_changes : float
        Percentage of papers with >5 citation changes
        
    Returns:
    --------
    str : Simple interpretation
    """
    
    # Determine impact level based on simple thresholds
    if avg_absolute_change > 7 and percent_big_changes > 30:
        impact = "🎯 HIGH IMPACT"
        meaning = "Team composition changes significantly affect citation predictions"
    elif avg_absolute_change > 3 and percent_big_changes > 15:
        impact = "✅ MODERATE IMPACT" 
        meaning = "Team composition changes have noticeable effects on predictions"
    else:
        impact = "⚠️ LOW IMPACT"
        meaning = "Team composition changes have minimal effect on predictions"
    
    # Direction of change
    if abs(mean_change) < 0.5:
        direction = "no overall bias"
    elif mean_change > 0:
        direction = f"tends to increase citations by {abs(mean_change)} on average"
    else:
        direction = f"tends to decrease citations by {abs(mean_change)} on average"
    
    return f"{impact} - {meaning}, {direction}"

def print_simple_results(metrics):
    """Print easy-to-understand results for one experiment."""
    
    interpretation = interpret_change(
        metrics['mean_change'],
        metrics['avg_absolute_change_per_paper'], 
        metrics['percent_papers_with_big_changes']
    )
    
    print(f"\n{'='*70}")
    print(f"🔬 EXPERIMENT: {metrics['experiment'].replace('_', ' ').upper()}")
    print(f"{'='*70}")
    
    print(f"\n📊 CITATION PREDICTION CHANGES:")
    print(f"   Full model (baseline): {metrics['baseline_mean_citations']} citations on average")
    print(f"   This experiment:      {metrics['experiment_mean_citations']} citations on average")
    print(f"   Overall change:       {metrics['mean_change']:+.1f} citations")
    
    print(f"\n🔍 HOW MUCH DO INDIVIDUAL PAPERS CHANGE?")
    print(f"   Average change per paper: {metrics['avg_absolute_change_per_paper']} citations")
    print(f"   Papers with big changes (>5 citations): {metrics['percent_papers_with_big_changes']}%")
    
    print(f"\n🎯 INTERPRETATION:")
    print(f"   {interpretation}")

def create_simple_summary_table(all_results):
    """Create an easy-to-read summary table."""
    
    print(f"\n{'='*90}")
    print(f"📊 SUMMARY: HOW TEAM CHANGES AFFECT CITATION PREDICTIONS")
    print(f"{'='*90}")
    
    # Table header
    header = f"{'Experiment':<20} {'Baseline':<10} {'Changed':<10} {'Difference':<12} {'Per-Paper':<12} {'Impact':<15}"
    print(header)
    print("-" * 90)
    
    # Table rows
    for result in all_results:
        
        # Determine impact symbol
        if result['avg_absolute_change_per_paper'] > 7:
            impact_symbol = "🎯 HIGH"
        elif result['avg_absolute_change_per_paper'] > 3:
            impact_symbol = "✅ MODERATE"  
        else:
            impact_symbol = "⚠️ LOW"
        
        row = (f"{result['experiment']:<20} "
               f"{result['baseline_mean_citations']:<10.1f} " 
               f"{result['experiment_mean_citations']:<10.1f} "
               f"{result['mean_change']:+8.1f}    "
               f"{result['avg_absolute_change_per_paper']:<12.1f} "
               f"{impact_symbol:<15}")
        print(row)

def analyze_overall_impact(all_results):
    """Give a simple overall assessment."""
    
    if not all_results:
        print("❌ No experiments to analyze!")
        return
    
    # Simple statistics
    all_changes = [r['avg_absolute_change_per_paper'] for r in all_results]
    avg_change = np.mean(all_changes)
    max_change = np.max(all_changes)
    
    # Count impact levels
    high_impact = sum(1 for c in all_changes if c > 7)
    moderate_impact = sum(1 for c in all_changes if 3 < c <= 7)
    low_impact = sum(1 for c in all_changes if c <= 3)
    
    print(f"\n{'='*80}")
    print(f"🎯 OVERALL MODEL ASSESSMENT FOR COUNTERFACTUAL PREDICTION")
    print(f"{'='*80}")
    
    print(f"\n📈 KEY FINDINGS:")
    print(f"   📊 Average change across experiments: {avg_change:.1f} citations per paper")
    print(f"   📊 Biggest change observed: {max_change:.1f} citations per paper")
    print(f"   📊 Number of experiments tested: {len(all_results)}")
    
    print(f"\n🔍 IMPACT DISTRIBUTION:")
    print(f"   🎯 High impact experiments: {high_impact}")
    print(f"   ✅ Moderate impact experiments: {moderate_impact}")  
    print(f"   ⚠️ Low impact experiments: {low_impact}")
    
    # Simple recommendation
    print(f"\n🎯 BOTTOM LINE FOR YOUR RESEARCH:")
    if avg_change > 6:
        print("   🚀 EXCELLENT: Your model is sensitive to team changes!")
        print("   → Perfect for counterfactual team prediction research")
        print("   → Team composition clearly affects citation predictions")
    elif avg_change > 3:
        print("   ✅ GOOD: Your model shows meaningful sensitivity to team changes")
        print("   → Suitable for counterfactual analysis")
        print("   → Some team changes matter more than others")
    else:
        print("   ⚠️ LIMITED: Your model has low sensitivity to team changes")
        print("   → Team composition has minimal impact on predictions")
        print("   → Model may be too topic-focused for counterfactual prediction")

def main():
    """Run simple, clear team sensitivity analysis."""
    
    print("🔬 TEAM SENSITIVITY ANALYSIS")
    print("=" * 60)
    print("How do citation predictions change when you modify team composition?")
    print("=" * 60)
    
    # Load all experiment files
    pred_files = []
    if os.path.exists("./sensitivity_results"):
        for filename in os.listdir("./sensitivity_results"):
            if filename.endswith("_predictions.npy"):
                pred_files.append(os.path.join("./sensitivity_results", filename))
    
    if len(pred_files) < 2:
        print("❌ Need at least 2 experiments for sensitivity analysis!")
        return
    
    print(f"📁 Found {len(pred_files)} prediction files")
    
    # Load all results
    all_predictions = {}
    all_raw_metrics = {}
    
    for pred_file in pred_files:
        predictions, raw_metrics = load_experiment_results(pred_file)
        
        if predictions is not None:
            filename = os.path.basename(pred_file)
            exp_name = filename.replace("author_", "").replace("__topic_all features_predictions.npy", "")
            all_predictions[exp_name] = predictions
            all_raw_metrics[exp_name] = raw_metrics
            print(f"✓ Loaded: {exp_name}")
    
    # Find baseline
    baseline_name = "all authors"
    if baseline_name not in all_predictions:
        print(f"❌ Baseline '{baseline_name}' not found!")
        return
    
    baseline_predictions = all_predictions[baseline_name]
    print(f"\n📍 Using baseline: {baseline_name}")
    
    # Compare each experiment to the baseline
    all_results = []
    
    for exp_name, predictions in all_predictions.items():
        if exp_name == baseline_name:
            continue
            
        # Simple comparison
        result = compare_predictions(baseline_predictions, predictions, exp_name)
        all_results.append(result)
        
        # Print results for this experiment
        print_simple_results(result)
    
    # Create summary table
    create_simple_summary_table(all_results)
    
    # Overall analysis
    analyze_overall_impact(all_results)
    
    # Save simple results (convert numpy types to regular Python types)
    output = {
        'baseline_model': baseline_name,
        'results': all_results,
        'summary': {
            'total_experiments': len(all_results),
            'average_change': float(np.mean([r['avg_absolute_change_per_paper'] for r in all_results])) if all_results else 0,
            'explanation': 'This shows how much citation predictions change when you modify team composition'
        }
    }
    
    with open('./sensitivity_results/team_sensitivity_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: ./sensitivity_results/team_sensitivity_analysis.json")
    print("\n🎉 Rigorous sensitivity analysis complete!")

if __name__ == "__main__":
    main()