#!/usr/bin/env python3
"""
Generate comprehensive visualizations comparing baseline vs hardware-aware models.

Creates graphs for:
- Training loss over rounds
- Training time/latency per round
- Energy consumption
- Model accuracy comparison
- Per-client latency breakdown
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import sys

# Try to import required libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
except ImportError:
    print("Installing matplotlib...")
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib
    matplotlib.use('Agg')

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("Installing seaborn for better plots...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns
    sns.set_style("whitegrid")


def load_metrics(experiment_name):
    """Load metrics from CSV file."""
    metrics_path = Path(f"results/metrics_{experiment_name}.csv")
    if metrics_path.exists():
        try:
            df = pd.read_csv(metrics_path)
            # Check if dataframe is empty or has no data rows
            if len(df) > 0 and not df.empty:
                return df
        except Exception as e:
            print(f"Warning: Could not load {metrics_path}: {e}")

    # Also try generic metrics.csv
    generic_path = Path("results/metrics.csv")
    if generic_path.exists():
        try:
            df = pd.read_csv(generic_path)
            if len(df) > 0 and not df.empty:
                # Filter by experiment if possible
                return df
        except:
            pass

    return None


def evaluate_model(model_path, experiment_name):
    """Evaluate a model and return metrics."""
    from cecs_574_dist_computing_project.task import Net, load_data, test

    if not Path(model_path).exists():
        return None

    try:
        model = Net()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Use partition 0 for evaluation
        _, testloader = load_data(0, 10, batch_size=32)

        eval_loss, eval_acc = test(model, testloader, device)
        return {"loss": eval_loss, "accuracy": eval_acc}
    except Exception as e:
        print(f"Warning: Could not evaluate {model_path}: {e}")
        return None


def create_loss_comparison(baseline_df, hardware_df, output_dir):
    """Create training loss comparison graph."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if baseline_df is not None and 'train_loss' in baseline_df.columns:
        rounds = baseline_df.get('round', range(len(baseline_df)))
        ax.plot(rounds, baseline_df['train_loss'],
                marker='o', label='Baseline', linewidth=2, markersize=8)

    if hardware_df is not None and 'train_loss' in hardware_df.columns:
        rounds = hardware_df.get('round', range(len(hardware_df)))
        ax.plot(rounds, hardware_df['train_loss'],
                marker='s', label='Hardware-Aware', linewidth=2, markersize=8)

    ax.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Comparison: Baseline vs Hardware-Aware',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'loss_comparison.png'}")


def create_latency_comparison(baseline_df, hardware_df, output_dir):
    """Create training time/latency comparison graphs."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Training time per round
    ax1 = axes[0]
    if baseline_df is not None and 'train_time_sec' in baseline_df.columns:
        rounds = baseline_df.get('round', range(len(baseline_df)))
        ax1.plot(rounds, baseline_df['train_time_sec'],
                 marker='o', label='Baseline', linewidth=2, markersize=8, color='#2E86AB')
        ax1.bar([r - 0.2 for r in rounds], baseline_df['train_time_sec'],
                width=0.4, alpha=0.3, color='#2E86AB')

    if hardware_df is not None and 'train_time_sec' in hardware_df.columns:
        rounds = hardware_df.get('round', range(len(hardware_df)))
        ax1.plot(rounds, hardware_df['train_time_sec'],
                 marker='s', label='Hardware-Aware', linewidth=2, markersize=8, color='#A23B72')
        ax1.bar([r + 0.2 for r in rounds], hardware_df['train_time_sec'],
                width=0.4, alpha=0.3, color='#A23B72')

    ax1.set_xlabel('Federated Round', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Training Latency per Round', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative training time
    ax2 = axes[1]
    if baseline_df is not None and 'train_time_sec' in baseline_df.columns:
        rounds = baseline_df.get('round', range(len(baseline_df)))
        cumulative = baseline_df['train_time_sec'].cumsum()
        ax2.plot(rounds, cumulative,
                 marker='o', label='Baseline (Cumulative)', linewidth=2, markersize=8, color='#2E86AB')

    if hardware_df is not None and 'train_time_sec' in hardware_df.columns:
        rounds = hardware_df.get('round', range(len(hardware_df)))
        cumulative = hardware_df['train_time_sec'].cumsum()
        ax2.plot(rounds, cumulative,
                 marker='s', label='Hardware-Aware (Cumulative)', linewidth=2, markersize=8, color='#A23B72')

    ax2.set_xlabel('Federated Round', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Training Time (seconds)',
                   fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Training Latency',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'latency_comparison.png'}")


def create_energy_comparison(baseline_df, hardware_df, output_dir):
    """Create energy consumption comparison graph."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Energy per round
    ax1 = axes[0]
    if baseline_df is not None and 'energy_joules' in baseline_df.columns:
        rounds = baseline_df.get('round', range(len(baseline_df)))
        ax1.bar([r - 0.2 for r in rounds], baseline_df['energy_joules'],
                width=0.4, label='Baseline', alpha=0.8, color='#2E86AB')

    if hardware_df is not None and 'energy_joules' in hardware_df.columns:
        rounds = hardware_df.get('round', range(len(hardware_df)))
        ax1.bar([r + 0.2 for r in rounds], hardware_df['energy_joules'],
                width=0.4, label='Hardware-Aware', alpha=0.8, color='#A23B72')

    ax1.set_xlabel('Federated Round', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Energy Consumption (Joules)',
                   fontsize=11, fontweight='bold')
    ax1.set_title('Energy Consumption per Round',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Total energy comparison
    ax2 = axes[1]
    experiments = []
    total_energy = []
    colors = []

    if baseline_df is not None and 'energy_joules' in baseline_df.columns:
        experiments.append('Baseline')
        total_energy.append(baseline_df['energy_joules'].sum())
        colors.append('#2E86AB')

    if hardware_df is not None and 'energy_joules' in hardware_df.columns:
        experiments.append('Hardware-Aware')
        total_energy.append(hardware_df['energy_joules'].sum())
        colors.append('#A23B72')

    if experiments:
        bars = ax2.bar(experiments, total_energy,
                       alpha=0.8, color=colors, width=0.6)
        ax2.set_ylabel('Total Energy (Joules)', fontsize=11, fontweight='bold')
        ax2.set_title('Total Energy Consumption',
                      fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, total_energy):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{value:.2f} J',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'energy_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'energy_comparison.png'}")


def create_performance_comparison(baseline_eval, hardware_eval, output_dir):
    """Create model performance comparison graph."""
    if baseline_eval is None and hardware_eval is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    experiments = []
    accuracies = []
    losses = []
    colors = []

    if baseline_eval:
        experiments.append('Baseline')
        accuracies.append(baseline_eval['accuracy'] * 100)
        losses.append(baseline_eval['loss'])
        colors.append('#2E86AB')

    if hardware_eval:
        experiments.append('Hardware-Aware')
        accuracies.append(hardware_eval['accuracy'] * 100)
        losses.append(hardware_eval['loss'])
        colors.append('#A23B72')

    # Accuracy comparison
    ax1 = axes[0]
    if experiments:
        bars = ax1.bar(experiments, accuracies,
                       alpha=0.8, color=colors, width=0.6)
        ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Model Accuracy Comparison',
                      fontsize=12, fontweight='bold')
        ax1.set_ylim([0, max(accuracies) * 1.1] if accuracies else [0, 100])
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'{value:.2f}%',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Loss comparison
    ax2 = axes[1]
    if experiments:
        bars = ax2.bar(experiments, losses, alpha=0.8, color=colors, width=0.6)
        ax2.set_ylabel('Test Loss', fontsize=11, fontweight='bold')
        ax2.set_title('Model Loss Comparison', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, losses):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{value:.4f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'performance_comparison.png'}")


def create_hardware_breakdown(hardware_df, output_dir):
    """Create per-hardware type breakdown for hardware-aware experiment."""
    if hardware_df is None or 'hardware_type' not in hardware_df.columns:
        return

    # Decode hardware_type: 0=gpu, 1=cpu-medium, 2=cpu-slow, 3=uniform
    hw_decode_map = {0: "gpu", 1: "cpu-medium", 2: "cpu-slow", 3: "uniform"}
    hw_df = hardware_df.copy()
    hw_df['hardware_type_str'] = hw_df['hardware_type'].map(
        hw_decode_map).fillna('unknown')

    # Filter out 'uniform' (baseline) if present
    hw_df = hw_df[hw_df['hardware_type_str'] != 'uniform'].copy()
    if len(hw_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Group by hardware type string
    hw_groups = hw_df.groupby('hardware_type_str')

    # Plot 1: Training time by hardware type
    ax1 = axes[0, 0]
    if 'train_time_sec' in hw_df.columns:
        hw_times = hw_groups['train_time_sec'].mean()
        colors_map = {'gpu': '#FF6B6B',
                      'cpu-medium': '#4ECDC4', 'cpu-slow': '#95E1D3'}
        colors = [colors_map.get(hw, '#95A5A6') for hw in hw_times.index]
        bars = ax1.bar(hw_times.index, hw_times.values,
                       alpha=0.8, color=colors, width=0.6)
        ax1.set_ylabel('Avg Training Time (seconds)',
                       fontsize=11, fontweight='bold')
        ax1.set_title('Training Time by Hardware Type',
                      fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars, hw_times.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'{value:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 2: Energy consumption by hardware type
    ax2 = axes[0, 1]
    if 'energy_joules' in hw_df.columns:
        hw_energy = hw_groups['energy_joules'].sum()
        colors = [colors_map.get(hw, '#95A5A6') for hw in hw_energy.index]
        bars = ax2.bar(hw_energy.index, hw_energy.values,
                       alpha=0.8, color=colors, width=0.6)
        ax2.set_ylabel('Total Energy (Joules)', fontsize=11, fontweight='bold')
        ax2.set_title('Energy Consumption by Hardware Type',
                      fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars, hw_energy.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{value:.2f}J', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 3: Batch sizes used by hardware type
    ax3 = axes[1, 0]
    if 'batch_size' in hw_df.columns:
        # Should be same for all in group
        hw_batch = hw_groups['batch_size'].first()
        colors = [colors_map.get(hw, '#95A5A6') for hw in hw_batch.index]
        bars = ax3.bar(hw_batch.index, hw_batch.values,
                       alpha=0.8, color=colors, width=0.6)
        ax3.set_ylabel('Batch Size', fontsize=11, fontweight='bold')
        ax3.set_title('Batch Size by Hardware Type',
                      fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars, hw_batch.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(value)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 4: Epochs used by hardware type
    ax4 = axes[1, 1]
    if 'epochs' in hw_df.columns:
        # Should be same for all in group
        hw_epochs = hw_groups['epochs'].first()
        colors = [colors_map.get(hw, '#95A5A6') for hw in hw_epochs.index]
        bars = ax4.bar(hw_epochs.index, hw_epochs.values,
                       alpha=0.8, color=colors, width=0.6)
        ax4.set_ylabel('Epochs', fontsize=11, fontweight='bold')
        ax4.set_title('Epochs by Hardware Type',
                      fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars, hw_epochs.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(value)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'hardware_breakdown.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'hardware_breakdown.png'}")


def create_summary_table(baseline_df, hardware_df, baseline_eval, hardware_eval, output_dir):
    """Create a summary comparison table."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data
    data = []
    headers = ['Metric', 'Baseline', 'Hardware-Aware', 'Difference']

    # Training metrics
    if baseline_df is not None and hardware_df is not None:
        baseline_avg_time = baseline_df['train_time_sec'].mean(
        ) if 'train_time_sec' in baseline_df.columns else None
        hardware_avg_time = hardware_df['train_time_sec'].mean(
        ) if 'train_time_sec' in hardware_df.columns else None

        if baseline_avg_time and hardware_avg_time:
            data.append(['Avg Training Time (s)', f'{baseline_avg_time:.2f}',
                        f'{hardware_avg_time:.2f}',
                        f'{hardware_avg_time - baseline_avg_time:+.2f}'])

        baseline_total_energy = baseline_df['energy_joules'].sum(
        ) if 'energy_joules' in baseline_df.columns else None
        hardware_total_energy = hardware_df['energy_joules'].sum(
        ) if 'energy_joules' in hardware_df.columns else None

        if baseline_total_energy and hardware_total_energy:
            data.append(['Total Energy (J)', f'{baseline_total_energy:.2f}',
                        f'{hardware_total_energy:.2f}',
                        f'{hardware_total_energy - baseline_total_energy:+.2f}'])

        baseline_avg_loss = baseline_df['train_loss'].mean(
        ) if 'train_loss' in baseline_df.columns else None
        hardware_avg_loss = hardware_df['train_loss'].mean(
        ) if 'train_loss' in hardware_df.columns else None

        if baseline_avg_loss and hardware_avg_loss:
            data.append(['Avg Training Loss', f'{baseline_avg_loss:.4f}',
                        f'{hardware_avg_loss:.4f}',
                        f'{hardware_avg_loss - baseline_avg_loss:+.4f}'])

    # Evaluation metrics
    if baseline_eval and hardware_eval:
        data.append(['Test Accuracy (%)', f'{baseline_eval["accuracy"]*100:.2f}',
                    f'{hardware_eval["accuracy"]*100:.2f}',
                    f'{(hardware_eval["accuracy"] - baseline_eval["accuracy"])*100:+.2f}'])
        data.append(['Test Loss', f'{baseline_eval["loss"]:.4f}',
                    f'{hardware_eval["loss"]:.4f}',
                    f'{hardware_eval["loss"] - baseline_eval["loss"]:+.4f}'])

    if data:
        table = ax.table(cellText=data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.3, 0.2, 0.2, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4A90E2')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style difference column
        for i in range(1, len(data) + 1):
            diff_val = data[i-1][3]
            if diff_val.startswith('+'):
                # Red for positive (worse)
                table[(i, 3)].set_facecolor('#FFE5E5')
            elif diff_val.startswith('-'):
                # Green for negative (better)
                table[(i, 3)].set_facecolor('#E5FFE5')

    plt.title('Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'summary_table.png'}")


def main():
    """Generate all comparison visualizations."""
    print("="*60)
    print("Generating Comparison Visualizations")
    print("="*60)

    # Create output directory
    output_dir = Path("results/comparison_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    print("\n📊 Loading metrics...")
    baseline_df = load_metrics("baseline")
    hardware_df = load_metrics("hardware-aware")

    if baseline_df is not None:
        print(f"   ✅ Loaded baseline metrics: {len(baseline_df)} rounds")
    else:
        print("   ⚠️  Baseline metrics not found")

    if hardware_df is not None:
        print(f"   ✅ Loaded hardware-aware metrics: {len(hardware_df)} rounds")
    else:
        print("   ⚠️  Hardware-aware metrics not found")

    # Evaluate models
    print("\n🔍 Evaluating models...")
    baseline_eval = evaluate_model(
        "models/final_model_baseline.pt", "baseline")
    hardware_eval = evaluate_model(
        "models/final_model_hardware-aware.pt", "hardware-aware")

    if baseline_eval:
        print(
            f"   ✅ Baseline: Loss={baseline_eval['loss']:.4f}, Acc={baseline_eval['accuracy']*100:.2f}%")
    else:
        print("   ⚠️  Could not evaluate baseline model")

    if hardware_eval:
        print(
            f"   ✅ Hardware-Aware: Loss={hardware_eval['loss']:.4f}, Acc={hardware_eval['accuracy']*100:.2f}%")
    else:
        print("   ⚠️  Could not evaluate hardware-aware model")

    # Generate visualizations
    print("\n📈 Generating visualizations...")

    create_loss_comparison(baseline_df, hardware_df, output_dir)
    create_latency_comparison(baseline_df, hardware_df, output_dir)
    create_energy_comparison(baseline_df, hardware_df, output_dir)
    create_performance_comparison(baseline_eval, hardware_eval, output_dir)
    # NEW: Per-hardware analysis
    create_hardware_breakdown(hardware_df, output_dir)
    create_summary_table(baseline_df, hardware_df,
                         baseline_eval, hardware_eval, output_dir)

    print("\n" + "="*60)
    print("✅ All visualizations generated!")
    print("="*60)
    print(f"\n📁 Plots saved in: {output_dir}")
    print("\nGenerated files:")
    print("  - loss_comparison.png")
    print("  - latency_comparison.png")
    print("  - energy_comparison.png")
    print("  - performance_comparison.png")
    print("  - summary_table.png")


if __name__ == "__main__":
    main()
