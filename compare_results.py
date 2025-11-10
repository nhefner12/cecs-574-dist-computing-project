#!/usr/bin/env python3
"""
Compare results between baseline and hardware-aware experiments.

This script analyzes the differences in:
- Model performance (accuracy, loss)
- Training time
- Energy consumption
"""

import torch
import csv
from pathlib import Path
import pandas as pd
import numpy as np


def load_model_weights(model_path):
    """Load model weights from file."""
    if not Path(model_path).exists():
        return None
    return torch.load(model_path, map_location='cpu')


def load_metrics(metrics_path):
    """Load metrics from CSV file."""
    if not Path(metrics_path).exists():
        return None

    try:
        df = pd.read_csv(metrics_path)
        return df
    except Exception as e:
        print(f"Warning: Could not load metrics from {metrics_path}: {e}")
        return None


def evaluate_model(model_path, test_data_loader):
    """Evaluate a model on test data."""
    from cecs_574_dist_computing_project.task import Net, test

    if not Path(model_path).exists():
        return None

    model = Net()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use a sample partition for evaluation
    from cecs_574_dist_computing_project.task import load_data
    _, testloader = load_data(0, 10, batch_size=32)

    eval_loss, eval_acc = test(model, testloader, device)
    return {"loss": eval_loss, "accuracy": eval_acc}


def compare_metrics():
    """Compare metrics from both experiments."""
    print("="*60)
    print("Metrics Comparison")
    print("="*60)

    baseline_metrics = load_metrics("results/metrics_baseline.csv")
    hardware_metrics = load_metrics("results/metrics_hardware-aware.csv")

    if baseline_metrics is None and hardware_metrics is None:
        print("⚠️  No metrics CSV files found. Metrics may not be available in this Flower version.")
        print("   The models were still trained and saved.")
        return

    if baseline_metrics is not None:
        print("\n📊 Baseline Metrics:")
        print(baseline_metrics.to_string(index=False))
        print(
            f"\nAverage Train Loss: {baseline_metrics['train_loss'].mean():.4f}")
        print(
            f"Average Train Time: {baseline_metrics['train_time_sec'].mean():.2f} sec")
        print(f"Total Energy: {baseline_metrics['energy_joules'].sum():.2f} J")

    if hardware_metrics is not None:
        print("\n📊 Hardware-Aware Metrics:")
        print(hardware_metrics.to_string(index=False))
        print(
            f"\nAverage Train Loss: {hardware_metrics['train_loss'].mean():.4f}")
        print(
            f"Average Train Time: {hardware_metrics['train_time_sec'].mean():.2f} sec")
        print(f"Total Energy: {hardware_metrics['energy_joules'].sum():.2f} J")

    if baseline_metrics is not None and hardware_metrics is not None:
        print("\n" + "="*60)
        print("Comparison Summary")
        print("="*60)

        loss_diff = hardware_metrics['train_loss'].mean(
        ) - baseline_metrics['train_loss'].mean()
        time_diff = hardware_metrics['train_time_sec'].mean(
        ) - baseline_metrics['train_time_sec'].mean()
        energy_diff = hardware_metrics['energy_joules'].sum(
        ) - baseline_metrics['energy_joules'].sum()

        print(
            f"Train Loss Difference:  {loss_diff:+.4f} (hardware-aware vs baseline)")
        print(f"Train Time Difference:  {time_diff:+.2f} sec")
        print(f"Total Energy Difference: {energy_diff:+.2f} J")

        if loss_diff < 0:
            print("✅ Hardware-aware achieved lower loss (better)")
        else:
            print("⚠️  Baseline achieved lower loss")

        if energy_diff < 0:
            print("✅ Hardware-aware used less energy")
        else:
            print("⚠️  Baseline used less energy")


def compare_models():
    """Compare final model weights."""
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)

    baseline_model = Path("models/final_model_baseline.pt")
    hardware_model = Path("models/final_model_hardware-aware.pt")

    if not baseline_model.exists():
        print("⚠️  Baseline model not found")
        return

    if not hardware_model.exists():
        print("⚠️  Hardware-aware model not found")
        return

    baseline_weights = load_model_weights(baseline_model)
    hardware_weights = load_model_weights(hardware_model)

    if baseline_weights is None or hardware_weights is None:
        print("⚠️  Could not load model weights")
        return

    # Calculate weight differences
    total_diff = 0.0
    num_params = 0

    for key in baseline_weights.keys():
        if key in hardware_weights:
            diff = torch.abs(baseline_weights[key] - hardware_weights[key])
            total_diff += diff.sum().item()
            num_params += diff.numel()

    avg_diff = total_diff / num_params if num_params > 0 else 0.0

    print(f"Models loaded successfully")
    print(f"Average weight difference: {avg_diff:.6f}")
    print(f"Total parameters compared: {num_params:,}")

    # Evaluate both models
    print("\nEvaluating models on test set...")
    try:
        from cecs_574_dist_computing_project.task import load_data, Net, test
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, testloader = load_data(0, 10, batch_size=32)

        # Evaluate baseline
        baseline_model = Net()
        baseline_model.load_state_dict(torch.load(
            "models/final_model_baseline.pt", map_location='cpu'))
        baseline_model.to(device)
        baseline_loss, baseline_acc = test(baseline_model, testloader, device)

        # Evaluate hardware-aware
        hardware_model = Net()
        hardware_model.load_state_dict(torch.load(
            "models/final_model_hardware-aware.pt", map_location='cpu'))
        hardware_model.to(device)
        hardware_loss, hardware_acc = test(hardware_model, testloader, device)

        print(f"\n📈 Model Performance:")
        print(f"Baseline:")
        print(f"  Test Loss:     {baseline_loss:.4f}")
        print(f"  Test Accuracy: {baseline_acc*100:.2f}%")
        print(f"\nHardware-Aware:")
        print(f"  Test Loss:     {hardware_loss:.4f}")
        print(f"  Test Accuracy: {hardware_acc*100:.2f}%")
        print(f"\nDifference:")
        print(f"  Loss:     {hardware_loss - baseline_loss:+.4f}")
        print(f"  Accuracy: {(hardware_acc - baseline_acc)*100:+.2f}%")

    except Exception as e:
        print(f"⚠️  Could not evaluate models: {e}")


def main():
    """Main comparison function."""
    print("="*60)
    print("Federated Learning Results Comparison")
    print("Baseline vs Hardware-Aware")
    print("="*60)

    compare_metrics()
    compare_models()

    print("\n" + "="*60)
    print("Comparison Complete")
    print("="*60)


if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("⚠️  pandas not installed. Installing...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd

    main()
