#!/usr/bin/env python3
"""
Run comparison between baseline and hardware-aware federated learning.

This script runs both experiments and saves results for comparison.
"""

import subprocess
import sys
import shutil
from pathlib import Path
import time


def run_experiment(config_file, experiment_name, original_config_path=None):
    """Run a single experiment with the given config file."""
    print(f"\n{'='*60}")
    print(f"Running {experiment_name} experiment...")
    print(f"{'='*60}\n")

    original_config = Path("pyproject.toml")

    # Only copy if config_file is different from pyproject.toml
    if config_file.resolve() != original_config.resolve():
        # Copy experiment config to pyproject.toml
        shutil.copy(config_file, original_config)
        needs_restore = True
    else:
        # Config file is already pyproject.toml, no need to copy
        needs_restore = False

    try:
        # Run the experiment
        start_time = time.time()
        result = subprocess.run(
            ["flwr", "run", "."],
            capture_output=False,
            text=True
        )
        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(
                f"\n✅ {experiment_name} experiment completed in {elapsed_time:.2f} seconds")
            return True
        else:
            print(
                f"\n❌ {experiment_name} experiment failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n❌ Error running {experiment_name} experiment: {e}")
        return False
    finally:
        # Restore original config if we changed it
        if needs_restore and original_config_path and Path(original_config_path).exists():
            shutil.copy(original_config_path, original_config)


def main():
    """Run both baseline and hardware-aware experiments."""
    print("="*60)
    print("Federated Learning Comparison: Baseline vs Hardware-Aware")
    print("="*60)

    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save original pyproject.toml at the start
    original_config = Path("pyproject.toml")
    backup_config = Path("pyproject.toml.original_backup")

    if original_config.exists():
        shutil.copy(original_config, backup_config)
        print(f"📋 Backed up original config to {backup_config}")
    else:
        print("⚠️  Warning: pyproject.toml not found!")

    # Run baseline experiment
    baseline_config = Path("pyproject.toml.baseline")
    if not baseline_config.exists():
        print(f"❌ Error: {baseline_config} not found!")
        sys.exit(1)

    baseline_success = run_experiment(
        baseline_config, "Baseline", str(backup_config))

    # Run hardware-aware experiment
    hardware_config = Path("pyproject.toml.hardware-aware")
    if not hardware_config.exists():
        print(f"❌ Error: {hardware_config} not found!")
        sys.exit(1)

    hardware_aware_success = run_experiment(
        hardware_config, "Hardware-Aware", str(backup_config))

    # Restore original config at the end
    if backup_config.exists():
        shutil.copy(backup_config, original_config)
        backup_config.unlink()
        print(f"\n📋 Restored original pyproject.toml")

    # Summary
    print(f"\n{'='*60}")
    print("Experiment Summary")
    print(f"{'='*60}")
    print(
        f"Baseline:        {'✅ Success' if baseline_success else '❌ Failed'}")
    print(
        f"Hardware-Aware:  {'✅ Success' if hardware_aware_success else '❌ Failed'}")
    print(f"\nResults saved in:")
    print(f"  - models/final_model_baseline.pt")
    print(f"  - models/final_model_hardware-aware.pt")
    print(f"  - results/metrics_baseline.csv (if available)")
    print(f"  - results/metrics_hardware-aware.csv (if available)")
    print(f"\nRun 'python compare_results.py' to analyze the differences.")


if __name__ == "__main__":
    main()
