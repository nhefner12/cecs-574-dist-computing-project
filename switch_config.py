#!/usr/bin/env python3
"""
Switch between different experiment configurations (fast, medium, slow).

Usage:
    python switch_config.py fast    # Switch to fast configs
    python switch_config.py medium  # Switch to medium configs
    python switch_config.py slow    # Switch to slow configs
    python switch_config.py         # Show current config and options
"""

import sys
import shutil
from pathlib import Path

# Data subset ratios for each speed
DATA_RATIOS = {
    "fast": 0.3,
    "medium": 0.4,
    "slow": 0.5,
}

# Config file mappings
CONFIG_FILES = {
    "fast": {
        "baseline": "pyproject.toml.baseline.fast",
        "hardware-aware": "pyproject.toml.hardware-aware.fast",
    },
    "medium": {
        "baseline": "pyproject.toml.baseline.medium",
        "hardware-aware": "pyproject.toml.hardware-aware.medium",
    },
    "slow": {
        "baseline": "pyproject.toml.baseline.slow",
        "hardware-aware": "pyproject.toml.hardware-aware.slow",
    },
}


def update_data_ratio(speed):
    """Update subset_ratio in task.py."""
    task_file = Path("cecs_574_dist_computing_project/task.py")
    if not task_file.exists():
        print("⚠️  task.py not found")
        return False

    ratio = DATA_RATIOS[speed]

    try:
        content = task_file.read_text()

        # Find and replace the subset_ratio line
        import re
        pattern = r'subset_ratio\s*=\s*[\d.]+'
        replacement = f'subset_ratio = {ratio}'

        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            task_file.write_text(content)
            print(f"   ✅ Updated data subset ratio to {ratio*100}% in task.py")
            return True
        else:
            print("   ⚠️  Could not find subset_ratio in task.py")
            return False
    except Exception as e:
        print(f"   ❌ Error updating task.py: {e}")
        return False


def switch_config(speed):
    """Switch to the specified speed configuration."""
    if speed not in CONFIG_FILES:
        print(f"❌ Invalid speed: {speed}")
        print(f"   Valid options: {', '.join(CONFIG_FILES.keys())}")
        return False

    print(f"🔄 Switching to {speed.upper()} configuration...")

    # Update data ratio in task.py
    update_data_ratio(speed)

    # Copy baseline config
    baseline_src = Path(CONFIG_FILES[speed]["baseline"])
    baseline_dst = Path("pyproject.toml.baseline")

    if not baseline_src.exists():
        print(f"❌ Config file not found: {baseline_src}")
        return False

    shutil.copy(baseline_src, baseline_dst)
    print(f"   ✅ Updated {baseline_dst}")

    # Copy hardware-aware config
    hw_src = Path(CONFIG_FILES[speed]["hardware-aware"])
    hw_dst = Path("pyproject.toml.hardware-aware")

    if not hw_src.exists():
        print(f"❌ Config file not found: {hw_src}")
        return False

    shutil.copy(hw_src, hw_dst)
    print(f"   ✅ Updated {hw_dst}")

    # Show configuration summary
    print(f"\n📋 {speed.upper()} Configuration:")
    print(f"   Data subset: {DATA_RATIOS[speed]*100}%")

    # Read and show key settings (simple text parsing)
    try:
        with open(baseline_dst, 'r') as f:
            content = f.read()
            # Extract rounds
            import re
            rounds_match = re.search(r'num-server-rounds\s*=\s*(\d+)', content)
            epochs_match = re.search(r'local-epochs\s*=\s*(\d+)', content)
            if rounds_match:
                print(f"   Rounds: {rounds_match.group(1)}")
            if epochs_match:
                print(f"   Epochs: {epochs_match.group(1)}")
    except Exception as e:
        # If parsing fails, just show defaults
        defaults = {
            "fast": {"rounds": 5, "epochs": 1},
            "medium": {"rounds": 8, "epochs": 2},
            "slow": {"rounds": 10, "epochs": 2},
        }
        if speed in defaults:
            print(f"   Rounds: {defaults[speed]['rounds']}")
            print(f"   Epochs: {defaults[speed]['epochs']}")

    print(f"\n✅ Configuration switched to {speed.upper()}")
    print(f"   Estimated time: {get_estimated_time(speed)}")
    return True


def get_estimated_time(speed):
    """Get estimated time for the speed setting."""
    times = {
        "fast": "~5-8 minutes per experiment",
        "medium": "~10-12 minutes per experiment",
        "slow": "~15-20 minutes per experiment",
    }
    return times.get(speed, "unknown")


def show_current_config():
    """Show current configuration status."""
    print("="*60)
    print("Configuration Switcher")
    print("="*60)

    # Check which configs exist
    baseline = Path("pyproject.toml.baseline")
    hw = Path("pyproject.toml.hardware-aware")

    print("\n📁 Current config files:")
    print(f"   Baseline: {'✅' if baseline.exists() else '❌'} {baseline}")
    print(f"   Hardware-aware: {'✅' if hw.exists() else '❌'} {hw}")

    # Try to detect current speed
    try:
        if baseline.exists():
            with open(baseline, 'r') as f:
                content = f.read()
                import re
                # Detect speed from experiment name
                exp_match = re.search(
                    r'experiment-name\s*=\s*"([^"]+)"', content)
                rounds_match = re.search(
                    r'num-server-rounds\s*=\s*(\d+)', content)
                epochs_match = re.search(r'local-epochs\s*=\s*(\d+)', content)

                if exp_match:
                    exp_name = exp_match.group(1)
                    if 'fast' in exp_name:
                        current = "FAST"
                    elif 'medium' in exp_name:
                        current = "MEDIUM"
                    elif 'slow' in exp_name:
                        current = "SLOW"
                    else:
                        current = "UNKNOWN"

                    print(f"\n🔍 Detected configuration: {current}")
                    if rounds_match:
                        print(f"   Rounds: {rounds_match.group(1)}")
                    if epochs_match:
                        print(f"   Epochs: {epochs_match.group(1)}")
    except Exception as e:
        print(f"\n⚠️  Could not detect current configuration: {e}")

    print("\n📋 Available configurations:")
    print("\n   FAST:")
    print("     - 5 rounds, 1 epoch, 30% data")
    print("     - Time: ~5-8 min per experiment")
    print("     - Accuracy: ~30-35%")
    print("\n   MEDIUM:")
    print("     - 8 rounds, 2 epochs, 40% data")
    print("     - Time: ~10-12 min per experiment")
    print("     - Accuracy: ~35-40%")
    print("\n   SLOW:")
    print("     - 10 rounds, 2 epochs, 50% data")
    print("     - Time: ~15-20 min per experiment")
    print("     - Accuracy: ~40-50%")

    print("\n💡 Usage:")
    print("   python switch_config.py fast    # Switch to fast")
    print("   python switch_config.py medium  # Switch to medium")
    print("   python switch_config.py slow   # Switch to slow")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        show_current_config()
        return

    speed = sys.argv[1].lower()

    if speed in CONFIG_FILES:
        success = switch_config(speed)
        if success:
            print("\n🚀 Ready to run experiments!")
            print("   Run: python run_comparison.py")
        else:
            sys.exit(1)
    else:
        print(f"❌ Invalid speed: {speed}")
        print(f"   Valid options: {', '.join(CONFIG_FILES.keys())}")
        show_current_config()
        sys.exit(1)


if __name__ == "__main__":
    main()
