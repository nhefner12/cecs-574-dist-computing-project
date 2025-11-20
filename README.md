# Hardware-Aware Federated Learning with Flower

A federated learning project comparing baseline (uniform) and hardware-aware (adaptive) training strategies using Flower framework and PyTorch ResNet18 on CIFAR-10.

## Setup

```bash
# Install dependencies
pip install -e .
```

## Run Experiments

Run both baseline and hardware-aware experiments:

```bash
# Slow configuration (recommended for final results)
python run_comparison.py slow

# Medium configuration (for development/testing)
python run_comparison.py medium

# Fast configuration (for quick iteration)
python run_comparison.py fast
```

This will:
- Run the baseline experiment (uniform settings)
- Run the hardware-aware experiment (adaptive settings)
- Save models to `models/` directory
- Save metrics to `results/` directory

## Manual Execution

### Run Baseline Only

```bash
# Switch to baseline config
cp pyproject.toml.baseline.slow pyproject.toml

# Run experiment
flwr run .

# Model saved to: models/final_model_baseline-slow.pt
```

### Run Hardware-Aware Only

```bash
# Switch to hardware-aware config
cp pyproject.toml.hardware-aware.slow pyproject.toml

# Run experiment
flwr run .

# Model saved to: models/final_model_hardware-aware-slow.pt
```

## Run Comparison Notebook

```bash
# Option 1: Jupyter Notebook
jupyter notebook comparison_analysis.ipynb

# Option 2: JupyterLab
jupyter lab comparison_analysis.ipynb

# Option 3: VS Code
code comparison_analysis.ipynb
```

The notebook analyzes the slow experiment configuration and compares baseline vs hardware-aware results.
