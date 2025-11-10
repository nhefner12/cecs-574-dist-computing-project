# Hardware-Aware Federated Learning with Flower

A federated learning project comparing baseline (uniform) and hardware-aware (adaptive) training strategies using Flower framework and PyTorch ResNet18 on CIFAR-10.

## Overview

This project demonstrates how hardware-aware federated learning can optimize training by adapting to heterogeneous client hardware (GPU, CPU-medium, CPU-slow), resulting in:
- **Reduced latency** through better resource utilization
- **Lower power consumption** via hardware-appropriate settings
- **Memory/GPU/CPU constraint management** with adaptive batch sizes and training parameters

### Key Features

- **Baseline Model**: Uniform settings for all clients (same learning rate, batch size, epochs)
- **Hardware-Aware Model**: Adaptive settings per hardware type:
  - GPU: Higher learning rate, larger batches, more epochs
  - CPU-medium: Balanced settings
  - CPU-slow: Lower learning rate, smaller batches, fewer epochs

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -e .
```

### 2. Choose Configuration Speed

Switch between fast, medium, or slow configurations:

```bash
# Fast: ~5-8 min per experiment, ~30-35% accuracy (for quick iteration)
python switch_config.py fast

# Medium: ~10-12 min per experiment, ~35-40% accuracy (recommended for development)
python switch_config.py medium

# Slow: ~15-20 min per experiment, ~40-50% accuracy (for final results)
python switch_config.py slow
```

### 3. Run Experiments

Run both baseline and hardware-aware experiments:

```bash
python run_comparison.py
```

This will:
1. Run the baseline experiment (uniform settings)
2. Run the hardware-aware experiment (adaptive settings)
3. Save models to `models/` directory
4. Save metrics to `results/` directory

### 4. Visualize Results

```bash
# Generate comparison plots
python visualize_comparison.py

# Or use the Jupyter notebook for detailed analysis
jupyter notebook comparison_analysis.ipynb
```

## Configuration Details

### Speed Configurations

| Speed | Rounds | Epochs | Data | Time/Exp | Accuracy | Use Case |
|-------|--------|--------|------|----------|----------|----------|
| **Fast** | 5 | 1 | 30% | ~5-8 min | ~30-35% | Quick iteration, debugging |
| **Medium** | 8 | 2 | 40% | ~10-12 min | ~35-40% | Development, testing |
| **Slow** | 10 | 2 | 50% | ~15-20 min | ~40-50% | Final results, publication |

### Hardware Profiles

The hardware-aware configuration uses different settings per hardware type:

| Hardware | Learning Rate | Batch Size | Epochs | Speed | Power | Latency |
|----------|---------------|------------|--------|-------|-------|---------|
| **GPU** | 0.002 | 32 | 2 | 2.5x | 35W | Low |
| **CPU-medium** | 0.001 | 16 | 1 | 1.0x | 10W | Medium |
| **CPU-slow** | 0.0005 | 8 | 1 | 0.5x | 12W | High |

## Project Structure

```
.
├── cecs_574_dist_computing_project/
│   ├── client_app.py          # Flower client with hardware-aware logic
│   ├── server_app.py          # Flower server and model saving
│   ├── task.py                # ResNet18 model, data loading, training
│   └── hardware_profiles.py   # Hardware profile definitions
├── models/                    # Generated model files (git-ignored)
├── results/                   # Metrics and visualizations
├── comparison_analysis.ipynb  # Jupyter notebook for analysis
├── run_comparison.py          # Run both experiments
├── switch_config.py           # Switch between speed configs
├── collect_metrics.py         # Collect metrics from models
├── visualize_comparison.py   # Generate comparison plots
├── compare_results.py         # Compare model performance
└── pyproject.toml             # Main configuration (updated by switch_config.py)
```

## Running Experiments

### Automated Comparison (Recommended)

```bash
# 1. Choose your speed configuration
python switch_config.py medium

# 2. Run both experiments
python run_comparison.py

# 3. Visualize results
python visualize_comparison.py
```

### Manual Execution

#### Run Baseline Only

```bash
# Switch to baseline config
cp pyproject.toml.baseline pyproject.toml

# Run experiment
flwr run .

# Model saved to: models/final_model_baseline-{speed}.pt
```

#### Run Hardware-Aware Only

```bash
# Switch to hardware-aware config
cp pyproject.toml.hardware-aware pyproject.toml

# Run experiment
flwr run .

# Model saved to: models/final_model_hardware-aware-{speed}.pt
```

## Understanding Results

### Metrics Collected

- **Training Loss**: Model loss per federated round
- **Training Time**: Time per round and cumulative time
- **Energy Consumption**: Energy (Joules) per round and total
- **Model Accuracy**: Test set accuracy
- **Hardware Breakdown**: Per-hardware-type metrics

### Visualizations Generated

1. **Loss Comparison**: Training loss over rounds
2. **Latency Comparison**: Training time per round and cumulative
3. **Energy Comparison**: Energy consumption per round and total
4. **Performance Comparison**: Accuracy and loss comparison
5. **Hardware Breakdown**: Per-hardware-type analysis

### Expected Differences

**Hardware-Aware Advantages:**
- ✅ Better GPU utilization (faster training on GPU clients)
- ✅ More efficient power usage (appropriate settings per hardware)
- ✅ Adaptive batch sizes (prevents OOM on slow hardware)
- ✅ Potentially faster convergence (adaptive learning rates)

**Baseline Advantages:**
- ✅ Simpler implementation
- ✅ More predictable behavior
- ✅ Uniform resource usage

## Configuration Files

### Main Config Files

- `pyproject.toml.baseline`: Baseline experiment config (uniform settings)
- `pyproject.toml.hardware-aware`: Hardware-aware experiment config (adaptive settings)

These are automatically updated by `switch_config.py` based on speed selection.

### Speed Variants

The switcher uses these template files:
- `pyproject.toml.baseline.{fast,medium,slow}`
- `pyproject.toml.hardware-aware.{fast,medium,slow}`

You typically don't need to edit these directly.

## Memory Optimization

The project includes several memory optimizations:

- **Reduced DataLoader workers**: `num_workers=0` (main process only)
- **Adaptive batch sizes**: Smaller batches for CPU-slow (8), medium (16), GPU (32)
- **Limited parallel clients**: `fraction-train=0.3` (3 clients in parallel)
- **Data subset**: Configurable subset ratio (30-50% based on speed)

If you encounter memory issues:
1. Use a faster configuration (reduces data usage)
2. Further reduce batch sizes in `hardware_profiles.py`
3. Reduce `fraction-train` in config files
4. Reduce `num-supernodes` in config files

## Troubleshooting

### Models Not Found

Models are saved to `models/` directory. If scripts can't find them:
- Check that `models/` directory exists
- Verify model files are named correctly (e.g., `final_model_baseline-fast.pt`)
- Ensure you've run the experiments first

### Metrics CSV Files Missing

If metrics CSV files are not generated:
- This is normal if your Flower version doesn't expose metrics via `result.metrics`
- Run `python collect_metrics.py` to generate metrics from trained models
- Models are still saved and can be evaluated manually

### Out of Memory Errors

1. Switch to a faster configuration:
   ```bash
   python switch_config.py fast
   ```

2. Reduce batch sizes in `hardware_profiles.py`

3. Reduce `fraction-train` in config files (fewer parallel clients)

4. Reduce `subset_ratio` in `task.py`

### Experiments Take Too Long

- Use `fast` configuration for quick iteration
- Reduce `num-server-rounds` in config files
- Reduce `num-supernodes` for faster simulation

## Research Workflow

### Phase 1: Fast Iteration (Development)
```bash
python switch_config.py fast
python run_comparison.py
# Quick feedback, verify code works
```

### Phase 2: Medium Testing (Comparison)
```bash
python switch_config.py medium
python run_comparison.py
python visualize_comparison.py
# Better results, still fast enough
```

### Phase 3: Final Results (Publication)
```bash
python switch_config.py slow
python run_comparison.py
python visualize_comparison.py
# Best accuracy, publication-ready
```

## Advanced Usage

### Custom Hardware Profiles

Edit `cecs_574_dist_computing_project/hardware_profiles.py` to modify:
- Learning rates per hardware type
- Batch sizes per hardware type
- Epochs per hardware type
- Speed multipliers
- Power consumption
- Network latency

### Custom Analysis

Use `comparison_analysis.ipynb` for:
- Detailed per-configuration analysis
- Custom visualizations
- Statistical comparisons
- Export to other formats

### Multiple Speed Configurations

The notebook automatically detects and compares all available speed configurations:
- `models/final_model_baseline-fast.pt`
- `models/final_model_baseline-medium.pt`
- `models/final_model_baseline-slow.pt`
- `models/final_model_hardware-aware-fast.pt`
- `models/final_model_hardware-aware-medium.pt`
- `models/final_model_hardware-aware-slow.pt`

## Key Research Contributions

This project demonstrates:

1. **Latency Reduction**: Hardware-aware training reduces overall system latency by better utilizing powerful hardware (GPUs)

2. **Power Consumption Optimization**: Adaptive power usage based on hardware capabilities, leading to better efficiency

3. **Memory/Resource Constraints**: Adaptive batch sizes prevent OOM errors on slower hardware while maximizing GPU utilization

4. **Different Local Training Rounds**: Hardware-specific epochs allow more training on capable hardware

5. **Data Processing Adaptation**: Different batch sizes and learning rates optimize for each hardware's capabilities

## Dependencies

- `flwr[simulation]>=1.23.0`: Flower framework for federated learning
- `flwr-datasets[vision]>=0.5.0`: Vision datasets for Flower
- `torch==2.7.1`: PyTorch
- `torchvision==0.22.1`: Torchvision
- `matplotlib`, `seaborn`, `pandas`: For visualization and analysis
- `jupyter`: For notebook analysis

## Resources

- [Flower Documentation](https://flower.ai/docs/)
- [Flower GitHub](https://github.com/adap/flower)
- [Flower Community](https://flower.ai/join-slack/)

## License

Apache-2.0
