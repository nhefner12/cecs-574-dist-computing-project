# cecs_574_dist_computing_project/hardware_profiles.py
# Hardware profiles define all hardware-specific parameters
# These are used by hardware-aware experiments
# Baseline experiments use CPU-medium profile settings uniformly

HARDWARE_PROFILES = {
    "gpu": {
        "speed": 1.0,        # 1.0x compute speed (fastest)
        "power": 35.0,       # 35W power consumption
        "latency": 0.01,     # 0.01s network latency
        "local_epochs": 3,   # 3 epochs (max for powerful hardware)
        "batch_size": 32,    # 32 batch size (larger for GPU)
        "lr": 0.002,         # 0.002 learning rate (higher for faster convergence)
    },
    "cpu-medium": {
        "speed": 0.7,        # 0.7x compute speed (70% of GPU)
        "power": 10.0,       # 10W power consumption
        "latency": 0.02,     # 0.02s network latency
        "local_epochs": 2,   # 2 epochs (balanced)
        "batch_size": 16,    # 16 batch size (medium)
        "lr": 0.001,         # 0.001 learning rate (standard)
    },
    "cpu-slow": {
        "speed": 0.4,        # 0.4x compute speed (40% of GPU, slowest)
        "power": 7.0,        # 7W power consumption (lowest)
        "latency": 0.08,     # 0.08s network latency (highest)
        "local_epochs": 1,   # 1 epoch (min to compensate for slowness)
        "batch_size": 8,     # 8 batch size (smallest to save memory)
        "lr": 0.0005,        # 0.0005 learning rate (lowest for stability)
    },
}


def get_profile(context):
    """Get hardware profile with all settings for the current node."""
    hw = context.node_config.get("hardware", "cpu-medium")
    return HARDWARE_PROFILES.get(hw, HARDWARE_PROFILES["cpu-medium"])


def get_hardware_type(context):
    """Get the hardware type string for the current node."""
    return context.node_config.get("hardware", "cpu-medium")
