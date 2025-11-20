# cecs_574_dist_computing_project/hardware_profiles.py

HARDWARE_PROFILES = {
    "gpu": {
        "speed": 1.0,  # Baseline - no slowdown, runs at full speed
        "power": 35.0,
        "latency": 0.01,
        "local_epochs": 3,  # More epochs for powerful hardware
        "batch_size": 32,   # Larger batches for GPU
        "lr": 0.002,        # Higher LR for faster convergence
    },
    "cpu-medium": {
        "speed": 0.7,  # 70% of GPU speed - will be slowed via time.sleep
        "power": 10.0,
        "latency": 0.02,
        "local_epochs": 2,  # Balanced epochs
        "batch_size": 16,   # Medium batches
        "lr": 0.001,        # Standard LR
    },
    "cpu-slow": {
        "speed": 0.4,  # 40% of GPU speed - will be slowed more via time.sleep
        "power": 7.0,  # Low-power system (lower than CPU-Medium)
        "latency": 0.08,
        "local_epochs": 2,  # Same epochs but slower
        "batch_size": 8,    # Smaller batches to save memory
        "lr": 0.0005,       # Lower LR for stability
    },
}


def get_profile(context):
    """Get hardware profile with all settings for the current node."""
    hw = context.node_config.get("hardware", "cpu-medium")
    return HARDWARE_PROFILES.get(hw, HARDWARE_PROFILES["cpu-medium"])


def get_hardware_type(context):
    """Get the hardware type string for the current node."""
    return context.node_config.get("hardware", "cpu-medium")
