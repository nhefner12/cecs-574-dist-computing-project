# cecs_574_dist_computing_project/hardware_profiles.py

HARDWARE_PROFILES = {
    "gpu":        {"speed": 2.5, "power": 35.0, "latency": 0.01},
    "cpu-medium": {"speed": 1.0, "power": 10.0, "latency": 0.02},
    "cpu-slow":   {"speed": 0.5, "power": 12.0, "latency": 0.08},
}


def get_profile(context):
    cid = context.node_id  # Example: "0", "1", ...
    hw = context.node_config.get("hardware", "cpu-medium")
    return HARDWARE_PROFILES.get(hw, HARDWARE_PROFILES["cpu-medium"])
