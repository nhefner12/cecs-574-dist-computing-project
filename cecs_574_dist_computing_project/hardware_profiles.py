# cecs_574_dist_computing_project/hardware_profiles.py

HARDWARE_PROFILES = {
    "gpu":        {"speed": 2.5, "power": 120, "latency": 0.05},
    "cpu-medium": {"speed": 1.0, "power": 60,  "latency": 0.15},
    "cpu-slow":   {"speed": 0.6, "power": 25,  "latency": 0.30},
}


def get_profile(context):
    cid = context.node_id  # Example: "0", "1", ...
    hw = context.node_config.get("hardware", "cpu-medium")
    return HARDWARE_PROFILES.get(hw, HARDWARE_PROFILES["cpu-medium"])
