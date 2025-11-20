"""cecs-574-dist-computing-project: Client App with performance logging"""

import time
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from cecs_574_dist_computing_project.task import Net, load_data, train as train_fn, test as test_fn
from cecs_574_dist_computing_project.hardware_profiles import get_profile, get_hardware_type

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Hardware-aware params ---
    hardware_aware = context.run_config.get("hardware-aware", False)

    if hardware_aware:
        # Get hardware-specific profile with all settings
        profile = get_profile(context)
        hw_type = get_hardware_type(context)

        # Use hardware-specific settings from profile
        epochs = profile["local_epochs"]
        batch_size = profile["batch_size"]
        hw_lr = profile["lr"]

        # Use hardware-specific LR from profile
        # The server sends a base LR, but we override with hardware-specific LR
        lr = hw_lr
    else:
        # Baseline: uniform settings matching CPU-Medium profile for fair comparison
        # CPU-Medium speed, power, and latency
        profile = {"speed": 0.7, "power": 10.0, "latency": 0.02}  # CPU-Medium settings
        hw_type = "uniform"
        batch_size = context.run_config.get("batch-size", 16)  # CPU-Medium default
        epochs = context.run_config["local-epochs"]
        lr = msg.content["config"]["lr"]

    # Load data
    # Get num-supernodes from federation config or use default
    num_partitions = context.run_config.get("num-supernodes", 10)
    # Ensure partition_id is within valid range [0, num_partitions-1]
    partition_id = int(context.node_id) % num_partitions
    trainloader, _ = load_data(
        partition_id, num_partitions, batch_size=batch_size)

    # Start timing
    start = time.time()

    # Train with hardware-specific learning rate
    train_loss = train_fn(model, trainloader, epochs, lr, device)

    # Measure actual training time (before slowdown delays)
    actual_train_time = time.time() - start

    # Apply speed-based slowdown (for both baseline and hardware-aware)
    # GPU=1.0 means no delay, slower hardware gets delayed to simulate slower compute
    # Baseline uses CPU-Medium speed (0.7x), hardware-aware uses profile-specific speed
    speed_multiplier = profile["speed"]
    if speed_multiplier < 1.0:
        # Calculate additional sleep time to simulate slower hardware
        # If speed=0.5, hardware is 2x slower, so add 1x the training time as sleep
        # Formula: additional_time = actual_time * (1/speed - 1)
        # Example: speed=0.4 → (1/0.4 - 1) = (2.5 - 1) = 1.5x additional time
        # Example: speed=0.7 → (1/0.7 - 1) = (1.43 - 1) = 0.43x additional time
        slowdown_delay = actual_train_time * (1.0 / speed_multiplier - 1.0)
        time.sleep(slowdown_delay)
    
    # Apply network latency simulation (for both baseline and hardware-aware)
    # Baseline uses CPU-Medium latency (0.02s), hardware-aware uses profile-specific latency
    network_latency = profile["latency"] * len(trainloader) * 0.1
    time.sleep(network_latency)

    # End timing + compute energy
    # train_time now includes actual training + slowdown delays + network latency
    train_time = time.time() - start
    energy = train_time * profile["power"]  # Joules

    # Package model + metrics
    arrays = ArrayRecord(model.state_dict())

    # MetricRecord only accepts numeric types (int, float, list[int], list[float])
    # Encode hardware_type as numeric: 0=gpu, 1=cpu-medium, 2=cpu-slow, 3=uniform
    hw_type_map = {"gpu": 0, "cpu-medium": 1, "cpu-slow": 2, "uniform": 3}
    hw_type_encoded = hw_type_map.get(hw_type, 3)

    metrics = {
        "train_loss": train_loss,
        "train_time_sec": train_time,
        "energy_joules": energy,
        "num-examples": len(trainloader.dataset),
        "hardware_type": hw_type_encoded,  # Encoded as numeric for MetricRecord
        "batch_size": float(batch_size),   # Ensure numeric type
        "epochs": float(epochs),           # Ensure numeric type
        "learning_rate": float(lr),        # Ensure numeric type
    }
    content = RecordDict({"arrays": arrays, "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get num-supernodes from federation config or use default
    num_partitions = context.run_config.get("num-supernodes", 10)
    # Ensure partition_id is within valid range [0, num_partitions-1]
    partition_id = int(context.node_id) % num_partitions
    _, valloader = load_data(partition_id, num_partitions, batch_size=32)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
