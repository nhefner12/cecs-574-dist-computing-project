"""cecs-574-dist-computing-project: Client App with performance logging"""

import time
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from cecs_574_dist_computing_project.task import Net, load_data, train as train_fn, test as test_fn
from cecs_574_dist_computing_project.hardware_profiles import get_profile

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Hardware-aware params ---
    profile = get_profile(context)
    lr = msg.content["config"]["lr"]
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config.get("batch-size", 32)

    # Load data
    partition_id = int(context.node_id)
    num_partitions = context.run_config["num-supernodes"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size=batch_size)

    # Start timing
    start = time.time()

    # Train
    train_loss = train_fn(model, trainloader, epochs, lr * profile["speed"], device)

    # Apply compute delay to simulate slower hardware
    simulated_delay = profile["latency"] * len(trainloader)
    time.sleep(simulated_delay)

    # End timing + compute energy
    train_time = time.time() - start
    energy = train_time * profile["power"]  # Joules

    # Package model + metrics
    arrays = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "train_time_sec": train_time,
        "energy_joules": energy,
        "num-examples": len(trainloader.dataset),
    }
    content = RecordDict({"arrays": arrays, "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = int(context.node_id)
    num_partitions = context.run_config["num-supernodes"]
    _, valloader = load_data(partition_id, num_partitions, batch_size=32)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
