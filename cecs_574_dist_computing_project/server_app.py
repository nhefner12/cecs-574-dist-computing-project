"""cecs-574-dist-computing-project: Server side of Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

import csv
from pathlib import Path

# Import the ResNet18-based model from task.py
from cecs_574_dist_computing_project.task import Net

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # These values come from pyproject.toml [tool.flwr.app.run]
    num_rounds: int = context.run_config["num-server-rounds"]
    local_lr: float = context.run_config["lr"]
    fraction_train: float = context.run_config["fraction-train"]

    print("=== Server Configuration ===")
    print(f"Num Rounds:      {num_rounds}")
    print(f"Learning Rate:   {local_lr}")
    print(f"Fraction Train:  {fraction_train}")
    print("============================")

    # Initialize global model (ResNet18)
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg
    strategy = FedAvg(fraction_train=fraction_train)

    # Execute federated training
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": local_lr}),
        num_rounds=num_rounds,
    )

    # Save trained model
    print("\n✅ Training complete — saving final model to disk...")
    final_state = result.arrays.to_torch_state_dict()
    torch.save(final_state, "final_model.pt")
    print("📦 Saved as final_model.pt")

    output_file = Path("results/metrics.csv")
    output_file.parent.mkdir(exist_ok=True)

    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "client_id", "train_time_sec", "energy_joules", "train_loss"])

        for rnd, train_metrics in result.train_metrics.items():
            for cid, metrics in train_metrics.items():
                writer.writerow([rnd, cid, metrics.get("train_time_sec"), metrics.get("energy_joules"), metrics.get("train_loss")])

    print(f"✅ Metrics saved to {output_file}")
