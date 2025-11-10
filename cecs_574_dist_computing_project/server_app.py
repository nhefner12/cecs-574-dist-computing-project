"""cecs-574-dist-computing-project: Server side of Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

import csv
from pathlib import Path

# Import the ResNet18-based model from task.py
from cecs_574_dist_computing_project.task import Net


app = ServerApp()


def aggregate_fit_metrics(results):
    """
    Aggregate client-reported training metrics by averaging.
    `results` is a list of (client_id, metrics_dict).
    """
    if not results:
        return {}

    aggregated = {}
    metric_keys = results[0][1].keys()

    for key in metric_keys:
        values = [metrics[key] for _, metrics in results]
        aggregated[key] = sum(values) / len(values)

    return aggregated


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Run config values from pyproject.toml
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

    # Define strategy
    strategy = FedAvg(
        fraction_train=fraction_train,
    )

    # Run federated training
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": local_lr}),
        num_rounds=num_rounds,
    )

    # Save final model weights
    final_state = result.arrays.to_torch_state_dict()
    torch.save(final_state, "final_model.pt")
    print("✅ Training complete — model saved to final_model.pt")

    # Try to save metrics if available
    # Note: Metrics are collected from clients but may not be accessible via result.metrics
    # in all Flower versions. The metrics are still being sent from clients.
    try:
        if hasattr(result, 'metrics'):
            metrics_output = result.metrics
            fit_history = metrics_output.get("fit", {})

            output_file = Path("results/metrics.csv")
            output_file.parent.mkdir(exist_ok=True)

            with output_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "train_loss", "train_time_sec",
                                "energy_joules", "num-examples"])

                for rnd, metrics_dict in fit_history.items():
                    writer.writerow([
                        rnd,
                        metrics_dict.get("train_loss"),
                        metrics_dict.get("train_time_sec"),
                        metrics_dict.get("energy_joules"),
                        metrics_dict.get("num-examples"),
                    ])

            print(f"📊 Metrics saved to {output_file}")
        else:
            print("ℹ️  Metrics collection: Metrics are sent from clients but not aggregated in this Flower version.")
            print(
                "   Client metrics (train_loss, train_time_sec, energy_joules) are still being collected.")
    except Exception as e:
        print(f"⚠️  Could not save metrics: {e}")
        print("   Training completed successfully. Model saved to final_model.pt")
