"""cecs-574-dist-computing-project: Server side of Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

import csv
from pathlib import Path
from collections import defaultdict

# Import the ResNet18-based model from task.py
from cecs_574_dist_computing_project.task import Net


app = ServerApp()

# Global storage for per-client metrics per round
per_client_metrics_history = defaultdict(list)
# Global storage for aggregated metrics per round (from Flower's aggregation)
aggregated_metrics_history = {}
current_round = 0


def aggregate_fit_metrics(results):
    """
    Aggregate client-reported training metrics by averaging.
    `results` is a list of tuples. Format may vary by Flower version:
    - (client_id, metrics_dict) 
    - (num_examples, metrics_dict)
    - FitRes objects
    
    Also stores per-client metrics for detailed analysis.
    """
    if not results:
        return {}

    # Store per-client metrics for current round
    global current_round, per_client_metrics_history
    
    # Handle different result formats
    processed_results = []
    for item in results:
        try:
            if hasattr(item, 'metrics'):  # FitRes object
                client_id = getattr(item, 'client_id', 'unknown')
                metrics = item.metrics
                # Convert MetricRecord to dict if needed
                if not isinstance(metrics, dict):
                    if hasattr(metrics, 'to_dict'):
                        metrics = metrics.to_dict()
                    elif hasattr(metrics, 'keys'):
                        metrics = {k: metrics[k] for k in metrics.keys()}
                    else:
                        metrics = {}
                processed_results.append((client_id, metrics))
                if metrics:  # Only store if we have valid metrics
                    per_client_metrics_history[current_round].append({
                        'client_id': client_id,
                        **metrics
                    })
            elif isinstance(item, tuple) and len(item) >= 2:
                # Tuple format: (client_id/num_examples, metrics_dict)
                identifier, metrics = item[0], item[1]
                # Try to use as client_id, fallback to string representation
                client_id = str(identifier) if not isinstance(identifier, (int, str)) else identifier
                if not isinstance(metrics, dict):
                    metrics = {}
                processed_results.append((client_id, metrics))
                if metrics:
                    per_client_metrics_history[current_round].append({
                        'client_id': client_id,
                        **metrics
                    })
            else:
                # Unknown format, skip per-client tracking but try to aggregate
                processed_results.append((f'client_{len(processed_results)}', item if isinstance(item, dict) else {}))
        except Exception as e:
            # Skip problematic items but continue processing
            print(f"⚠️  Warning: Could not process metrics item: {e}")
            continue

    if not processed_results:
        return {}

    aggregated = {}
    metric_keys = processed_results[0][1].keys() if processed_results[0][1] else []

    for key in metric_keys:
        values = [metrics.get(key, 0) for _, metrics in processed_results if key in metrics]
        if values:
            aggregated[key] = sum(values) / len(values)

    return aggregated


class TrackingFedAvg(FedAvg):
    """Custom FedAvg strategy that tracks round numbers and per-client metrics."""
    
    def __init__(self, *args, **kwargs):
        # Remove fit_metrics_aggregation_fn if present (not supported in this Flower version)
        kwargs.pop('fit_metrics_aggregation_fn', None)
        super().__init__(*args, **kwargs)
        self._round_num = 0
    
    def aggregate_fit(self, server_round, results, failures):
        """Override to track round number and capture per-client metrics before aggregation."""
        global current_round, per_client_metrics_history
        current_round = server_round - 1  # server_round is 1-indexed, we use 0-indexed
        
        # Capture per-client metrics from raw results
        # Results are typically FitRes objects with .metrics attribute
        for fit_res in results:
            try:
                # Debug: Print what we're working with
                if server_round == 1:  # Only print debug info for first round
                    print(f"🔍 Debug: FitRes type: {type(fit_res)}")
                    print(f"🔍 Debug: FitRes attributes: {dir(fit_res)}")
                    if hasattr(fit_res, 'metrics'):
                        print(f"🔍 Debug: fit_res.metrics type: {type(fit_res.metrics)}")
                        print(f"🔍 Debug: fit_res.metrics value: {fit_res.metrics}")
                
                # Try multiple ways to access metrics
                metrics_dict = None
                client_id = None
                
                # Method 1: Direct metrics attribute
                if hasattr(fit_res, 'metrics') and fit_res.metrics:
                    metrics_dict = fit_res.metrics
                    client_id = getattr(fit_res, 'client_id', f'client_{len(per_client_metrics_history[current_round])}')
                
                # Method 2: Check if metrics are in content (newer Flower API)
                elif hasattr(fit_res, 'content') and hasattr(fit_res.content, 'get'):
                    content = fit_res.content
                    if 'metrics' in content:
                        metrics_dict = content['metrics']
                        client_id = getattr(fit_res, 'client_id', f'client_{len(per_client_metrics_history[current_round])}')
                
                # Method 3: Check if it's a dict-like object
                elif isinstance(fit_res, dict):
                    metrics_dict = fit_res.get('metrics', {})
                    client_id = fit_res.get('client_id', f'client_{len(per_client_metrics_history[current_round])}')
                
                if metrics_dict is not None:
                    # Handle MetricRecord conversion to dict
                    if not isinstance(metrics_dict, dict):
                        if hasattr(metrics_dict, 'to_dict'):
                            metrics_dict = metrics_dict.to_dict()
                        elif hasattr(metrics_dict, 'keys'):
                            metrics_dict = {k: metrics_dict[k] for k in metrics_dict.keys()}
                        elif hasattr(metrics_dict, '__dict__'):
                            metrics_dict = metrics_dict.__dict__
                        else:
                            metrics_dict = {}
                    
                    if metrics_dict:  # Only append if we have valid metrics
                        per_client_metrics_history[current_round].append({
                            'client_id': client_id,
                            **metrics_dict
                        })
                        if server_round == 1:
                            print(f"✅ Captured metrics for client {client_id}: {list(metrics_dict.keys())}")
            except Exception as e:
                # Skip problematic results but continue
                print(f"⚠️  Warning: Could not extract metrics from FitRes: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Call parent to handle model aggregation and get aggregated metrics
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        # Also try to capture aggregated metrics from the result
        # The aggregated result might contain metrics that Flower computed
        if hasattr(aggregated_result, 'metrics') and aggregated_result.metrics:
            try:
                agg_metrics = aggregated_result.metrics
                # Convert MetricRecord to dict if needed
                if not isinstance(agg_metrics, dict):
                    if hasattr(agg_metrics, 'to_dict'):
                        agg_metrics = agg_metrics.to_dict()
                    elif hasattr(agg_metrics, 'keys'):
                        agg_metrics = {k: agg_metrics[k] for k in agg_metrics.keys()}
                
                if agg_metrics:
                    # Store aggregated metrics for this round
                    aggregated_metrics_history[server_round] = agg_metrics
                    if server_round == 1:
                        print(f"✅ Captured aggregated metrics for round {server_round}: {list(agg_metrics.keys())}")
            except Exception as e:
                if server_round == 1:
                    print(f"⚠️  Could not extract aggregated metrics: {e}")
        
        return aggregated_result


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

    # Define strategy with per-client metric tracking
    # Note: We capture metrics in aggregate_fit override, so we don't need fit_metrics_aggregation_fn
    strategy = TrackingFedAvg(
        fraction_train=fraction_train,
    )

    # Run federated training
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": local_lr}),
        num_rounds=num_rounds,
    )

    # Get experiment name from config (for comparison)
    experiment_name = context.run_config.get("experiment-name", "default")

    # Save final model weights
    final_state = result.arrays.to_torch_state_dict()
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_filename = models_dir / f"final_model_{experiment_name}.pt"
    torch.save(final_state, model_filename)
    print(f"✅ Training complete — model saved to {model_filename}")

    # Save metrics (both aggregated and per-client)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    try:
        # Decode hardware_type mapping (reverse of client_app.py encoding)
        hw_type_decode = {0: "gpu", 1: "cpu-medium", 2: "cpu-slow", 3: "uniform"}
        
        # Aggregate metrics from per_client_metrics_history or aggregated_metrics_history
        # This ensures we always have metrics even if result.metrics is empty
        aggregated_metrics = {}
        
        # First, try to use per-client metrics (more detailed)
        if per_client_metrics_history:
            for round_num, clients in per_client_metrics_history.items():
                if not clients:
                    continue
                    
                # Aggregate metrics across all clients in this round
                round_metrics = {
                    "train_loss": [],
                    "train_time_sec": [],
                    "energy_joules": [],
                    "num-examples": []
                }
                
                for client_data in clients:
                    if "train_loss" in client_data:
                        round_metrics["train_loss"].append(client_data["train_loss"])
                    if "train_time_sec" in client_data:
                        round_metrics["train_time_sec"].append(client_data["train_time_sec"])
                    if "energy_joules" in client_data:
                        round_metrics["energy_joules"].append(client_data["energy_joules"])
                    if "num-examples" in client_data:
                        round_metrics["num-examples"].append(client_data["num-examples"])
                
                # Average the metrics
                aggregated_metrics[round_num + 1] = {
                    "train_loss": sum(round_metrics["train_loss"]) / len(round_metrics["train_loss"]) if round_metrics["train_loss"] else 0.0,
                    "train_time_sec": sum(round_metrics["train_time_sec"]) / len(round_metrics["train_time_sec"]) if round_metrics["train_time_sec"] else 0.0,
                    "energy_joules": sum(round_metrics["energy_joules"]) / len(round_metrics["energy_joules"]) if round_metrics["energy_joules"] else 0.0,
                    "num-examples": sum(round_metrics["num-examples"]) if round_metrics["num-examples"] else 0,
                }
        
        # If we don't have per-client metrics, try aggregated metrics history
        if not aggregated_metrics and aggregated_metrics_history:
            print("ℹ️  Using aggregated metrics from round-by-round history")
            for round_num, metrics_dict in aggregated_metrics_history.items():
                if not isinstance(metrics_dict, dict):
                    if hasattr(metrics_dict, 'to_dict'):
                        metrics_dict = metrics_dict.to_dict()
                    elif hasattr(metrics_dict, 'keys'):
                        metrics_dict = {k: metrics_dict[k] for k in metrics_dict.keys()}
                    else:
                        continue
                aggregated_metrics[round_num] = metrics_dict
        
        # Also try to get metrics from result.metrics (if available)
        # Debug: Check what result contains
        print(f"🔍 Debug: result type: {type(result)}")
        print(f"🔍 Debug: result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
        if hasattr(result, 'metrics'):
            print(f"🔍 Debug: result.metrics type: {type(result.metrics)}")
            print(f"🔍 Debug: result.metrics value: {result.metrics}")
        
        # Check for other possible metric storage locations
        for attr_name in ['train_metrics', 'fit_metrics', 'client_metrics', 'aggregated_metrics']:
            if hasattr(result, attr_name):
                attr_value = getattr(result, attr_name)
                print(f"🔍 Debug: result.{attr_name} = {attr_value} (type: {type(attr_value)})")
        
        # Helper function to convert string values to floats
        def convert_metric_value(value):
            """Convert metric value from string (scientific notation) to float."""
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return value
            return value
        
        # Try multiple ways to access metrics from result
        if hasattr(result, 'metrics') and result.metrics:
            metrics_output = result.metrics
            
            # Handle different metric structures
            if isinstance(metrics_output, dict):
                # Check if this is a dict with round numbers as keys (Flower's structure)
                # Keys should be integers 1, 2, 3, etc. representing rounds
                round_keys = [k for k in metrics_output.keys() if isinstance(k, (int, str)) and str(k).isdigit()]
                
                if round_keys:
                    # This is Flower's per-round metrics structure: {1: {...}, 2: {...}, ...}
                    print(f"✅ Found per-round metrics in result.metrics for rounds: {sorted(round_keys)}")
                    for rnd_key, metrics_dict in metrics_output.items():
                        try:
                            round_num = int(rnd_key)
                            if not isinstance(metrics_dict, dict):
                                if hasattr(metrics_dict, 'to_dict'):
                                    metrics_dict = metrics_dict.to_dict()
                                elif hasattr(metrics_dict, 'keys'):
                                    metrics_dict = {k: metrics_dict[k] for k in metrics_dict.keys()}
                                else:
                                    continue
                            
                            # Convert all string values to floats
                            converted_metrics = {}
                            for key, value in metrics_dict.items():
                                converted_metrics[key] = convert_metric_value(value)
                            
                            # Only store if we don't already have this round from per-client metrics
                            if round_num not in aggregated_metrics:
                                aggregated_metrics[round_num] = converted_metrics
                                print(f"✅ Extracted metrics for round {round_num}: {list(converted_metrics.keys())}")
                        except (ValueError, TypeError):
                            continue
                else:
                    # Try 'fit' key (Flower's alternative structure)
                    fit_history = metrics_output.get("fit", {})
                    if fit_history:
                        for rnd, metrics_dict in fit_history.items():
                            try:
                                round_key = int(rnd) if isinstance(rnd, (int, str)) and str(rnd).isdigit() else int(rnd)
                            except (ValueError, TypeError):
                                continue
                            
                            if not isinstance(metrics_dict, dict):
                                if hasattr(metrics_dict, 'to_dict'):
                                    metrics_dict = metrics_dict.to_dict()
                                elif hasattr(metrics_dict, 'keys'):
                                    metrics_dict = {k: metrics_dict[k] for k in metrics_dict.keys()}
                                else:
                                    continue
                            
                            # Convert string values to floats
                            converted_metrics = {k: convert_metric_value(v) for k, v in metrics_dict.items()}
                            
                            if round_key not in aggregated_metrics:
                                aggregated_metrics[round_key] = converted_metrics
            elif hasattr(metrics_output, 'to_dict'):
                # MetricRecord object - convert to dict
                metrics_dict = metrics_output.to_dict()
                # Convert string values to floats
                converted_metrics = {k: convert_metric_value(v) for k, v in metrics_dict.items()}
                # If we don't have per-round metrics, use this for all rounds
                if not aggregated_metrics:
                    for rnd in range(1, num_rounds + 1):
                        aggregated_metrics[rnd] = converted_metrics
            elif hasattr(metrics_output, 'keys'):
                # Dict-like object (MetricRecord with keys method)
                metrics_dict = {k: metrics_output[k] for k in metrics_output.keys()}
                # Convert string values to floats
                converted_metrics = {k: convert_metric_value(v) for k, v in metrics_dict.items()}
                # Use for all rounds if we don't have per-round data
                if not aggregated_metrics:
                    for rnd in range(1, num_rounds + 1):
                        aggregated_metrics[rnd] = converted_metrics

        # Flower exposes client-side metrics in dedicated attributes (e.g., train_metrics_clientapp)
        # Use these as an additional fallback
        if not aggregated_metrics and hasattr(result, "train_metrics_clientapp"):
            clientapp_metrics = getattr(result, "train_metrics_clientapp")
            if isinstance(clientapp_metrics, dict):
                print(f"✅ Found train_metrics_clientapp for rounds: {sorted(clientapp_metrics.keys())}")
                for rnd, metrics_dict in clientapp_metrics.items():
                    if not isinstance(metrics_dict, dict):
                        continue
                    try:
                        round_num = int(rnd)
                    except (ValueError, TypeError):
                        continue
                    converted_metrics = {k: convert_metric_value(v) for k, v in metrics_dict.items()}
                    aggregated_metrics[round_num] = converted_metrics

        if not aggregated_metrics and hasattr(result, "evaluate_metrics_clientapp"):
            eval_metrics = getattr(result, "evaluate_metrics_clientapp")
            if isinstance(eval_metrics, dict):
                print(f"ℹ️  Found evaluate_metrics_clientapp for rounds: {sorted(eval_metrics.keys())}")
                # These are evaluation metrics; only store if nothing else is available
                for rnd, metrics_dict in eval_metrics.items():
                    if not isinstance(metrics_dict, dict):
                        continue
                    try:
                        round_num = int(rnd)
                    except (ValueError, TypeError):
                        continue
                    converted_metrics = {k: convert_metric_value(v) for k, v in metrics_dict.items()}
                    aggregated_metrics.setdefault(round_num, converted_metrics)

        # Save aggregated metrics
        if aggregated_metrics:
            output_file = results_dir / f"metrics_{experiment_name}.csv"
            
            with output_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "train_loss", "train_time_sec",
                                "energy_joules", "num-examples"])

                for rnd in sorted(aggregated_metrics.keys()):
                    metrics_dict = aggregated_metrics[rnd]
                    writer.writerow([
                        rnd,
                        metrics_dict.get("train_loss", 0.0),
                        metrics_dict.get("train_time_sec", 0.0),
                        metrics_dict.get("energy_joules", 0.0),
                        metrics_dict.get("num-examples", 0),
                    ])

            print(f"📊 Aggregated metrics saved to {output_file}")
        else:
            print("⚠️  No aggregated metrics to save. Check if clients are reporting metrics.")
        
        # Save detailed per-client metrics per round
        if per_client_metrics_history:
            detailed_file = results_dir / f"client_details_{experiment_name}.csv"
            
            with detailed_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "round", "client_id", "hardware_type", "batch_size", 
                    "epochs", "learning_rate", "train_loss", "train_time_sec",
                    "energy_joules", "num-examples"
                ])
                
                for round_num, clients in per_client_metrics_history.items():
                    for client_data in clients:
                        hw_type_encoded = client_data.get("hardware_type", 3)
                        hw_type_str = hw_type_decode.get(int(hw_type_encoded), "unknown")
                        
                        writer.writerow([
                            round_num + 1,  # Round numbers start at 1
                            client_data.get("client_id", "unknown"),
                            hw_type_str,
                            int(client_data.get("batch_size", 0)),
                            int(client_data.get("epochs", 0)),
                            client_data.get("learning_rate", 0.0),
                            client_data.get("train_loss", 0.0),
                            client_data.get("train_time_sec", 0.0),
                            client_data.get("energy_joules", 0.0),
                            int(client_data.get("num-examples", 0)),
                        ])
            
            print(f"📊 Per-client details saved to {detailed_file}")
            print(f"   This file shows which machines and profiles were selected per round.")
        else:
            print("ℹ️  No per-client metrics collected. This may be due to Flower version differences.")
            
    except Exception as e:
        print(f"⚠️  Could not save metrics: {e}")
        import traceback
        traceback.print_exc()
        print("   Training completed successfully. Model saved.")
