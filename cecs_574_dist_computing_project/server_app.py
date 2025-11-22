"""cecs-574-dist-computing-project: Server side of Flower / PyTorch app."""

import torch
import random
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Dict, Any

# Import the ResNet18-based model from task.py
from cecs_574_dist_computing_project.task import Net


app = ServerApp()

# Global storage for per-client metrics per round
per_client_metrics_history = defaultdict(list)
# Global storage for aggregated metrics per round (from Flower's aggregation)
aggregated_metrics_history = {}
current_round = 0
# Global storage for client selection per round (client_id -> hardware_profile)
client_selection_history: Dict[int, List[Dict[str, Any]]] = {}
# Global storage for saved client selection (loaded from file)
saved_client_selection: Optional[Dict[int, List[Dict[str, Any]]]] = None


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
    
    def __init__(self, *args, client_selection_file: Optional[str] = None, grid: Optional[Grid] = None, node_hw_mapping: Optional[Dict[str, str]] = None, fraction_train: float = 0.1, num_supernodes: Optional[int] = None, node_config: Optional[Dict[str, Any]] = None, **kwargs):
        # Remove fit_metrics_aggregation_fn if present (not supported in this Flower version)
        kwargs.pop('fit_metrics_aggregation_fn', None)
        # Remove grid from kwargs if present (we don't pass it to parent)
        kwargs.pop('grid', None)
        super().__init__(*args, **kwargs)
        self._round_num = 0
        self._client_selection_file = client_selection_file
        self._fraction_train = fraction_train  # Store fraction_train to check if we should select all clients
        self._num_supernodes_config = num_supernodes  # Store num_supernodes from config
        self._node_config = node_config  # Store node configuration (from nodes_*.json file)
        self._saved_selection: Optional[Dict[int, List[Dict[str, Any]]]] = None
        self._saved_config_indices: Dict[int, List[int]] = {}  # Store saved config indices per round
        self._saved_node_ids: Dict[int, List[str]] = {}  # Store saved node_ids per round (simpler approach)
        self._saved_num_supernodes: Dict[int, int] = {}  # Store num_supernodes per round for config_index calculation
        self._selected_clients_per_round: Dict[int, List[str]] = {}  # Track selected clients per round
        self._node_hw_mapping: Dict[str, str] = node_hw_mapping or {}  # Store node->hardware mapping (config index -> hardware)
        self._selected_config_indices_per_round: Dict[int, List[int]] = {}  # Store selected config indices (0-9) per round, in order
        self._train_config: Optional[Any] = None  # Store train_config for use in configure_train (ConfigRecord type)
        
        # Load saved client selection if provided
        if client_selection_file and Path(client_selection_file).exists():
            try:
                with open(client_selection_file, 'r') as f:
                    data = json.load(f)
                    # Keep the full structure with client_id, hardware_profile, and config_index
                    self._saved_selection = {
                        int(round_num): clients
                        for round_num, clients in data.items()
                    }
                    print(f"✅ Loaded client selection from {client_selection_file}")
                    print(f"   Rounds available: {sorted(self._saved_selection.keys())}")
                    
                    # Extract node_ids for each round (simpler approach - just use node_ids)
                    self._saved_node_ids = {}
                    self._saved_num_supernodes = {}
                    for round_num, clients in self._saved_selection.items():
                        node_ids = []
                        for client in clients:
                            if 'client_id' in client:
                                node_ids.append(str(client['client_id']))
                        self._saved_node_ids[round_num] = node_ids
                        # Store num_supernodes as the count of saved clients
                        self._saved_num_supernodes[round_num] = len(node_ids)
                        print(f"   Round {round_num}: saved node_ids {node_ids} (num_supernodes: {len(node_ids)})")
                    
                    # Also extract config indices for backward compatibility
                    self._saved_config_indices = {}
                    for round_num, clients in self._saved_selection.items():
                        config_indices = []
                        for client in clients:
                            if 'config_index' in client:
                                config_indices.append(client['config_index'])
                        self._saved_config_indices[round_num] = config_indices
            except Exception as e:
                print(f"⚠️  Warning: Could not load client selection file: {e}")
                self._saved_selection = None
                self._saved_config_indices = {}
    
    def start(self, *args, **kwargs):
        """Override start to set random seed and prepare for client selection tracking."""
        global saved_client_selection
        
        # Store train_config if provided (for use in configure_train)
        if 'train_config' in kwargs:
            self._train_config = kwargs['train_config']
        elif len(args) >= 3:  # Check if train_config is in positional args
            # start() typically receives (grid, initial_arrays, train_config, num_rounds, ...)
            # train_config is usually the 3rd positional arg
            try:
                from flwr.app import ConfigRecord
                if isinstance(args[2], ConfigRecord):
                    self._train_config = args[2]
            except:
                pass
        
        # Set a fixed random seed for deterministic client selection as a fallback
        random.seed(42)
        try:
            import numpy as np
            np.random.seed(42)
        except ImportError:
            pass
        
        # Store saved selection globally so it can be accessed in configure_fit
        if self._saved_selection:
            saved_client_selection = self._saved_selection
            print(f"📋 Loaded client selection for {len(self._saved_selection)} rounds")
        else:
            saved_client_selection = None
        
        # Call parent start method - pass through all arguments as-is
        return super().start(*args, **kwargs)
    
    def configure_train(self, *args, **kwargs):
        """Override to capture which clients are selected for training (newer Flower API)."""
        # Call parent method first to get normal behavior
        configs = super().configure_train(*args, **kwargs)
        
        # Extract arguments for our tracking logic
        server_round = args[0] if args else kwargs.get('server_round', None)
        
        # Store round number for later use in aggregate_train
        if server_round:
            print(f"🔍 configure_train called for round {server_round}", flush=True)
        
        return configs
        
        # After parent sampling, capture which config indices were selected
        # We'll use the order of results to match them
        if server_round is not None and client_manager is not None:
            import sys
            print(f"🔍 configure_train called for round {server_round}", flush=True)
            sys.stdout.flush()
            
            try:
                # The configs dict keys are the selected client identifiers
                # We need to map these to config indices (0-9)
                selected_config_indices = []
                
                if isinstance(configs, dict) and len(configs) > 0:
                    print(f"🔍 Configs dict has {len(configs)} keys: {list(configs.keys())[:3]}... (showing first 3)", flush=True)
                    print(f"🔍 Config key types: {[type(k).__name__ for k in list(configs.keys())[:3]]}", flush=True)
                    
                    # Try to get all clients to create a mapping
                    try:
                        all_clients = list(client_manager.all())
                        num_supernodes = len(all_clients)
                        print(f"🔍 Got {num_supernodes} clients from client_manager", flush=True)
                        
                        # Create mapping: client object -> config index (0 to num_supernodes-1)
                        # Use the same calculation as the client: config_idx = int(node_id) % num_supernodes
                        client_to_index = {}
                        node_id_to_index = {}
                        for client in all_clients:
                            config_idx = int(client.node_id) % num_supernodes
                            client_to_index[client] = config_idx
                            node_id_to_index[str(client.node_id)] = config_idx
                        
                        # Map each config key (client) to its config index
                        for client_key in configs.keys():
                            # The key might be a client object or node_id
                            if client_key in client_to_index:
                                selected_config_indices.append(client_to_index[client_key])
                                print(f"   Found client in mapping: config index {client_to_index[client_key]}", flush=True)
                            elif str(client_key) in node_id_to_index:
                                selected_config_indices.append(node_id_to_index[str(client_key)])
                                print(f"   Found node_id in mapping: {client_key} -> config index {node_id_to_index[str(client_key)]}", flush=True)
                            else:
                                # Try to find by node_id and calculate config index
                                found = False
                                for client in all_clients:
                                    if str(client.node_id) == str(client_key) or client == client_key:
                                        config_idx = int(client.node_id) % num_supernodes
                                        selected_config_indices.append(config_idx)
                                        print(f"   Found client by comparison: config index {config_idx}", flush=True)
                                        found = True
                                        break
                                if not found:
                                    print(f"   ⚠️  Could not map config key {client_key} (type: {type(client_key).__name__}) to any client", flush=True)
                        
                        # If we still don't have indices, use the order (assume first N clients)
                        if not selected_config_indices and len(configs) <= len(all_clients):
                            selected_config_indices = list(range(len(configs)))
                            print(f"   Using order fallback: first {len(configs)} clients", flush=True)
                        
                        self._selected_config_indices_per_round[server_round] = selected_config_indices
                        print(f"🔍 Stored selected config indices for round {server_round}: {selected_config_indices}", flush=True)
                        print(f"   Hardware: {[self._node_hw_mapping.get(str(idx), 'unknown') for idx in selected_config_indices]}", flush=True)
                    except Exception as e:
                        print(f"⚠️  Could not map clients to config indices: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        # Fallback: We can't determine which clients were selected, so we'll track by result order
                        # But we need to know which indices were selected. Since we can't get that, we'll
                        # use a placeholder that aggregate_train can handle
                        if isinstance(configs, dict):
                            num_selected = len(configs)
                            # Store a placeholder - aggregate_train will need to handle this differently
                            self._selected_config_indices_per_round[server_round] = []
                            print(f"⚠️  Cannot determine selected indices, will use result order in aggregate_train", flush=True)
            except Exception as e:
                print(f"⚠️  Error in configure_train tracking: {e}", flush=True)
                import traceback
                traceback.print_exc()
        
        return configs
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Override to capture which clients are selected for training."""
        global client_selection_history
        import sys
        
        print(f"🔍 configure_fit/configure_train called for round {server_round}", flush=True)
        sys.stdout.flush()
        
        # Get the configuration from parent (this will trigger client sampling)
        configs = super().configure_fit(server_round, parameters, client_manager)
        
        # After parent sampling, try to get which clients were actually selected
        # Note: Flower's client_manager.sample() is called internally, so we need to
        # track this differently. We'll capture it in aggregate_fit from the results.
        try:
            # Get all available clients for reference
            all_clients = client_manager.all()
            print(f"🔍 Available clients: {[str(c.node_id) for c in all_clients]}", flush=True)
            
            # Store mapping for later use in aggregate_fit
            # We can't get the exact selected clients here, but we'll use the results in aggregate_fit
        except Exception as e:
            print(f"⚠️  Could not capture client list in configure_fit: {e}", flush=True)
        
        return configs
    
    def aggregate_fit(self, server_round, results, failures):
        """Override to track round number and capture per-client metrics before aggregation."""
        try:
            global current_round, per_client_metrics_history, client_selection_history
            current_round = server_round - 1  # server_round is 1-indexed, we use 0-indexed
            
            # Force print to stdout immediately
            import sys
            print(f"🔍 aggregate_fit called for round {server_round}, received {len(results)} results", flush=True)
            sys.stdout.flush()
            
            # Decode hardware_type mapping (reverse of client_app.py encoding)
            hw_type_decode = {0: "gpu", 1: "cpu-medium", 2: "cpu-slow", 3: "uniform"}
            
            # Capture client selection for this round (client IDs and their hardware profiles)
            round_selection = []
            
            # Get client IDs that were selected in configure_fit (if available)
            selected_client_ids = self._selected_clients_per_round.get(server_round, [])
            print(f"🔍 Selected client IDs from configure_fit: {selected_client_ids}", flush=True)
            print(f"🔍 Node hardware mapping available: {len(self._node_hw_mapping)} nodes", flush=True)
        
            # Capture per-client metrics from raw results
            # Results can be FitRes objects, tuples, or other formats
            for idx, fit_res in enumerate(results):
                try:
                    # Debug: Print what we're working with
                    print(f"🔍 Processing result {idx+1}/{len(results)}: type={type(fit_res)}")
                    
                    # Handle tuple format: (num_examples, FitRes) or (client_id, metrics)
                    if isinstance(fit_res, tuple) and len(fit_res) >= 2:
                        print(f"   Result is a tuple: {fit_res[0]}, {type(fit_res[1])}")
                        # If second element is a FitRes, use it
                        if hasattr(fit_res[1], 'metrics'):
                            fit_res = fit_res[1]
                            # First element might be num_examples or client_id
                            potential_client_id = fit_res[0]
                            print(f"   Potential client_id from tuple: {potential_client_id}")
                        else:
                            # Tuple might be (client_id, metrics_dict)
                            potential_client_id = fit_res[0]
                            metrics_dict = fit_res[1] if isinstance(fit_res[1], dict) else {}
                            client_id = str(potential_client_id)
                            if metrics_dict:
                                hw_type_encoded = metrics_dict.get("hardware_type", 3)
                                try:
                                    hw_type_encoded = int(hw_type_encoded)
                                except (ValueError, TypeError):
                                    hw_type_encoded = 3
                                hw_type_str = hw_type_decode.get(hw_type_encoded, "unknown")
                                round_selection.append({
                                    'client_id': client_id,
                                    'hardware_profile': hw_type_str
                                })
                                per_client_metrics_history[current_round].append({
                                    'client_id': client_id,
                                    **metrics_dict
                                })
                            continue
                    
                    if hasattr(fit_res, '__dict__'):
                        print(f"   Attributes: {list(fit_res.__dict__.keys())}")
                    if hasattr(fit_res, '__slots__'):
                        print(f"   Slots: {fit_res.__slots__}")
                    
                    # Try multiple ways to access metrics and client_id
                    metrics_dict = None
                    client_id = None
                    
                    # Try to get client_id first - check multiple possible attributes
                    for attr_name in ['client_id', 'node_id', 'cid', 'client']:
                        if hasattr(fit_res, attr_name):
                            client_id = getattr(fit_res, attr_name)
                            print(f"   Found client_id via {attr_name}: {client_id}")
                            break
                    
                    # If still no client_id, try to get from node_id or use index
                    if client_id is None:
                        # Try node_id attribute
                        if hasattr(fit_res, 'node_id'):
                            client_id = fit_res.node_id
                        elif hasattr(fit_res, 'content') and hasattr(fit_res.content, 'get'):
                            # Try to get node_id from content
                            content = fit_res.content
                            if 'node_id' in content:
                                client_id = content['node_id']
                        elif selected_client_ids and idx < len(selected_client_ids):
                            # Use the client ID from configure_fit
                            client_id = selected_client_ids[idx]
                            print(f"   Using client_id from configure_fit: {client_id}")
                        else:
                            # Use index as fallback
                            client_id = f'node_{idx}'
                        print(f"   Final client_id: {client_id}")
                    
                    # If we have node hardware mapping and no hardware_type in metrics, use mapping
                    if self._node_hw_mapping and client_id:
                        mapped_hw = self._node_hw_mapping.get(str(client_id))
                        if mapped_hw:
                            print(f"   Found hardware from node mapping: {mapped_hw}")
                    
                    # Method 1: Direct metrics attribute
                    if hasattr(fit_res, 'metrics') and fit_res.metrics:
                        metrics_dict = fit_res.metrics
                        print(f"   Found metrics via .metrics attribute")
                    
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
                        # Extract node_id from metrics (client sends this)
                        node_id_from_metrics = None
                        if 'node_id' in metrics_dict:
                            try:
                                node_id_from_metrics = str(int(float(metrics_dict['node_id'])))
                                print(f"   Found node_id in metrics: {node_id_from_metrics}")
                            except (ValueError, TypeError):
                                pass
                        
                        # Use node_id from metrics as the authoritative client_id (this is the actual participating node)
                        if node_id_from_metrics:
                            client_id = node_id_from_metrics
                            print(f"   Using node_id from metrics as client_id: {client_id}")
                        
                        per_client_metrics_history[current_round].append({
                            'client_id': client_id,
                            **metrics_dict
                        })
                        
                        # Capture hardware profile for client selection tracking
                        # ALWAYS use node hardware mapping from config (actual hardware), not client-reported "uniform"
                        # This ensures baseline experiments show actual hardware profiles even though clients use uniform settings
                        hw_type_str = None
                        
                        # First priority: Get from node hardware mapping (actual hardware from config)
                        # Use the node_id from metrics if available, otherwise use client_id
                        node_id_for_mapping = node_id_from_metrics if node_id_from_metrics else str(client_id)
                        if self._node_hw_mapping and node_id_for_mapping:
                            mapped_hw = self._node_hw_mapping.get(node_id_for_mapping)
                            if mapped_hw:
                                hw_type_str = mapped_hw
                                print(f"   Using actual hardware from config: {hw_type_str} for node {node_id_for_mapping}")
                        
                        # Fallback: If no mapping available, try from metrics (but this should rarely happen)
                        if hw_type_str is None:
                            hw_type_encoded = metrics_dict.get("hardware_type", None)
                            if hw_type_encoded is not None:
                                try:
                                    hw_type_encoded = int(hw_type_encoded)
                                    hw_type_str = hw_type_decode.get(hw_type_encoded, "unknown")
                                except (ValueError, TypeError):
                                    hw_type_str = "unknown"
                            else:
                                hw_type_str = "unknown"
                        
                        # Only add to round_selection if we have a valid hardware profile
                        if hw_type_str and hw_type_str != "unknown":
                            round_selection.append({
                                'client_id': node_id_for_mapping,  # Use actual node ID
                                'hardware_profile': hw_type_str
                            })
                            print(f"   ✅ Added to round selection: node {node_id_for_mapping} ({hw_type_str})")
                        
                        if server_round == 1:
                            print(f"✅ Captured metrics for client {client_id}: {list(metrics_dict.keys())}")
                            print(f"   Hardware type: {hw_type_encoded} -> {hw_type_str}")
                except Exception as e:
                    # Skip problematic results but continue
                    print(f"⚠️  Warning: Could not extract metrics from FitRes: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Store client selection for this round
            # If round_selection is empty, try to get it from per_client_metrics_history
            if not round_selection and current_round in per_client_metrics_history:
                print(f"⚠️  Round {server_round}: round_selection was empty, trying to extract from per_client_metrics_history")
                for client_data in per_client_metrics_history[current_round]:
                    # Try to get node_id from metrics first (this is the actual participating node)
                    node_id = None
                    if 'node_id' in client_data:
                        try:
                            node_id = str(int(float(client_data['node_id'])))
                        except (ValueError, TypeError):
                            pass
                    
                    # Use node_id from metrics if available, otherwise use client_id
                    client_id = node_id if node_id else str(client_data.get('client_id', 'unknown'))
                    
                    # ALWAYS prioritize node hardware mapping (actual hardware) over client-reported "uniform"
                    hw_type_str = None
                    if self._node_hw_mapping and client_id:
                        hw_type_str = self._node_hw_mapping.get(client_id)
                    
                    # Fallback: If not in mapping, try from metrics
                    if hw_type_str is None:
                        hw_type_encoded = client_data.get("hardware_type", 3)
                        try:
                            hw_type_encoded = int(hw_type_encoded)
                            hw_type_str = hw_type_decode.get(hw_type_encoded, "unknown")
                        except (ValueError, TypeError):
                            hw_type_str = "unknown"
                    
                    # Only add if we have a valid hardware profile
                    if hw_type_str and hw_type_str != "unknown":
                        round_selection.append({
                            'client_id': client_id,  # Use actual node ID
                            'hardware_profile': hw_type_str
                        })
        
            if round_selection:
                client_selection_history[server_round] = round_selection
                print(f"📋 Round {server_round}: Selected {len(round_selection)} clients")
                for sel in round_selection:
                    print(f"   - Client {sel['client_id']}: {sel['hardware_profile']}")
            else:
                print(f"⚠️  Round {server_round}: No client selection captured!")
            
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
        except Exception as e:
            # If there's an error, log it but still call parent method
            import traceback
            print(f"❌ ERROR in aggregate_fit: {e}", flush=True)
            traceback.print_exc()
            # Still call parent to ensure training continues
            return super().aggregate_fit(server_round, results, failures)
    
    def aggregate_train(self, *args, **kwargs):
        """Override to track round number and capture per-client metrics (newer Flower API)."""
        # Extract arguments - parent may call with different signature
        # Try to extract server_round, results, and failures from args/kwargs
        server_round = None
        results = None
        failures = None
        
        # Try to extract from args (positional)
        if len(args) >= 1:
            # First arg might be server_round or results
            if isinstance(args[0], int):
                server_round = args[0]
            elif isinstance(args[0], (list, tuple)):
                results = args[0]
        
        if len(args) >= 2:
            if results is None and isinstance(args[1], (list, tuple)):
                results = args[1]
            elif server_round is None and isinstance(args[1], int):
                server_round = args[1]
        
        if len(args) >= 3:
            failures = args[2]
        
        # Try to extract from kwargs
        server_round = kwargs.get('server_round', server_round)
        results = kwargs.get('results', results)
        failures = kwargs.get('failures', failures)
        
        # If we still don't have results, try to get it from args[0] or args[1]
        if results is None and args:
            # The parent might be calling with just (results,) or (server_round, results)
            for arg in args:
                if isinstance(arg, (list, tuple)) and len(arg) > 0:
                    # Check if it looks like a list of results
                    if hasattr(arg[0], 'metrics') or (isinstance(arg[0], tuple) and len(arg[0]) >= 2):
                        results = arg
                        break
        
        # If we have results, do the tracking
        if results is not None:
            try:
                global current_round, per_client_metrics_history, client_selection_history
                
                # Try to determine server_round from context if not provided
                if server_round is None:
                    # Use the current round from global state or estimate from history
                    if client_selection_history:
                        server_round = max(client_selection_history.keys()) + 1
                    else:
                        server_round = 1
                
                current_round = server_round - 1
                
                import sys
                print(f"🔍 aggregate_train called for round {server_round}, received {len(results)} results", flush=True)
                sys.stdout.flush()
                
                # Do the same tracking logic as aggregate_fit
                hw_type_decode = {0: "gpu", 1: "cpu-medium", 2: "cpu-slow", 3: "uniform"}
                round_selection = []
                
                # Use node_config file to assign hardware profiles deterministically
                # This ensures both baseline and hardware-aware use the same hardware profile assignment
                if self._node_config:
                    desired_nodes = self._node_config.get('nodes', [])
                    print(f"🔍 Using node_config file with {len(desired_nodes)} nodes", flush=True)
                    
                    # Get all unique node_ids from results (sorted for deterministic order)
                    result_node_ids = []
                    for fit_res in results:
                        try:
                            metrics_dict = None
                            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                                metrics_dict = fit_res.metrics
                            elif hasattr(fit_res, 'content') and hasattr(fit_res.content, 'get'):
                                content = fit_res.content
                                if 'metrics' in content:
                                    metrics_dict = content['metrics']
                            
                            if metrics_dict:
                                if not isinstance(metrics_dict, dict):
                                    if hasattr(metrics_dict, 'to_dict'):
                                        metrics_dict = metrics_dict.to_dict()
                                    elif hasattr(metrics_dict, 'keys'):
                                        metrics_dict = {k: metrics_dict[k] for k in metrics_dict.keys()}
                                
                                if 'node_id' in metrics_dict:
                                    node_id = str(int(float(metrics_dict['node_id'])))
                                    if node_id not in result_node_ids:
                                        result_node_ids.append(node_id)
                        except Exception:
                            continue
                    
                    # Sort node_ids deterministically
                    result_node_ids = sorted(result_node_ids, key=lambda x: int(x))
                    
                    print(f"   Found {len(result_node_ids)} unique node_ids in results: {result_node_ids[:5]}...", flush=True)
                    
                    # Assign hardware profiles from node_config by position
                    # This ensures deterministic assignment: first client gets first hardware profile, etc.
                    for i, node_id in enumerate(result_node_ids):
                        if i < len(desired_nodes):
                            # Use hardware profile from node_config file (by position)
                            hw_type_str = desired_nodes[i].get('hardware_profile', 'unknown')
                            config_idx = desired_nodes[i].get('node_id', i)  # Use node_id from config as config_index
                        else:
                            # Fallback: calculate from node_id if we have more clients than nodes in config
                            num_supernodes = len(result_node_ids)
                            config_idx = int(node_id) % num_supernodes
                            hw_type_str = self._node_hw_mapping.get(str(config_idx), 'unknown')
                        
                        round_selection.append({
                            'client_id': node_id,
                            'hardware_profile': hw_type_str,
                            'config_index': config_idx
                        })
                        print(f"   ✅ Added node_id {node_id} (position {i}, config_index: {config_idx}, hw: {hw_type_str})", flush=True)
                
                # If we didn't build round_selection yet, use node_config to assign hardware profiles
                if not round_selection and self._node_config:
                    print(f"🔍 Using node_config to assign hardware profiles deterministically", flush=True)
                    desired_nodes = self._node_config.get('nodes', [])
                    
                    # Get all unique node_ids from results (sorted for deterministic order)
                    result_node_ids = []
                    for fit_res in results:
                        try:
                            metrics_dict = None
                            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                                metrics_dict = fit_res.metrics
                            elif hasattr(fit_res, 'content') and hasattr(fit_res.content, 'get'):
                                content = fit_res.content
                                if 'metrics' in content:
                                    metrics_dict = content['metrics']
                            
                            if metrics_dict:
                                if not isinstance(metrics_dict, dict):
                                    if hasattr(metrics_dict, 'to_dict'):
                                        metrics_dict = metrics_dict.to_dict()
                                    elif hasattr(metrics_dict, 'keys'):
                                        metrics_dict = {k: metrics_dict[k] for k in metrics_dict.keys()}
                                
                                if 'node_id' in metrics_dict:
                                    node_id = str(int(float(metrics_dict['node_id'])))
                                    if node_id not in result_node_ids:
                                        result_node_ids.append(node_id)
                        except Exception:
                            continue
                    
                    # Sort node_ids deterministically
                    result_node_ids = sorted(result_node_ids, key=lambda x: int(x))
                    
                    print(f"   Found {len(result_node_ids)} unique node_ids: {result_node_ids}", flush=True)
                    
                    # Assign hardware profiles from node_config by position
                    for i, node_id in enumerate(result_node_ids):
                        if i < len(desired_nodes):
                            hw_type_str = desired_nodes[i].get('hardware_profile', 'unknown')
                            config_idx = desired_nodes[i].get('node_id', i)
                        else:
                            # Fallback if more clients than nodes in config
                            num_supernodes = len(result_node_ids)
                            config_idx = int(node_id) % num_supernodes
                            hw_type_str = self._node_hw_mapping.get(str(config_idx), 'unknown')
                        
                        round_selection.append({
                            'client_id': node_id,
                            'hardware_profile': hw_type_str,
                            'config_index': config_idx
                        })
                        print(f"   ✅ Position {i}: node_id={node_id}, hw={hw_type_str}, config_idx={config_idx}", flush=True)
                
                # Final fallback: read from results if still no round_selection
                if not round_selection:
                    print(f"🔍 No node_config found, reading from client metrics...", flush=True)
                    for idx, fit_res in enumerate(results):
                        try:
                            # Debug: Print what we're working with
                            print(f"🔍 Processing result {idx+1}/{len(results)}: type={type(fit_res)}", flush=True)
                            
                            # Handle tuple format: (num_examples, FitRes) or (client_id, metrics)
                            if isinstance(fit_res, tuple) and len(fit_res) >= 2:
                                print(f"   Result is a tuple: {fit_res[0]}, {type(fit_res[1])}", flush=True)
                                # If second element is a FitRes, use it
                                if hasattr(fit_res[1], 'metrics'):
                                    fit_res = fit_res[1]
                                else:
                                    # Tuple might be (client_id, metrics_dict)
                                    potential_client_id = fit_res[0]
                                    metrics_dict = fit_res[1] if isinstance(fit_res[1], dict) else {}
                                    if metrics_dict and 'node_id' in metrics_dict:
                                        try:
                                            node_id = str(int(float(metrics_dict['node_id'])))
                                            hw_type_str = self._node_hw_mapping.get(node_id) if self._node_hw_mapping else None
                                            if hw_type_str:
                                                round_selection.append({
                                                    'client_id': node_id,
                                                    'hardware_profile': hw_type_str
                                                })
                                                print(f"   ✅ Tracked node {node_id} ({hw_type_str}) from tuple", flush=True)
                                        except (ValueError, TypeError) as e:
                                            print(f"   ⚠️  Could not extract node_id from tuple: {e}", flush=True)
                                    continue
                            
                            # Try multiple ways to access metrics and client_id
                            metrics_dict = None
                            client_id = None
                            
                            # Method 1: Direct metrics attribute
                            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                                metrics_dict = fit_res.metrics
                                print(f"   Found metrics via .metrics attribute", flush=True)
                            
                            # Method 2: Check if metrics are in content (newer Flower API)
                            elif hasattr(fit_res, 'content') and hasattr(fit_res.content, 'get'):
                                content = fit_res.content
                                if 'metrics' in content:
                                    metrics_dict = content['metrics']
                                    print(f"   Found metrics via .content['metrics']", flush=True)
                            
                            # Method 3: Check if it's a dict-like object
                            elif isinstance(fit_res, dict):
                                metrics_dict = fit_res.get('metrics', {})
                                print(f"   Found metrics via dict.get('metrics')", flush=True)
                            
                            if metrics_dict is not None:
                                # Handle MetricRecord conversion to dict
                                if not isinstance(metrics_dict, dict):
                                    if hasattr(metrics_dict, 'to_dict'):
                                        metrics_dict = metrics_dict.to_dict()
                                        print(f"   Converted MetricRecord to dict", flush=True)
                                    elif hasattr(metrics_dict, 'keys'):
                                        metrics_dict = {k: metrics_dict[k] for k in metrics_dict.keys()}
                                        print(f"   Converted dict-like to dict", flush=True)
                                    elif hasattr(metrics_dict, '__dict__'):
                                        metrics_dict = metrics_dict.__dict__
                                        print(f"   Converted to dict via __dict__", flush=True)
                                    else:
                                        metrics_dict = {}
                                        print(f"   ⚠️  Could not convert metrics_dict to dict", flush=True)
                                
                                print(f"   Metrics dict keys: {list(metrics_dict.keys()) if metrics_dict else 'EMPTY'}", flush=True)
                                
                                if metrics_dict:  # Only process if we have valid metrics
                                    # Get config_index directly from client metrics (client sends this now)
                                    config_index = None
                                    if 'config_index' in metrics_dict:
                                        try:
                                            config_index = int(float(metrics_dict['config_index']))
                                            print(f"   Found config_index in metrics: {config_index}", flush=True)
                                        except (ValueError, TypeError) as e:
                                            print(f"   ⚠️  Could not convert config_index: {e}", flush=True)
                                    
                                    # If we have config_index, use it to look up hardware
                                    if config_index is not None:
                                        hw_type_str = self._node_hw_mapping.get(str(config_index))
                                        
                                        if hw_type_str:
                                            # Get node_id from metrics for reference
                                            node_id_from_metrics = None
                                            if 'node_id' in metrics_dict:
                                                try:
                                                    node_id_from_metrics = str(int(float(metrics_dict['node_id'])))
                                                except (ValueError, TypeError):
                                                    pass
                                            
                                            round_selection.append({
                                                'client_id': node_id_from_metrics or f'node_{config_index}',
                                                'hardware_profile': hw_type_str,
                                                'config_index': config_index  # Save config_index for later use
                                            })
                                            print(f"   ✅ Tracked result {idx+1} -> config index {config_index} ({hw_type_str})", flush=True)
                                        else:
                                            print(f"   ⚠️  No hardware mapping for config index {config_index} in mapping {self._node_hw_mapping}", flush=True)
                                    else:
                                        print(f"   ⚠️  No config_index found in metrics_dict", flush=True)
                                else:
                                    print(f"   ⚠️  Could not extract metrics_dict from result", flush=True)
                        except Exception as e:
                            # Skip problematic results but continue
                            print(f"⚠️  Warning: Could not extract metrics from result {idx+1}: {e}", flush=True)
                            import traceback
                            traceback.print_exc()
                            continue
                
                # Deduplicate round_selection by node_id (keep first occurrence)
                # This ensures we don't have duplicate entries for the same client
                if round_selection:
                    seen_node_ids = set()
                    deduplicated_selection = []
                    for sel in round_selection:
                        node_id = sel.get('client_id')
                        if node_id and node_id not in seen_node_ids:
                            seen_node_ids.add(node_id)
                            deduplicated_selection.append(sel)
                        elif not node_id:
                            # If no node_id, include it (shouldn't happen, but be safe)
                            deduplicated_selection.append(sel)
                    
                    if len(deduplicated_selection) != len(round_selection):
                        print(f"⚠️  Deduplicated round_selection by node_id: {len(round_selection)} -> {len(deduplicated_selection)} entries", flush=True)
                    round_selection = deduplicated_selection
                
                if round_selection:
                    client_selection_history[server_round] = round_selection
                    print(f"📋 Round {server_round}: Selected {len(round_selection)} clients")
                    for sel in round_selection:
                        print(f"   - Client {sel['client_id']}: {sel['hardware_profile']} (config_index: {sel.get('config_index', 'N/A')})")
                    
                    # Verify against saved selection if available
                    saved_selection = self._saved_selection.get(server_round) if self._saved_selection else None
                    if saved_selection:
                        # Compare hardware profiles
                        current_hw = sorted([sel['hardware_profile'] for sel in round_selection])
                        saved_hw = sorted([sel.get('hardware_profile', 'unknown') for sel in saved_selection])
                        
                        if current_hw == saved_hw:
                            print(f"✅ Round {server_round}: Hardware profiles match saved selection: {current_hw}", flush=True)
                        else:
                            print(f"⚠️  Round {server_round}: Hardware profiles DO NOT match!", flush=True)
                            print(f"   Current: {current_hw}", flush=True)
                            print(f"   Saved:   {saved_hw}", flush=True)
                            print(f"   This may affect comparison fairness!", flush=True)
            except Exception as e:
                import traceback
                print(f"⚠️  Warning: Error in aggregate_train tracking: {e}", flush=True)
                traceback.print_exc()
        
        # Call parent's aggregate_train with all arguments
        try:
            if hasattr(super(), 'aggregate_train'):
                return super().aggregate_train(*args, **kwargs)
            else:
                # Fall back to aggregate_fit if aggregate_train doesn't exist
                if server_round is not None and results is not None:
                    return super().aggregate_fit(server_round, results, failures or [])
                else:
                    return super().aggregate_fit(*args, **kwargs)
        except Exception as e:
            import traceback
            print(f"❌ ERROR in aggregate_train: {e}", flush=True)
            traceback.print_exc()
            # Last resort: try to call with minimal args
            if results is not None:
                return super().aggregate_train(results) if hasattr(super(), 'aggregate_train') else super().aggregate_fit(server_round or 1, results, failures or [])
            raise


def get_node_hardware_mapping(context: Context) -> Dict[str, str]:
    """Extract node_id -> hardware_profile mapping from federation config."""
    hw_mapping = {}
    try:
        # Try to get node configs from context
        # In Flower, node configs are typically in the federation config
        federation_config = context.federation_config if hasattr(context, 'federation_config') else {}
        
        # Try to access node configs - format may vary
        # Common patterns: node_config, nodes, or in federation config
        if hasattr(context, 'node_configs'):
            node_configs = context.node_configs
            for node_id, config in node_configs.items():
                hw = config.get('hardware', 'unknown')
                hw_mapping[str(node_id)] = hw
        elif hasattr(context, 'federation_config'):
            # Try to get from federation config
            fed_config = context.federation_config
            if isinstance(fed_config, dict):
                # Look for node-config section
                for key, value in fed_config.items():
                    if 'node' in key.lower() or 'hardware' in key.lower():
                        if isinstance(value, dict):
                            for node_key, node_value in value.items():
                                if isinstance(node_value, dict) and 'hardware' in node_value:
                                    hw_mapping[str(node_key)] = node_value['hardware']
    except Exception as e:
        print(f"⚠️  Could not extract node hardware mapping: {e}")
    
    # Fallback: try to read from pyproject.toml directly using string parsing
    if not hw_mapping:
        try:
            with open("pyproject.toml", "r") as f:
                content = f.read()
                # Look for node-config section and extract hardware assignments
                import re
                # Pattern to match: "0.hardware = "gpu"" or similar
                pattern = r'(\d+)\.hardware\s*=\s*"([^"]+)"'
                matches = re.findall(pattern, content)
                for node_id, hw_type in matches:
                    hw_mapping[node_id] = hw_type
        except Exception as e:
            print(f"⚠️  Could not read hardware mapping from pyproject.toml: {e}")
    
    print(f"📋 Node hardware mapping: {hw_mapping}")
    return hw_mapping


def load_node_config(experiment_name: str) -> Optional[Dict[str, Any]]:
    """Load node configuration file for the experiment type."""
    # Extract speed suffix from experiment name (e.g., "baseline-fast" -> "fast")
    speed_suffix = ""
    if "-fast" in experiment_name or "-medium" in experiment_name or "-slow" in experiment_name:
        if "-fast" in experiment_name:
            speed_suffix = "fast"
        elif "-medium" in experiment_name:
            speed_suffix = "medium"
        elif "-slow" in experiment_name:
            speed_suffix = "slow"
    
    if not speed_suffix:
        # Try to detect from experiment name
        if "fast" in experiment_name.lower():
            speed_suffix = "fast"
        elif "medium" in experiment_name.lower():
            speed_suffix = "medium"
        elif "slow" in experiment_name.lower():
            speed_suffix = "slow"
    
    node_file = Path(f"nodes_{speed_suffix}.json")
    if node_file.exists():
        try:
            with open(node_file, 'r') as f:
                node_config = json.load(f)
            print(f"📋 Loaded node configuration from {node_file}")
            print(f"   Nodes: {node_config.get('num_nodes', 0)} nodes defined")
            return node_config
        except Exception as e:
            print(f"⚠️  Could not load node configuration from {node_file}: {e}")
            return None
    else:
        print(f"⚠️  Node configuration file not found: {node_file}")
        return None


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Run config values from pyproject.toml
    num_rounds: int = context.run_config["num-server-rounds"]
    # LR is now defined in hardware_profiles.py, but we use CPU-medium LR (0.001) as default
    # Clients will override this based on their hardware profile
    local_lr: float = context.run_config.get("lr", 0.001)  # Default to CPU-medium LR
    fraction_train: float = context.run_config["fraction-train"]
    num_supernodes_config: int = context.run_config.get("num-supernodes", 10)  # Get from config

    print("=== Server Configuration ===")
    print(f"Num Rounds:      {num_rounds}")
    print(f"Learning Rate:   {local_lr} (base, overridden by hardware profiles)")
    print(f"Fraction Train:  {fraction_train}")
    print(f"Num Supernodes:  {num_supernodes_config}")
    print("============================")
    
    # Get node hardware mapping from config
    node_hw_mapping = get_node_hardware_mapping(context)

    # Load node configuration file (both baseline and hardware-aware use the same file)
    experiment_name = context.run_config.get("experiment-name", "default")
    node_config = load_node_config(experiment_name)
    
    if not node_config:
        print("⚠️  WARNING: No node configuration file found. Client selection may not be deterministic.")
    
    # Initialize global model (ResNet18)
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Define strategy with per-client metric tracking
    # Note: We capture metrics in aggregate_fit override, so we don't need fit_metrics_aggregation_fn
    # Both baseline and hardware-aware use the same node_config file, ensuring identical client selection
    strategy = TrackingFedAvg(
        fraction_train=fraction_train,
        client_selection_file=None,  # No longer using saved client selection files
        grid=grid,
        node_hw_mapping=node_hw_mapping,
        num_supernodes=num_supernodes_config,
        node_config=node_config,  # Pass node configuration
    )

    # Run federated training
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": local_lr}),
        num_rounds=num_rounds,
    )

    # Get experiment name from config (for comparison) - already set above

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

        # Helper to normalize Flower's RecordDict/dict-like objects into a plain dict
        def normalize_round_metrics(attr_value):
            """Return a standard dict[int, dict[str, float]] from various Flower types."""
            if attr_value is None:
                return {}
            # RecordDict / dataclass with to_dict
            if hasattr(attr_value, "to_dict"):
                try:
                    attr_value = attr_value.to_dict()
                except Exception:
                    pass
            # dict-like with items()
            if hasattr(attr_value, "items"):
                try:
                    attr_value = dict(attr_value.items())
                except Exception:
                    pass
            # If still not a dict, attempt to cast directly
            if not isinstance(attr_value, dict):
                try:
                    attr_value = dict(attr_value)
                except Exception:
                    return {}
            return attr_value

        train_metrics_cache = None

        # Flower exposes client-side metrics in dedicated attributes (train_metrics_clientapp, evaluate_metrics_clientapp)
        # Use these to populate aggregated_metrics if available
        for attr_name, is_train in [
            ("train_metrics_clientapp", True),
            ("evaluate_metrics_clientapp", False),
        ]:
            attr_value = getattr(result, attr_name, None)
            if attr_value is None:
                continue

            round_metrics_dict = normalize_round_metrics(attr_value)
            if not round_metrics_dict:
                continue

            label = "train" if is_train else "eval"
            print(f"✅ Found {attr_name} for rounds: {sorted(round_metrics_dict.keys())}")

            # Build a temporary dict so we can replace the whole structure in one go
            normalized_rounds = {}
            for rnd, metrics_dict in round_metrics_dict.items():
                if not isinstance(metrics_dict, dict):
                    if hasattr(metrics_dict, "to_dict"):
                        try:
                            metrics_dict = metrics_dict.to_dict()
                        except Exception:
                            metrics_dict = {}
                    elif hasattr(metrics_dict, "items"):
                        try:
                            metrics_dict = dict(metrics_dict.items())
                        except Exception:
                            metrics_dict = {}
                    else:
                        metrics_dict = {}
                if not metrics_dict:
                    continue
                try:
                    round_num = int(rnd)
                except (ValueError, TypeError):
                    continue

                converted_metrics = {k: convert_metric_value(v) for k, v in metrics_dict.items()}
                normalized_rounds[round_num] = converted_metrics

            if not normalized_rounds:
                continue

            if is_train:
                train_metrics_cache = normalized_rounds
                # Replace aggregated_metrics with the authoritative train metrics.
                # We only want data from the train metrics source for the CSV.
                aggregated_metrics = normalized_rounds
            else:
                for round_num, converted_metrics in normalized_rounds.items():
                    aggregated_metrics.setdefault(round_num, converted_metrics)

        # Fallback: if for some reason aggregated_metrics is still empty but we captured train metrics
        if not aggregated_metrics and train_metrics_cache:
            aggregated_metrics = train_metrics_cache

        print(f"🔍 Debug: aggregated_metrics keys before saving: {sorted(aggregated_metrics.keys()) if aggregated_metrics else 'EMPTY'}")
        print(f"🔍 Debug: client_selection_history keys: {sorted(client_selection_history.keys()) if client_selection_history else 'EMPTY'}")
        
        # If client_selection_history is empty, try to reconstruct it from per-client metrics or node mapping
        if not client_selection_history and per_client_metrics_history:
            print(f"⚠️  client_selection_history is empty, trying to reconstruct from per_client_metrics_history")
            for round_num, clients in per_client_metrics_history.items():
                round_selection = []
                for client_data in clients:
                    client_id = str(client_data.get('client_id', 'unknown'))
                    # ALWAYS prioritize node hardware mapping (actual hardware) over client-reported "uniform"
                    # This ensures baseline experiments show actual hardware profiles
                    hw_type_str = node_hw_mapping.get(client_id)
                    
                    # Fallback: If not in mapping, try from metrics (but prefer mapping)
                    if hw_type_str is None:
                        hw_type_encoded = client_data.get("hardware_type", None)
                        if hw_type_encoded is not None:
                            try:
                                hw_type_encoded = int(hw_type_encoded)
                                hw_type_str = hw_type_decode.get(hw_type_encoded, "unknown")
                            except (ValueError, TypeError):
                                hw_type_str = "unknown"
                        else:
                            hw_type_str = "unknown"
                    
                    round_selection.append({
                        'client_id': client_id,
                        'hardware_profile': hw_type_str
                    })
                if round_selection:
                    client_selection_history[round_num + 1] = round_selection  # round_num is 0-indexed
                    print(f"✅ Reconstructed client selection for round {round_num + 1}: {len(round_selection)} clients")
        
        # If still empty and we have node mapping, this is a last resort fallback
        # We can't know which specific nodes participated, so we'll create a summary
        # But this should rarely happen if aggregate_fit is working correctly
        if not client_selection_history and node_hw_mapping and aggregated_metrics:
            print(f"⚠️  Still no client selection - this should not happen if aggregate_fit is working")
            print(f"   Creating summary from all nodes (this is a fallback and may not reflect actual participants)")
            # Count hardware types across all nodes
            hw_counts = {}
            for node_id, hw_type in node_hw_mapping.items():
                hw_counts[hw_type] = hw_counts.get(hw_type, 0) + 1
            
            # For each round, create a summary (but note this is all nodes, not just participants)
            for round_num in aggregated_metrics.keys():
                round_selection = []
                # Use actual node IDs from the mapping
                for node_id, hw_type in sorted(node_hw_mapping.items(), key=lambda x: int(x[0])):
                    round_selection.append({
                        'client_id': node_id,  # Use actual node ID from config
                        'hardware_profile': hw_type  # Use actual hardware from config
                    })
                if round_selection:
                    client_selection_history[round_num] = round_selection
                    print(f"⚠️  Created FALLBACK client selection for round {round_num}: ALL {len(round_selection)} nodes (not just participants!)")
                    print(f"   Hardware distribution: {hw_counts}")
                    print(f"   NOTE: This shows all nodes, not just the ones that participated in this round")

        # Save aggregated metrics
        if aggregated_metrics:
            output_file = results_dir / f"metrics_{experiment_name}.csv"
            
            with output_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "train_loss", "train_time_sec",
                                "energy_joules", "num-examples", "hardware_profiles"])

                for rnd in sorted(aggregated_metrics.keys()):
                    metrics_dict = aggregated_metrics[rnd]
                    
                    # Get hardware profile summary for this round
                    hw_profile_summary = ""
                    if client_selection_history and rnd in client_selection_history:
                        # Count hardware profiles
                        hw_counts = {}
                        for sel in client_selection_history[rnd]:
                            hw = sel.get('hardware_profile', 'unknown')
                            hw_counts[hw] = hw_counts.get(hw, 0) + 1
                        # Format as "2 gpu, 1 cpu-medium, 1 cpu-slow"
                        hw_profile_summary = ", ".join([f"{count} {hw}" for hw, count in sorted(hw_counts.items())])
                        print(f"🔍 Debug: Round {rnd} hardware profiles: {hw_profile_summary}")
                    else:
                        print(f"⚠️  Debug: No client selection data for round {rnd}")
                    
                    writer.writerow([
                        rnd,
                        metrics_dict.get("train_loss", 0.0),
                        metrics_dict.get("train_time_sec", 0.0),
                        metrics_dict.get("energy_joules", 0.0),
                        metrics_dict.get("num-examples", 0),
                        hw_profile_summary,
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
        
        # Save detailed hardware profile selection per round to CSV (for all experiments)
        if client_selection_history:
            hw_profiles_file = results_dir / f"hardware_profiles_{experiment_name}.csv"
            
            with hw_profiles_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "client_id", "hardware_profile"])
                
                for round_num in sorted(client_selection_history.keys()):
                    for sel in client_selection_history[round_num]:
                        writer.writerow([
                            round_num,
                            sel.get('client_id', 'unknown'),
                            sel.get('hardware_profile', 'unknown'),
                        ])
            
            print(f"📊 Hardware profiles per round saved to {hw_profiles_file}")
        
        # Save client selection history to JSON file (only for baseline experiments)
        # This will be loaded and used in hardware-aware experiments to ensure same client selection
        if client_selection_history and experiment_name.startswith("baseline"):
            # Extract speed suffix if present (e.g., "baseline-slow" -> "slow")
            speed_suffix = ""
            if experiment_name.startswith("baseline-"):
                speed_suffix = experiment_name.replace("baseline-", "")
            
            selection_file = results_dir / f"client_selection_baseline-{speed_suffix}.json" if speed_suffix else results_dir / "client_selection_baseline.json"
            
            # Convert to JSON-serializable format
            selection_data = {
                str(round_num): clients
                for round_num, clients in client_selection_history.items()
            }
            
            with selection_file.open("w") as f:
                json.dump(selection_data, f, indent=2)
            
            print(f"📋 Client selection saved to {selection_file}")
            print(f"   This will be used to ensure the same clients are selected in the hardware-aware experiment.")
        elif client_selection_history:
            # Log that we captured selection but didn't save (not a baseline experiment)
            print(f"ℹ️  Client selection captured for {len(client_selection_history)} rounds (not saved - not a baseline experiment)")
        else:
            print("ℹ️  No client selection history to save.")
            
    except Exception as e:
        print(f"⚠️  Could not save metrics: {e}")
        import traceback
        traceback.print_exc()
        print("   Training completed successfully. Model saved.")
