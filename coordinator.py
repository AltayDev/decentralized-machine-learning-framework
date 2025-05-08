import os
import sys
import time
import uuid
import json
import logging
import threading
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.serving import run_simple

from model_registry import model_registry
from utils import serialize_model, deserialize_model, federated_averaging, get_system_info
import config

# Global state
current_model_name = "t5-small"  # Default model name
nodes = {}  # Stores information about connected nodes
node_models = {}  # Stores model updates from nodes
global_model = None  # The global model being trained
training_round = 0  # Current training round
training_in_progress = False  # Flag indicating if training is in progress
lock = threading.RLock()  # Lock for thread safety

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOGS_DIR, 'coordinator.log'))
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

class Coordinator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.global_model = None
        self.current_round = 0
        self._initialize_global_model()
    
    def _initialize_global_model(self):
        """Initialize the global model."""
        model_info = model_registry.get_model(self.model_name)
        self.global_model = model_info["model_class"](**model_info["config"])
        logger.info(f"Initialized global model: {self.model_name}")
    
    def register_node(self, node_id: str, system_info: Dict[str, Any]) -> bool:
        """Register a new node with the coordinator."""
        if node_id in self.nodes:
            logger.warning(f"Node {node_id} already registered")
            return False
        
        self.nodes[node_id] = {
            "system_info": system_info,
            "last_heartbeat": time.time(),
            "model_state": None
        }
        logger.info(f"Registered node: {node_id}")
        return True
    
    def update_node_heartbeat(self, node_id: str, system_info: Dict[str, Any]) -> bool:
        """Update node heartbeat and system info."""
        if node_id not in self.nodes:
            logger.warning(f"Node {node_id} not registered")
            return False
        
        self.nodes[node_id]["last_heartbeat"] = time.time()
        self.nodes[node_id]["system_info"] = system_info
        return True
    
    def update_node_model(self, node_id: str, model_state: Dict[str, torch.Tensor]) -> bool:
        """Update node's model state."""
        if node_id not in self.nodes:
            logger.warning(f"Node {node_id} not registered")
            return False
        
        self.nodes[node_id]["model_state"] = model_state
        return True
    
    def aggregate_models(self) -> bool:
        """Aggregate models from all nodes using federated averaging."""
        active_nodes = [
            node for node_id, node in self.nodes.items()
            if time.time() - node["last_heartbeat"] < config.NODE_TIMEOUT
            and node["model_state"] is not None
        ]
        
        if not active_nodes:
            logger.warning("No active nodes with model updates")
            return False
        
        model_states = [node["model_state"] for node in active_nodes]
        self.global_model.load_state_dict(federated_averaging(model_states))
        self.current_round += 1
        logger.info(f"Aggregated models from {len(active_nodes)} nodes in round {self.current_round}")
        return True

# Create global coordinator instance
coordinator = None

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    node_id = data.get("node_id")
    system_info = data.get("system_info")
    model_name = data.get("model_name")
    
    if not all([node_id, system_info, model_name]):
        return jsonify({"error": "Missing required fields"}), 400
    
    global coordinator
    if coordinator is None:
        coordinator = Coordinator(model_name)
    
    success = coordinator.register_node(node_id, system_info)
    if success:
        return jsonify({"node_id": node_id}), 200
    else:
        return jsonify({"error": "Registration failed"}), 400

@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    data = request.json
    node_id = data.get("node_id")
    system_info = data.get("system_info")
    model_name = data.get("model_name")
    
    if not all([node_id, system_info, model_name]):
        return jsonify({"error": "Missing required fields"}), 400
    
    global coordinator
    if coordinator is None:
        return jsonify({"error": "Coordinator not initialized"}), 500
    
    success = coordinator.update_node_heartbeat(node_id, system_info)
    if success:
        return jsonify({"status": "ok"}), 200
    else:
        return jsonify({"error": "Heartbeat update failed"}), 400

@app.route("/update_model", methods=["POST"])
def update_model():
    data = request.json
    node_id = data.get("node_id")
    model_state = data.get("model_state")
    
    if not all([node_id, model_state]):
        return jsonify({"error": "Missing required fields"}), 400
    
    global coordinator
    if coordinator is None:
        return jsonify({"error": "Coordinator not initialized"}), 500
    
    success = coordinator.update_node_model(node_id, model_state)
    if success:
        # Try to aggregate models
        coordinator.aggregate_models()
        return jsonify({"status": "ok"}), 200
    else:
        return jsonify({"error": "Model update failed"}), 400

@app.route("/get_model", methods=["GET"])
def get_model():
    global coordinator
    if coordinator is None:
        return jsonify({"error": "Coordinator not initialized"}), 500
    
    if coordinator.global_model is None:
        return jsonify({"error": "Global model not initialized"}), 500
    
    model_state = coordinator.global_model.state_dict()
    return jsonify({"model_state": model_state}), 200

def get_coordinator_status():
    """Get the current status of the coordination system."""
    active_nodes = 0
    gpu_nodes = 0
    cpu_nodes = 0
    
    # Count active nodes
    with lock:
        current_time = time.time()
        for node_id, node in nodes.items():
            if current_time - node["last_heartbeat"] < config.NODE_TIMEOUT:
                active_nodes += 1
                if node["system_info"].get("gpu_available", False):
                    gpu_nodes += 1
                else:
                    cpu_nodes += 1
    
    return {
        "training_round": training_round,
        "training_in_progress": training_in_progress,
        "active_nodes": active_nodes,
        "gpu_nodes": gpu_nodes,
        "cpu_nodes": cpu_nodes,
        "updates_received": len(node_models),
        "min_nodes_to_start": config.MIN_NODES_TO_START,
        "current_model": current_model_name,
        "nodes": nodes  # Include the full nodes dictionary
    }

def check_if_should_aggregate():
    """Check if we should aggregate models and start a new training round."""
    global training_round, training_in_progress, global_model, node_models
    
    # Don't aggregate if training is already in progress
    if training_in_progress:
        return False
    
    # Count how many nodes have submitted updates for the current round
    with lock:
        updates_for_current_round = sum(1 for node_id, node in nodes.items() 
                                      if node_id in node_models and 
                                         node_models[node_id]["model_name"] == current_model_name and
                                         node_models.get("update_round", -1) == training_round)
        
        # Count active nodes
        current_time = time.time()
        active_nodes = sum(1 for node in nodes.values() 
                          if current_time - node["last_heartbeat"] < config.NODE_TIMEOUT and
                             node["model_name"] == current_model_name)
    
    # If we have updates from all active nodes (or at least 75% of them and it's been a while)
    if (updates_for_current_round >= active_nodes or 
        (updates_for_current_round >= max(2, int(active_nodes * 0.75)) and 
         len(node_models) >= config.MIN_NODES_TO_START)):
        
        # Trigger aggregation in a separate thread
        threading.Thread(target=aggregate_models).start()
        return True
    
    return False

def aggregate_models():
    """Aggregate model updates from nodes using federated averaging."""
    global training_round, training_in_progress, global_model, node_models
    
    with lock:
        if training_in_progress or len(node_models) < 1:
            return
            
        training_in_progress = True
        logger.info(f"Starting model aggregation for round {training_round}")
        
        try:
            # Get the list of model state dictionaries for current model
            model_states = [
                node_models[node_id]["state_dict"]
                for node_id in node_models
                if node_models[node_id]["model_name"] == current_model_name
            ]
            
            if not model_states:
                logger.warning("No model states to aggregate")
                return
            
            # Perform federated averaging
            averaged_state = federated_averaging(model_states)
            
            # Update the global model
            global_model.load_state_dict(averaged_state)
            
            # Save the global model
            if config.SAVE_MODEL:
                # Save model specific to this round
                round_model_path = os.path.join(config.MODELS_DIR, f"global_model_round_{training_round}.pt")
                torch.save(global_model.state_dict(), round_model_path)
                
                # Save as the latest model
                torch.save(global_model.state_dict(), config.MODEL_PATH)
            
            # Clear node models for next round
            node_models = {}
            
            # Increment training round
            training_round += 1
            
            logger.info(f"Model aggregation complete. Starting round {training_round}")
            
        except Exception as e:
            logger.error(f"Error during model aggregation: {e}")
        
        finally:
            training_in_progress = False

def clean_inactive_nodes():
    """Remove nodes that haven't sent a heartbeat recently."""
    global nodes
    
    with lock:
        current_time = time.time()
        inactive_nodes = [node_id for node_id, node in nodes.items() 
                        if current_time - node["last_heartbeat"] > config.NODE_TIMEOUT]
        
        for node_id in inactive_nodes:
            logger.info(f"Removing inactive node: {node_id}")
            nodes.pop(node_id, None)
            node_models.pop(node_id, None)

def start_maintenance_tasks():
    """Start background maintenance tasks."""
    def maintenance_worker():
        while True:
            try:
                clean_inactive_nodes()
                check_if_should_aggregate()
            except Exception as e:
                logger.error(f"Error in maintenance task: {e}")
            
            time.sleep(10)  # Run maintenance every 10 seconds
    
    thread = threading.Thread(target=maintenance_worker, daemon=True)
    thread.start()
    logger.info("Maintenance tasks started")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Decentralized ML Coordinator")
    parser.add_argument("--host", type=str, default=config.COORDINATOR_HOST,
                        help=f"Host to bind to (default: {config.COORDINATOR_HOST})")
    parser.add_argument("--port", type=int, default=config.COORDINATOR_PORT,
                        help=f"Port to listen on (default: {config.COORDINATOR_PORT})")
    parser.add_argument("--model", type=str, default="t5-small",
                        help="Model name to use (default: t5-small)")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set current model name
    current_model_name = args.model
    
    # Initialize the global model
    coordinator = Coordinator(current_model_name)
    
    # Start maintenance tasks
    start_maintenance_tasks()
    
    # Start the server
    logger.info(f"Starting coordinator server on {args.host}:{args.port}")
    run_simple(args.host, args.port, app, use_reloader=args.debug) 