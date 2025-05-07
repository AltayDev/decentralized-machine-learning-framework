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

from model import get_model
from utils import serialize_model, deserialize_model, federated_averaging
import config

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

# Global state
nodes = {}  # Stores information about connected nodes
node_models = {}  # Stores model updates from nodes
global_model = None  # The global model being trained
training_round = 0  # Current training round
training_in_progress = False  # Flag indicating if training is in progress
lock = threading.RLock()  # Lock for thread safety

def initialize_global_model():
    """Initialize the global model."""
    global global_model
    logger.info("Initializing global model")
    global_model = get_model()
    return global_model

@app.route('/api/register', methods=['POST'])
def register_node():
    """Register a new node to the network."""
    data = request.json
    node_id = str(uuid.uuid4())
    
    with lock:
        nodes[node_id] = {
            "id": node_id,
            "ip": request.remote_addr,
            "last_heartbeat": time.time(),
            "system_info": data.get("system_info", {}),
            "status": "registered"
        }
    
    logger.info(f"New node registered: {node_id} from {request.remote_addr}")
    return jsonify({
        "node_id": node_id,
        "message": "Node registered successfully",
        "coordinator_status": get_coordinator_status()
    })

@app.route('/api/heartbeat', methods=['POST'])
def heartbeat():
    """Process heartbeat from a node."""
    data = request.json
    node_id = data.get("node_id")
    system_info = data.get("system_info", {})
    
    if not node_id:
        return jsonify({"error": "Missing node ID"}), 400
    
    with lock:
        if node_id not in nodes:
            logger.warning(f"Heartbeat from unknown node ID: {node_id}, re-registering...")
            # Auto-register if node_id not found
            new_node = {
                "id": node_id,
                "ip": request.remote_addr,
                "last_heartbeat": time.time(),
                "system_info": system_info,
                "status": "re-registered"
            }
            nodes[node_id] = new_node
            logger.info(f"Auto-registered node: {node_id} from {request.remote_addr}")
        else:
            # Update existing node
            nodes[node_id]["last_heartbeat"] = time.time()
            nodes[node_id]["system_info"] = system_info
    
    return jsonify({"message": "Heartbeat received", "status": get_coordinator_status()})

@app.route('/api/get_model', methods=['GET'])
def get_global_model():
    """Return the current global model to a node."""
    node_id = request.args.get("node_id")
    
    if not node_id:
        return jsonify({"error": "Missing node ID"}), 400
    
    with lock:
        if node_id not in nodes:
            logger.warning(f"Model request from unknown node ID: {node_id}, auto-registering...")
            # Auto-register if node_id not found
            new_node = {
                "id": node_id,
                "ip": request.remote_addr,
                "last_heartbeat": time.time(),
                "system_info": {},
                "status": "auto-registered"
            }
            nodes[node_id] = new_node
            logger.info(f"Auto-registered node: {node_id} from {request.remote_addr}")
        
    if global_model is None:
        initialize_global_model()
    
    # Serialize the model
    model_data = serialize_model(global_model)
    
    # Record that this node has the latest model
    with lock:
        nodes[node_id]["has_latest_model"] = True
        nodes[node_id]["model_version"] = training_round
    
    # Create a temporary file to send
    temp_file = os.path.join(config.TEMP_DIR, f"temp_model_{node_id}.pt")
    torch.save(global_model.state_dict(), temp_file)
    
    try:
        response = send_file(
            temp_file,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f"global_model_round_{training_round}.pt"
        )
        
        # Schedule the temp file for deletion
        @response.call_on_close
        def remove_temp_file():
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return response
    except Exception as e:
        logger.error(f"Error sending model file: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return jsonify({"error": f"Failed to send model: {str(e)}"}), 500

@app.route('/api/submit_update', methods=['POST'])
def submit_model_update():
    """Receive a model update from a node."""
    node_id = request.form.get("node_id")
    
    if not node_id:
        return jsonify({"error": "Missing node ID"}), 400
    
    with lock:
        if node_id not in nodes:
            logger.warning(f"Update from unknown node ID: {node_id}, auto-registering...")
            # Auto-register if node_id not found
            new_node = {
                "id": node_id,
                "ip": request.remote_addr,
                "last_heartbeat": time.time(),
                "system_info": {},
                "status": "auto-registered-from-update"
            }
            nodes[node_id] = new_node
            logger.info(f"Auto-registered node during update: {node_id} from {request.remote_addr}")
    
    # Get the model file
    if 'model' not in request.files:
        return jsonify({"error": "No model file provided"}), 400
        
    model_file = request.files['model']
    
    # Save the model temporarily
    temp_path = os.path.join(config.TEMP_DIR, f"temp_update_{node_id}.pt")
    model_file.save(temp_path)
    
    # Load the model parameters
    try:
        state_dict = torch.load(temp_path)
        
        with lock:
            # Store the model update
            node_models[node_id] = state_dict
            nodes[node_id]["last_update"] = time.time()
            nodes[node_id]["update_round"] = training_round
            
            logger.info(f"Received model update from node {node_id} for round {training_round}")
            
        # Remove temporary file
        os.remove(temp_path)
        
        # Check if we should aggregate models
        should_aggregate = check_if_should_aggregate()
        
        return jsonify({
            "message": "Model update received successfully",
            "aggregation_triggered": should_aggregate
        })
        
    except Exception as e:
        logger.error(f"Error processing model update: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": f"Failed to process model update: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the status of the coordinator and connected nodes."""
    return jsonify(get_coordinator_status())

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
                                         node_models.get("update_round", -1) == training_round)
        
        # Count active nodes
        current_time = time.time()
        active_nodes = sum(1 for node in nodes.values() 
                          if current_time - node["last_heartbeat"] < config.NODE_TIMEOUT)
    
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
            # Get the list of model state dictionaries
            model_states = list(node_models.values())
            
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
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize the global model
    initialize_global_model()
    
    # Start maintenance tasks
    start_maintenance_tasks()
    
    # Start the server
    logger.info(f"Starting coordinator server on {args.host}:{args.port}")
    run_simple(args.host, args.port, app, use_reloader=args.debug) 