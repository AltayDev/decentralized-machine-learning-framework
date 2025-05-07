import os
import json
import time
import logging
import torch
import requests
import numpy as np
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_device() -> torch.device:
    """Set up and return the appropriate device (CPU/GPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, using CPU")
    return device

def get_system_info() -> Dict[str, Any]:
    """Get system information including CPU/GPU resources."""
    info = {
        "hostname": os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME', 'unknown'),
        "timestamp": time.time(),
        "cpu_count": os.cpu_count(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if info["gpu_available"]:
        info["gpu_info"] = []
        for i in range(info["gpu_count"]):
            info["gpu_info"].append({
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
                "memory_free": torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i),
            })
    
    return info

def serialize_model(model: torch.nn.Module) -> bytes:
    """Serialize a PyTorch model to bytes."""
    # Ensure model is initialized before serialization
    if hasattr(model, 'is_initialized') and not model.is_initialized:
        # Create a dummy input to initialize the model
        dummy_input = torch.zeros(1, 1, 28, 28)  # MNIST image size
        with torch.no_grad():
            model(dummy_input)  # This will initialize the model
    
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()

def deserialize_model(model: torch.nn.Module, serialized_model: bytes) -> torch.nn.Module:
    """Deserialize bytes to update a PyTorch model."""
    # Ensure model is initialized before deserialization
    if hasattr(model, 'is_initialized') and not model.is_initialized:
        # Create a dummy input to initialize the model
        dummy_input = torch.zeros(1, 1, 28, 28)  # MNIST image size
        with torch.no_grad():
            model(dummy_input)  # This will initialize the model
    
    buffer = io.BytesIO(serialized_model)
    state_dict = torch.load(buffer)
    model.load_state_dict(state_dict)
    return model

def federated_averaging(models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Perform federated averaging on a list of model parameters.
    
    Args:
        models: List of model state dictionaries
        
    Returns:
        Averaged model state dictionary
    """
    if not models:
        raise ValueError("No models provided for averaging")
        
    # Initialize with the first model
    avg_model = {}
    for key in models[0].keys():
        avg_model[key] = models[0][key].clone()
    
    # Add all models
    for i in range(1, len(models)):
        for key in avg_model.keys():
            avg_model[key] += models[i][key]
    
    # Divide by number of models
    for key in avg_model.keys():
        avg_model[key] = avg_model[key] / len(models)
        
    return avg_model

def send_heartbeat(coordinator_url: str, node_id: str, system_info: Dict[str, Any]) -> bool:
    """Send heartbeat to coordinator with system information."""
    try:
        response = requests.post(
            f"{coordinator_url}/api/heartbeat",
            json={"node_id": node_id, "system_info": system_info}
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to send heartbeat: {e}")
        return False

import io
import psutil

def get_detailed_system_info() -> Dict[str, Any]:
    """Get detailed system information."""
    info = get_system_info()
    
    # Add CPU usage information
    info["cpu_percent"] = psutil.cpu_percent(interval=1)
    info["memory_percent"] = psutil.virtual_memory().percent
    
    return info 