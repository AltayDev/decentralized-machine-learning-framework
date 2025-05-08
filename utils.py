import os
import json
import time
import logging
import torch
import requests
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import io
import psutil
import base64
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_device() -> torch.device:
    """Set up the device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

def get_system_info() -> Dict[str, Any]:
    """Get system information for the node."""
    return {
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

def get_detailed_system_info() -> Dict[str, Any]:
    """Get detailed system information."""
    info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "memory_percent": psutil.virtual_memory().percent,
        "gpu_available": torch.cuda.is_available(),
    }
    
    if info["gpu_available"]:
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
            "gpu_memory_allocated": torch.cuda.memory_allocated(0),
            "gpu_memory_cached": torch.cuda.memory_reserved(0)
        })
    
    return info

def send_heartbeat(coordinator_address: str, node_id: str, model_name: str) -> bool:
    """Send heartbeat signal to coordinator."""
    try:
        response = requests.post(
            f"http://{coordinator_address}/heartbeat",
            json={
                "node_id": node_id,
                "system_info": get_system_info(),
                "model_name": model_name
            }
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error sending heartbeat: {str(e)}")
        return False

def serialize_model(state_dict: Dict[str, torch.Tensor]) -> str:
    """Serialize model state dict to base64 string."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def deserialize_model(serialized_state: str) -> Dict[str, torch.Tensor]:
    """Deserialize base64 string to model state dict."""
    buffer = io.BytesIO(base64.b64decode(serialized_state))
    return torch.load(buffer)

def federated_averaging(model_states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Perform federated averaging on model states."""
    if not model_states:
        raise ValueError("No model states provided for averaging")
    
    # Initialize averaged state with zeros
    averaged_state = {}
    for key in model_states[0].keys():
        averaged_state[key] = torch.zeros_like(model_states[0][key])
    
    # Sum up all states
    for state in model_states:
        for key in averaged_state.keys():
            averaged_state[key] += state[key]
    
    # Divide by number of models
    for key in averaged_state.keys():
        averaged_state[key] /= len(model_states)
    
    return averaged_state

def calculate_metrics(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Calculate metrics for text generation."""
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize
    import nltk
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Calculate BLEU score
    bleu_scores = []
    for pred, target in zip(predictions, targets):
        pred_tokens = word_tokenize(pred.lower())
        target_tokens = word_tokenize(target.lower())
        bleu_scores.append(sentence_bleu([target_tokens], pred_tokens))
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    return {
        "bleu_score": avg_bleu
    }

def save_training_history(history: Dict[str, List[float]], filepath: str):
    """Save training history to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(history, f)

def load_training_history(filepath: str) -> Dict[str, List[float]]:
    """Load training history from a JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {} 