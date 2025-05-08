import os
import torch

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model paths
MODEL_PATH = os.path.join(MODELS_DIR, "global_model.pt")

# Coordinator settings
COORDINATOR_HOST = "0.0.0.0"
COORDINATOR_PORT = 5000

# Training settings
BATCH_SIZE = 32
TEST_BATCH_SIZE = 8
LEARNING_RATE = 1e-4
LOCAL_EPOCHS = 3
MIN_NODES_TO_START = 2
NODE_TIMEOUT = 90  # seconds
NODE_HEARTBEAT_INTERVAL = 30  # seconds
LOG_INTERVAL = 10

# Model settings
DEFAULT_MODEL = "t5-small"
MAX_LENGTH = 512
NUM_BEAMS = 4
EARLY_STOPPING = True

# System settings
CPU_WORKERS = 4
SAVE_MODEL = True

# Folder paths
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# Network Configuration
WORKER_PORT = int(os.environ.get("WORKER_PORT", 5001))
RESOURCE_POLL_INTERVAL = 5  # seconds

# Dataset Configuration
DATASET_NAME = "MNIST"  # Example dataset
DATA_DIR = os.path.join(DATA_DIR, DATASET_NAME.lower())

# Distributed Training Configuration
AGGREGATION_METHOD = "fedavg"  # Options: fedavg, fedsgd
COMMUNICATION_ROUNDS = 10

# System Configuration
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Training configuration
NUM_EPOCHS = 10
GRADIENT_CLIP = 1.0

# Model configuration
MODEL_CONFIGS = {
    "stable-diffusion": {
        "model_name": "runwayml/stable-diffusion-v1-5",
        "num_inference_steps": 50,
        "guidance_scale": 7.5
    }
}

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True) 