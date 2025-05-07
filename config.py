import os
import torch

# Folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)

# Network Configuration
COORDINATOR_HOST = os.environ.get("COORDINATOR_HOST", "0.0.0.0")
COORDINATOR_PORT = int(os.environ.get("COORDINATOR_PORT", 5000))
WORKER_PORT = int(os.environ.get("WORKER_PORT", 5001))
NODE_HEARTBEAT_INTERVAL = 10  # seconds
NODE_TIMEOUT = 30  # seconds

# Training Configuration
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10
SAVE_MODEL = True
MODEL_PATH = os.path.join(MODELS_DIR, "model.pt")

# System Configuration
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
CPU_WORKERS = int(os.environ.get("CPU_WORKERS", 4))
RESOURCE_POLL_INTERVAL = 5  # seconds

# Dataset Configuration
DATASET_NAME = "MNIST"  # Example dataset
DATA_DIR = os.path.join(DATA_DIR, DATASET_NAME.lower())

# Distributed Training Configuration
AGGREGATION_METHOD = "fedavg"  # Options: fedavg, fedsgd
MIN_NODES_TO_START = 1
COMMUNICATION_ROUNDS = 10
LOCAL_EPOCHS = 2 