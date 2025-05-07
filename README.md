# Decentralized Machine Learning Framework

A decentralized system for training machine learning models across multiple nodes using either CPU or GPU hardware.

## Table of Contents
- [Features](#features)
- [System Requirements](#system-requirements)
- [Architecture Overview](#architecture-overview)
- [Setup](#setup)
- [Manual Launch Commands](#manual-launch-commands)
- [Code Structure and Components](#code-structure-and-components)
- [GPU Configuration](#gpu-configuration)
- [Troubleshooting](#troubleshooting)

## Features

- Distributed training across multiple machines on different IP addresses
- Model aggregation using federated averaging
- Support for both CPU and GPU training
- Fault tolerance and node recovery mechanisms
- User-friendly monitoring tools
- Modular and extensible design

## System Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher (with CUDA for GPU support)
- Windows, Linux, or macOS operating system
- Network connectivity between nodes
- For GPU training: NVIDIA GPU and compatible drivers

## Architecture Overview

This system uses a decentralized approach instead of a classic client-server architecture:

1. **Coordinator**: Central coordination point of the system, but doesn't perform training. Responsible for:
   - Node registration and status tracking
   - Global model distribution
   - Collection of local model updates
   - Model aggregation using federated averaging

2. **Worker Nodes**: Perform the actual training process. Responsible for:
   - Fetching the latest global model from the coordinator
   - Training the model on their local datasets
   - Submitting training results back to the coordinator

3. **Monitors**: Additional tools to monitor the overall system state and training performance.

## Setup

### 1. Install Requirements

```bash
# Clone the repository
git clone https://github.com/AltayDev/decentralized-machine-learning-framework.git
cd decentralized-machine-learning-framework

# Install dependencies
pip install -r requirements.txt
```

### 2. Preparing Data and Model Directories

The system will automatically create the following directory structure:

```
/data        - For training data
/models      - Where trained models will be saved
/logs        - System logs
/temp        - Temporary files
```

## Manual Launch Commands

### Starting the Coordinator

```bash
# With default settings:
python coordinator.py

# With custom host and port:
python coordinator.py --host 192.168.1.100 --port 5000
```

### Starting a Worker Node

```bash
# Using GPU:
python worker.py --coordinator-address 192.168.1.100:5000 --gpu

# Using CPU:
python worker.py --coordinator-address 192.168.1.100:5000 --cpu

# With custom data directory:
python worker.py --coordinator-address 192.168.1.100:5000 --data-dir ./custom_data
```

### Starting a Monitor

```bash
# Basic monitoring:
python monitor.py --coordinator-address 192.168.1.100:5000

# Text-based monitoring:
python text_monitor.py --coordinator-address 192.168.1.100:5000
```

## Code Structure and Components

### Core Modules

1. **coordinator.py**: Implementation of the coordinator node
   - Node registration and status tracking
   - Model distribution and aggregation
   - RESTful API endpoints

2. **worker.py**: Implementation of the worker node
   - Model training and evaluation
   - Communication with coordinator
   - Data processing

3. **model.py**: Model definition and training functions
   - SimpleCNN class: A simple CNN model for image classification
   - train_batch: Function for training on a single batch
   - test: Model evaluation function

4. **config.py**: System-wide configuration settings
   - Directory paths
   - Network configuration
   - Training parameters

5. **utils.py**: Helper functions and tools
   - Device setup (GPU/CPU)
   - Model serialization and deserialization
   - Federated averaging implementation

6. **monitor.py/text_monitor.py**: Monitoring tools
   - Training progress visualization
   - System status and performance metrics

### Data Flow

1. Coordinator starts and waits for nodes to connect.
2. Worker nodes start and register with the coordinator.
3. Each worker downloads the latest global model from the coordinator.
4. Workers independently train the model on their local data.
5. After training is complete, updated model parameter values are sent to the coordinator.
6. The coordinator combines model updates from all nodes using the federated averaging algorithm.
7. This process continues for the configured number of communication rounds.

### Model Architecture

By default, the system uses a SimpleCNN model:
- Two convolutional layers (32 and 64 filters)
- Batch normalization
- MaxPooling
- Dropout regularization
- Two fully connected layers

The model architecture is optimized for CNN-based image classification tasks and is designed to perform well on simple image datasets like MNIST.

## GPU Configuration

### GPU Configuration on Windows

1. **Install NVIDIA Drivers**:
   - Download and install the latest NVIDIA drivers from [NVIDIA Driver Download](https://www.nvidia.com/Download/index.aspx).

2. **Install CUDA Toolkit**:
   - Download and install CUDA Toolkit (11.8 or 12.x recommended) from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).

3. **Check Environment Variables**:
   - Ensure `CUDA_PATH` and `Path` variables are correctly set.

4. **Install PyTorch with GPU Support**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Test GPU Availability**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

### GPU Configuration on Linux

1. **Install NVIDIA Drivers**:
   ```bash
   sudo apt update
   sudo apt install nvidia-driver-535  # Version number may vary
   ```

2. **Install CUDA Toolkit**:
   - Download and install CUDA Toolkit from NVIDIA's website.

3. **Install PyTorch with GPU Support**:
   ```bash
   pip install torch torchvision torchaudio
   ```

## Troubleshooting

### Model Saving Issues

If models are not being saved to the `models/` directory:
- Make sure the coordinator is running
- Verify that the `SAVE_MODEL` variable in `config.py` is set to `True`
- Check write permissions for the `models/` directory
- Examine coordinator logs for error messages
- Ensure a training round has been fully completed

### GPU Issues

1. **CUDA Availability Issues**:
   - Check that NVIDIA drivers are up-to-date
   - Verify CUDA Toolkit is properly installed
   - Confirm that a CUDA-enabled version of PyTorch is installed
   - Verify the GPU is recognized by the system using the `nvidia-smi` command

2. **Out of Memory Errors**:
   - Try reducing batch size (decrease the `BATCH_SIZE` value in `config.py`)
   - Use a smaller model
   - Reduce the amount of data

3. **Performance Issues**:
   - Optimize the number of CPU threads (adjust `CPU_WORKERS` in `config.py`)
   - Ensure the GPU is not being used by other applications

4. **Windows-Specific GPU Issues**:
   - Ensure Visual C++ Redistributable packages are installed
   - Make sure you're using the appropriate PyTorch version for your CUDA version

This decentralized machine learning framework provides a flexible and scalable solution for model training across different hardware configurations.
