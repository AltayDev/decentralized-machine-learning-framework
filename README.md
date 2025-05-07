# Decentralized Machine Learning Framework

A decentralized system for training machine learning models across multiple nodes with either CPU or GPU hardware.

## Features

- Distributed training across multiple machines on different IP addresses
- Automatic node discovery and coordination
- Support for both CPU and GPU training
- Federated averaging for model aggregation
- Fault tolerance and node recovery
- Dynamic resource allocation

## System Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher (with CUDA for GPU support)
- Network connectivity between nodes

## Setup Instructions

### 1. Install Requirements

On all machines that will participate in the training:

```bash
# Clone this repository
git clone <repository-url>
cd decentralized-ml

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Coordinator

Choose one machine to act as the coordinator. This machine will coordinate the distributed training but doesn't need to have a GPU.

On Windows:
```
run_coordinator.bat
```

On Linux/Mac:
```bash
chmod +x run_coordinator.sh
./run_coordinator.sh
```

Or manually:
```bash
python coordinator.py --host 127.0.0.1 --port 5000
```

Note the IP address of the coordinator machine. You can get it by running:
- On Windows: `ipconfig`
- On Linux/Mac: `ifconfig` or `ip addr`

### 3. Start Worker Nodes

On each machine that will participate in the training:

On Windows:
```
run_worker.bat
```

On Linux/Mac:
```bash
chmod +x run_worker.sh
./run_worker.sh
```

Or manually:
```bash
# For GPU nodes:
python worker.py --coordinator-address <COORDINATOR_IP>:5000 --gpu

# For CPU nodes:
python worker.py --coordinator-address <COORDINATOR_IP>:5000 --cpu
```

Replace `<COORDINATOR_IP>` with the IP address of the coordinator machine.

## How It Works

1. The coordinator starts and initializes the global model.
2. Worker nodes connect to the coordinator and register.
3. Workers fetch the latest global model from the coordinator.
4. Each worker trains the model locally on its own data.
5. Workers submit their model updates back to the coordinator.
6. The coordinator aggregates all model updates using federated averaging.
7. The process repeats for multiple communication rounds.

## Monitoring

The coordinator provides a status API endpoint at:
```
http://<COORDINATOR_IP>:5000/api/status
```

This returns JSON data about:
- Current training round
- Number of active nodes
- Number of GPU/CPU nodes
- Training progress

## Customization

### Changing the Dataset

By default, the system uses the MNIST dataset. To change this, modify:
- `config.py`: Update dataset configuration
- `worker.py`: Update the `_load_data` method

### Changing the Model Architecture

To use a different model:
- Edit `model.py` with your custom architecture
- Update `get_model()` function

### Adjusting Training Parameters

Edit `config.py` to change:
- Batch size
- Learning rate
- Number of epochs
- Aggregation method

## Troubleshooting

- **Connectivity Issues**: Make sure all nodes can reach the coordinator's IP and port
- **CUDA Errors**: Verify CUDA compatibility with your GPU and PyTorch version
- **Memory Errors**: Reduce batch size in `config.py`

## Security Considerations

This system is designed for trusted networks. For production use:
- Enable authentication between nodes
- Add TLS/SSL for encrypted communication
- Implement proper access controls 

## New Feature

### Monitor

To monitor the training process, you can use the `monitor.py` script:
```
python monitor.py --coordinator-address 127.0.0.1:5000 