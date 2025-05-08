# Decentralized Text-to-Text Training Framework

This framework enables distributed training of text-to-text models (like T5 and BART) across multiple machines using federated learning. It consists of three main components:

1. **Coordinator**: Manages the training process and aggregates model updates
2. **Worker**: Performs local training on data and sends updates to the coordinator
3. **Monitor**: Web-based dashboard to track training progress

## Features

- Support for multiple text-to-text models (T5, BART)
- Real-time training monitoring via web dashboard
- Automatic model aggregation using federated averaging
- GPU/CPU support
- Fault tolerance and automatic node recovery
- Easy to extend with new models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/decentralized-text-training.git
cd decentralized-text-training
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Start the Coordinator

```bash
python coordinator.py --host 0.0.0.0 --port 5000 --model t5-small
```

### 2. Start Worker Nodes

On each machine that will participate in training:

```bash
python worker.py --coordinator-address <coordinator-ip>:5000 --model t5-small [--gpu]
```

### 3. Start the Monitor

```bash
python web_monitor.py --coordinator-address <coordinator-ip>:5000 --port 8080
```

Then open your browser and navigate to `http://localhost:8080` to view the training dashboard.

## Configuration

The framework can be configured through `config.py`. Key settings include:

- `BATCH_SIZE`: Batch size for training
- `LEARNING_RATE`: Learning rate for model updates
- `LOCAL_EPOCHS`: Number of local training epochs
- `MIN_NODES_TO_START`: Minimum number of nodes required to start training
- `NODE_TIMEOUT`: Timeout for inactive nodes
- `NODE_HEARTBEAT_INTERVAL`: Interval for node heartbeats

## Adding New Models

To add a new model:

1. Add the model configuration to `model_registry.py`:
```python
model_registry.register_model(
    name="your-model-name",
    model_class=YourModelClass,
    tokenizer_class=YourTokenizerClass,
    pretrained_name="pretrained-model-name",
    config={
        "max_length": 512,
        "num_beams": 4,
        "early_stopping": True
    }
)
```

2. Update the worker to handle the new model's data format and training process.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
