import os
import sys
import time
import uuid
import json
import logging
import argparse
import threading
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import requests
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import numpy as np
from PIL import Image

from model_registry import model_registry
from utils import (
    setup_device, 
    get_detailed_system_info, 
    send_heartbeat,
    get_system_info,
    serialize_model,
    deserialize_model
)
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOGS_DIR, 'worker.log'))
    ]
)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Dataset for text-to-text tasks."""
    
    def __init__(self, texts: List[str], targets: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        
        # Tokenize inputs using encode method
        input_ids = self.tokenizer.encode(text, max_length=self.max_length)
        target_ids = self.tokenizer.encode(target, max_length=self.max_length)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_ids
        }

class ImageDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class Worker:
    """Worker node for decentralized model training."""
    
    def __init__(
        self,
        coordinator_url: str,
        model_name: str = "t5-small",
        use_gpu: bool = None,
        data_dir: str = config.DATA_DIR
    ):
        self.coordinator_url = coordinator_url
        self.model_name = model_name
        self.node_id = None
        self.device = setup_device() if use_gpu is None else torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.data_dir = data_dir
        self.train_loader = None
        self.test_loader = None
        self.is_running = False
        self.heartbeat_thread = None
        self.training_round = -1
        
        logger.info(f"Worker initialized with device: {self.device}")
        
        # Register with coordinator
        self._register()
        
        # Load model and tokenizer
        self._load_model()
        
        # Load data
        self._load_data()
        
        # Start heartbeat thread
        self._start_heartbeat()
    
    def _register(self):
        """Register with the coordinator."""
        try:
            logger.info(f"Registering with coordinator at {self.coordinator_url}")
            response = requests.post(
                f"{self.coordinator_url}/api/register",
                json={"system_info": get_detailed_system_info()}
            )
            
            if response.status_code != 200:
                logger.error(f"Registration failed with status {response.status_code}: {response.text}")
                raise Exception(f"Registration failed: {response.text}")
                
            data = response.json()
            self.node_id = data["node_id"]
            logger.info(f"Registered with coordinator, node ID: {self.node_id}")
            
            # Save node_id to disk for recovery
            node_id_file = os.path.join(config.TEMP_DIR, f"node_id_{self.coordinator_url.replace('http://', '').replace(':', '_')}.txt")
            with open(node_id_file, "w") as f:
                f.write(self.node_id)
            
        except Exception as e:
            logger.error(f"Failed to register with coordinator: {e}")
            
            # Try to recover node_id from disk
            try:
                node_id_file = os.path.join(config.TEMP_DIR, f"node_id_{self.coordinator_url.replace('http://', '').replace(':', '_')}.txt")
                if os.path.exists(node_id_file):
                    with open(node_id_file, "r") as f:
                        self.node_id = f.read().strip()
                    logger.info(f"Recovered node ID from file: {self.node_id}")
                    return
            except Exception as e2:
                logger.error(f"Could not recover node ID: {e2}")
            
            raise
    
    def _load_model(self):
        """Load model and tokenizer from registry."""
        try:
            logger.info(f"Loading model {self.model_name} from registry")
            self.model = model_registry.get_model(self.model_name)
            self.tokenizer = model_registry.get_tokenizer(self.model_name)
            self.model = self.model.to(self.device)
            
            # Create optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.LEARNING_RATE
            )
            
            # Create scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=100,
                num_training_steps=1000
            )
            
            logger.info(f"Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_data(self):
        """Load the training and testing data."""
        logger.info("Loading text dataset...")
        
        # TODO: Replace with actual text dataset loading
        # This is a placeholder for demonstration
        train_texts = ["Sample text 1", "Sample text 2"]
        train_targets = ["Target 1", "Target 2"]
        test_texts = ["Test text 1"]
        test_targets = ["Test target 1"]
        
        # Create datasets
        train_dataset = TextDataset(
            train_texts,
            train_targets,
            self.tokenizer,
            model_registry.get_config(self.model_name)["max_length"]
        )
        
        test_dataset = TextDataset(
            test_texts,
            test_targets,
            self.tokenizer,
            model_registry.get_config(self.model_name)["max_length"]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.CPU_WORKERS
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=config.CPU_WORKERS
        )
        
        logger.info("Data loading complete")
    
    def train_batch(self, batch):
        """Train on a single batch."""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return {
            "loss": loss.item(),
            "accuracy": 0.0  # TODO: Implement accuracy calculation
        }
    
    def test(self):
        """Test the model."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item() * len(input_ids)
                total_samples += len(input_ids)
        
        return {
            "test_loss": total_loss / total_samples if total_samples > 0 else 0,
            "test_accuracy": 0.0  # TODO: Implement accuracy calculation
        }
    
    def train_local(self) -> Dict[str, float]:
        """Train the model locally for a number of epochs."""
        if self.model is None:
            if not self._get_global_model():
                return {"error": "Failed to get global model"}
        
        logger.info(f"Starting local training for {config.LOCAL_EPOCHS} epochs...")
        metrics = {
            "train_loss": 0,
            "train_accuracy": 0,
            "test_loss": 0,
            "test_accuracy": 0
        }
        
        for epoch in range(config.LOCAL_EPOCHS):
            # Training phase
            total_loss = 0
            total_accuracy = 0
            batch_count = 0
            
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.LOCAL_EPOCHS}") as progress_bar:
                for batch_idx, batch in enumerate(progress_bar):
                    # Train on batch
                    result = self.train_batch(batch)
                    
                    # Update metrics
                    total_loss += result["loss"]
                    total_accuracy += result["accuracy"]
                    batch_count += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': result["loss"],
                        'accuracy': f"{result['accuracy']:.2f}%"
                    })
                    
                    # Log every LOG_INTERVAL batches
                    if batch_idx % config.LOG_INTERVAL == 0:
                        logger.info(f"Train Epoch: {epoch+1} [{batch_idx*len(batch['input_ids'])}/{len(self.train_loader.dataset)} "
                                   f"({100. * batch_idx / len(self.train_loader):.0f}%)]\t"
                                   f"Loss: {result['loss']:.6f}, Accuracy: {result['accuracy']:.2f}%")
            
            # Calculate average metrics for the epoch
            metrics["train_loss"] = total_loss / batch_count if batch_count > 0 else 0
            metrics["train_accuracy"] = total_accuracy / batch_count if batch_count > 0 else 0
            
            logger.info(f"Epoch {epoch+1} complete: "
                       f"Avg Loss: {metrics['train_loss']:.6f}, "
                       f"Avg Accuracy: {metrics['train_accuracy']:.2f}%")
        
        # Test the model
        test_results = self.test()
        metrics.update(test_results)
        
        logger.info(f"Testing complete: "
                   f"Loss: {test_results['test_loss']:.6f}, "
                   f"Accuracy: {test_results['test_accuracy']:.2f}%")
        
        return metrics
    
    def _get_global_model(self) -> bool:
        """Fetch the global model from the coordinator."""
        try:
            logger.info("Fetching global model from coordinator...")
            
            # Request the model
            response = requests.get(
                f"{self.coordinator_url}/api/get_model",
                params={"node_id": self.node_id},
                stream=True
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch model: {response.text}")
                return False
            
            # Save the model file
            model_path = os.path.join(config.TEMP_DIR, f"temp_global_model_{self.node_id}.pt")
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Load the model
            if self.model is None:
                self.model = model_registry.get_model(self.model_name)
                
                # For SimpleCNN, initialize with a dummy input to set up FC layers
                if hasattr(self.model, 'is_initialized') and not self.model.is_initialized:
                    logger.info("Initializing model layers with dummy input...")
                    dummy_input = torch.zeros(1, 1, 28, 28)  # MNIST image size
                    with torch.no_grad():
                        self.model(dummy_input)
            
            # Load state dict
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            
            # Create optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.LEARNING_RATE
            )
            
            # Create scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=100,
                num_training_steps=1000
            )
            
            # Get training round from coordinator status
            status_response = requests.get(f"{self.coordinator_url}/api/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                self.training_round = status_data.get("training_round", 0)
            
            # Remove temporary file
            os.remove(model_path)
            
            logger.info(f"Global model loaded for round {self.training_round}")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching global model: {e}")
            return False
    
    def _submit_model_update(self) -> bool:
        """Submit the local model update to the coordinator."""
        try:
            logger.info("Submitting model update to coordinator...")
            
            # Save the model temporarily
            model_path = os.path.join(config.TEMP_DIR, f"temp_local_model_{self.node_id}.pt")
            torch.save(self.model.state_dict(), model_path)
            
            # Send the model
            with open(model_path, 'rb') as f:
                files = {'model': f}
                response = requests.post(
                    f"{self.coordinator_url}/api/submit_update",
                    data={"node_id": self.node_id},
                    files=files
                )
            
            # Remove temporary file
            os.remove(model_path)
            
            if response.status_code != 200:
                logger.error(f"Failed to submit model update: {response.text}")
                return False
            
            logger.info("Model update submitted successfully")
            data = response.json()
            if data.get("aggregation_triggered", False):
                logger.info("Model aggregation was triggered")
            
            return True
            
        except Exception as e:
            logger.error(f"Error submitting model update: {e}")
            return False
    
    def run_training_loop(self):
        """Run the continuous training loop."""
        logger.info("Starting training loop")
        
        while self.is_running:
            try:
                # Get latest model
                if not self._get_global_model():
                    logger.error("Failed to get global model, retrying in 30 seconds")
                    time.sleep(30)
                    continue
                
                # Train locally
                metrics = self.train_local()
                
                # Submit model update
                if not self._submit_model_update():
                    logger.error("Failed to submit model update, retrying in 30 seconds")
                    time.sleep(30)
                    continue
                
                logger.info(f"Training round {self.training_round} complete with "
                           f"test accuracy: {metrics.get('test_accuracy', 0):.2f}%")
                
                # Wait a bit before next round
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                time.sleep(30)
        
        logger.info("Training loop stopped")
    
    def stop(self):
        """Stop the worker."""
        logger.info("Stopping worker...")
        self.is_running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1)
        logger.info("Worker stopped")
    
    def _start_heartbeat(self):
        """Start sending heartbeat signals to coordinator."""
        def heartbeat_worker():
            while self.is_running:
                try:
                    send_heartbeat(
                        self.coordinator_url,
                        self.node_id,
                        get_detailed_system_info(),
                        self.model_name
                    )
                except Exception as e:
                    logger.error(f"Error sending heartbeat: {e}")
                time.sleep(config.NODE_HEARTBEAT_INTERVAL)
        
        self.is_running = True
        self.heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()
        logger.info("Heartbeat thread started")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Decentralized ML Worker Node")
    parser.add_argument("--coordinator-address", type=str, required=True,
                        help="Coordinator address in the format <host>:<port>")
    parser.add_argument("--model", type=str, default="t5-small",
                        help="Model name to use (default: t5-small)")
    parser.add_argument("--gpu", action="store_true",
                        help="Force using GPU if available")
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR,
                        help=f"Directory to store dataset (default: {config.DATA_DIR})")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Determine which device to use
    use_gpu = None
    if args.gpu:
        use_gpu = True
    elif args.cpu:
        use_gpu = False
    
    # Construct coordinator URL
    coordinator_url = f"http://{args.coordinator_address}"
    
    # Create worker
    worker = Worker(
        coordinator_url=coordinator_url,
        model_name=args.model,
        use_gpu=use_gpu,
        data_dir=args.data_dir
    )
    
    try:
        # Run training loop
        worker.run_training_loop()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        worker.stop() 