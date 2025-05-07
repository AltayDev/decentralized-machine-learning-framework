import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

class SimpleCNN(nn.Module):
    """A simple Convolutional Neural Network for image classification."""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Max pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(0.25)
        
        # Fully connected layers will be initialized dynamically
        self.fc1 = None
        self.fc2 = nn.Linear(128, 10)
        
        # Flag to indicate if forward has been called yet
        self.is_initialized = False
        
    def _initialize_fc_layers(self, x):
        """Initialize fully connected layers based on input dimensions."""
        # Get the flattened size after convolutions and pooling
        with torch.no_grad():
            # First conv block
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            
            # Second conv block
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            
            # Pooling
            x = self.pool(x)
            
            # Get flattened size
            flattened_size = x.size(1) * x.size(2) * x.size(3)
            
            # Initialize FC layers
            self.fc1 = nn.Linear(flattened_size, 128)
            
            print(f"Initialized FC layer with input size: {flattened_size}")
        
        self.is_initialized = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If first run, initialize FC layers based on input dimensions
        if not self.is_initialized:
            self._initialize_fc_layers(x)
        
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Pooling and dropout
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten and feed to fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_size(self) -> int:
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
        
def get_model() -> nn.Module:
    """Factory function to create and initialize a model."""
    model = SimpleCNN()
    return model

def train_batch(model: nn.Module, 
               data: torch.Tensor, 
               target: torch.Tensor, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device) -> Dict[str, float]:
    """Train model on a single batch of data."""
    model.train()
    data, target = data.to(device), target.to(device)
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Forward pass
    output = model(data)
    loss = F.nll_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    # Calculate accuracy
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(data)
    
    return {
        "loss": loss.item(),
        "accuracy": accuracy
    }

def test(model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
    """Test the model on the test dataset."""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return {
        "test_loss": test_loss,
        "test_accuracy": accuracy
    } 