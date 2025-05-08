import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import logging

logger = logging.getLogger(__name__)

# Try importing transformers, but don't fail if not available
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from transformers import BartForConditionalGeneration, BartTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers package not available. Text-to-text models will not be available.")

class SimpleLSTM(nn.Module):
    """A simple LSTM model for text classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_dim: int = 512, num_layers: int = 2):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(output)

class SimpleTokenizer:
    """A simple tokenizer that splits text into characters."""
    
    def __init__(self):
        self.vocab = {chr(i): i for i in range(128)}  # ASCII characters
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
        # Convert text to character indices
        indices = [self.vocab.get(c, 0) for c in text[:max_length]]
        # Pad or truncate to max_length
        if len(indices) < max_length:
            indices.extend([0] * (max_length - len(indices)))
        return torch.tensor(indices)
    
    def decode(self, indices: torch.Tensor) -> str:
        # Convert indices back to text
        return ''.join(chr(i) for i in indices if i < 128)

class SimpleImageGenerator(nn.Module):
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        super().__init__()
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Optimize for low VRAM usage
        if torch.cuda.is_available():
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_model_cpu_offload()
        
        self.tokenizer = self.pipeline.tokenizer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler
        
    def forward(self, prompt: str, num_inference_steps: int = 50) -> torch.Tensor:
        # Tokenize the prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        # Get text embeddings
        text_embeddings = self.text_encoder(text_input_ids)[0]
        
        # Generate latent space
        latents = torch.randn(
            (1, self.unet.config.in_channels, 64, 64),
            device=text_embeddings.device
        )
        latents = latents * self.scheduler.init_noise_sigma
        
        # Denoising loop
        for t in self.scheduler.timesteps:
            # Predict noise residual
            noise_pred = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
            
            # Update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents to image
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        
        return image

class ModelRegistry:
    """Registry for managing different model types and their configurations."""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default model configurations."""
        # Register image generation model
        self.register_model(
            model_name="stable-diffusion",
            model_class=SimpleImageGenerator,
            tokenizer_class=None,  # Tokenizer is part of the pipeline
            config={"model_name": "runwayml/stable-diffusion-v1-5"}
        )
    
    def register_model(self, model_name: str, model_class: type, tokenizer_class: Optional[type], config: Dict[str, Any]):
        self.models[model_name] = {
            "model_class": model_class,
            "tokenizer_class": tokenizer_class,
            "config": config
        }
        logger.info(f"Registered model: {model_name}")
    
    def get_model(self, model_name: str) -> Dict[str, Any]:
        """Get a model instance by name."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if not TRANSFORMERS_AVAILABLE and model_name in ["t5-small", "bart-base"]:
            raise ImportError("transformers package is required for text-to-text models")
        
        model_info = self.models[model_name]
        model = model_info["model_class"](**model_info["config"])
        return model
    
    def get_tokenizer(self, name: str) -> Any:
        """Get a tokenizer instance by name."""
        if name not in self.models:
            raise ValueError(f"Model {name} not found in registry")
        
        if not TRANSFORMERS_AVAILABLE and name in ["t5-small", "bart-base"]:
            raise ImportError("transformers package is required for text-to-text models")
        
        model_info = self.models[name]
        return model_info["tokenizer_class"]()
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """Get model configuration by name."""
        if name not in self.models:
            raise ValueError(f"Model {name} not found in registry")
        
        return self.models[name]["config"]
    
    def list_models(self) -> list:
        """List all registered model names."""
        return list(self.models.keys())

# Create global registry instance
model_registry = ModelRegistry() 