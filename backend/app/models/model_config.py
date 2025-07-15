"""
Dynamic model configuration based on available VRAM
"""

import torch
import psutil
from typing import Dict, Any
from dataclasses import dataclass
from rich.console import Console

console = Console()

@dataclass
class ModelConfig:
    """Configuration for the transformer model"""
    d_model: int
    n_heads: int
    n_layers: int
    n_experts: int
    d_ff: int
    top_k: int
    dropout: float
    attention_dropout: float
    
    @property
    def estimated_params(self) -> float:
        """Estimate model parameters in millions"""
        # Rough estimation based on transformer architecture
        # Embedding: 16 * d_model
        # Attention: n_layers * n_heads * d_model * d_model * 3 (Q, K, V)
        # MoE: n_layers * n_experts * d_model * d_ff * 2
        # Output heads: d_model * 4 (policy) + d_model * 1 (value)
        
        embedding = 16 * self.d_model
        attention = self.n_layers * self.n_heads * self.d_model * self.d_model * 3
        moe = self.n_layers * self.n_experts * self.d_model * self.d_ff * 2
        output_heads = self.d_model * 5
        
        total = embedding + attention + moe + output_heads
        return total / 1e6  # Convert to millions

class DynamicModelConfig:
    """Dynamically configure model based on available resources"""
    
    # Predefined configurations for different VRAM levels
    CONFIGS = {
        "small": ModelConfig(
            d_model=256,
            n_heads=8,
            n_layers=4,
            n_experts=4,
            d_ff=1024,
            top_k=2,
            dropout=0.1,
            attention_dropout=0.1
        ),
        "medium": ModelConfig(
            d_model=384,
            n_heads=12,
            n_layers=6,
            n_experts=6,
            d_ff=1536,
            top_k=2,
            dropout=0.1,
            attention_dropout=0.1
        ),
        "large": ModelConfig(
            d_model=512,
            n_heads=16,
            n_layers=8,
            n_experts=8,
            d_ff=2048,
            top_k=2,
            dropout=0.1,
            attention_dropout=0.1
        )
    }
    
    @classmethod
    def get_available_vram(cls) -> float:
        """Get available VRAM in GB"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            # Get total and allocated memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            available_memory = total_memory - allocated_memory
            
            return available_memory / (1024**3)  # Convert to GB
        except Exception as e:
            console.print(f"[yellow]Warning: Could not get VRAM info: {e}")
            return 0.0
    
    @classmethod
    def get_system_ram(cls) -> float:
        """Get available system RAM in GB"""
        try:
            available_memory = psutil.virtual_memory().available
            return available_memory / (1024**3)  # Convert to GB
        except Exception as e:
            console.print(f"[yellow]Warning: Could not get RAM info: {e}")
            return 4.0  # Default fallback
    
    @classmethod
    def select_config(cls, target_vram: float = None) -> ModelConfig:
        """Select appropriate configuration based on available resources"""
        
        if target_vram is None:
            available_vram = cls.get_available_vram()
        else:
            available_vram = target_vram
        
        system_ram = cls.get_system_ram()
        
        console.print(f"[blue]Available VRAM: {available_vram:.1f}GB")
        console.print(f"[blue]Available RAM: {system_ram:.1f}GB")
        
        # Select configuration based on available VRAM
        if available_vram >= 6.0:
            config_name = "large"
        elif available_vram >= 4.0:
            config_name = "medium"
        else:
            config_name = "small"
        
        # Fallback to CPU if very low VRAM
        if available_vram < 2.0 and system_ram >= 8.0:
            console.print("[yellow]Low VRAM detected, using CPU-optimized config")
            config_name = "small"
        
        config = cls.CONFIGS[config_name]
        console.print(f"[green]Selected config: {config_name}")
        console.print(f"[green]Estimated parameters: {config.estimated_params:.1f}M")
        
        return config
    
    @classmethod
    def get_batch_size(cls, config: ModelConfig, available_vram: float = None) -> int:
        """Get appropriate batch size based on model config and VRAM"""
        
        if available_vram is None:
            available_vram = cls.get_available_vram()
        
        # Rough estimation: larger models need smaller batches
        if config.d_model >= 512:
            base_batch_size = 32
        elif config.d_model >= 384:
            base_batch_size = 64
        else:
            base_batch_size = 128
        
        # Adjust based on available VRAM
        if available_vram < 4.0:
            batch_size = max(16, base_batch_size // 4)
        elif available_vram < 6.0:
            batch_size = max(32, base_batch_size // 2)
        else:
            batch_size = base_batch_size
        
        console.print(f"[blue]Selected batch size: {batch_size}")
        return batch_size
    
    @classmethod
    def get_device(cls) -> torch.device:
        """Get the appropriate device (CUDA or CPU)"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            console.print(f"[green]Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            console.print("[yellow]Using CPU (CUDA not available)")
        
        return device 