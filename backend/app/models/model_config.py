"""
Dynamic model configuration based on available VRAM
"""

import torch
import psutil
from typing import Dict, Any, Optional
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
    # Optional label for UI/selection
    model_size: Optional[str] = None
    
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

    @property
    def estimated_active_params(self) -> float:
        """Estimate active parameters used per forward pass in millions.

        Assumes attention fully active and MoE activates only ``top_k`` experts
        per token.
        """
        attention_active = self.n_layers * self.n_heads * self.d_model * self.d_model * 3
        moe_active = self.n_layers * self.top_k * self.d_model * self.d_ff * 2
        output_heads = self.d_model * 5
        total_active = attention_active + moe_active + output_heads
        return total_active / 1e6

class DynamicModelConfig:
    """Dynamically configure model based on available resources"""

    # Predefined configurations targeting total/active parameter budgets
    # Profiles are tuned for 16-token 2048 boards and PPO stability.
    # lightning: compact, fast (edge/CPU); base: balanced; expert: larger but still efficient
    CONFIGS = {
        "lightning": ModelConfig(
            d_model=128,
            n_heads=4,   # 128 dims / 4 heads = 32 per head
            n_layers=4,  # a bit more depth helps with planning
            n_experts=6, # fewer but higher-quality experts for tiny models
            d_ff=256,
            top_k=2,     # allow two experts to reduce starvation
            dropout=0.1,
            attention_dropout=0.1,
            model_size="lightning",
        ),
        "base": ModelConfig(
            d_model=256,
            n_heads=8,   # 256 / 8 = 32
            n_layers=6,
            n_experts=8,
            d_ff=512,
            top_k=2,
            dropout=0.1,
            attention_dropout=0.1,
            model_size="base",
        ),
        "expert": ModelConfig(
            d_model=512,
            n_heads=8,   # 512 / 8 = 64
            n_layers=8,
            n_experts=16,
            d_ff=1024,
            top_k=2,
            dropout=0.1,
            attention_dropout=0.1,
            model_size="expert",
        ),
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
            return 0.0  # Always return a float, never None
    
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
    def select_config(
        cls,
        target_vram: Optional[float] = None,
        target_profile: Optional[str] = None,
    ) -> ModelConfig:
        """Select appropriate configuration.

        Priority:
        1) If ``target_profile`` in {lightning, base, expert} is provided, use that.
        2) Else select by available VRAM.
        """

        if target_profile in cls.CONFIGS:
            config_name = target_profile
            available_vram = target_vram if target_vram is not None else cls.get_available_vram()
        elif target_vram is not None:
            if target_vram < 4.0:
                config_name = "lightning"
            elif target_vram < 12.0:
                config_name = "base"
            else:
                config_name = "expert"
            available_vram = target_vram
        else:
            available_vram = cls.get_available_vram()
            if available_vram >= 12.0:
                config_name = "expert"
            elif available_vram >= 4.0:
                config_name = "base"
            else:
                config_name = "lightning"
        
        system_ram = cls.get_system_ram()
        
        console.print(f"[blue]Available VRAM: {available_vram:.1f}GB")
        console.print(f"[blue]Available RAM: {system_ram:.1f}GB")
        
        # Fallback to CPU-optimized config names preserved in our scheme
        if available_vram < 2.0 and system_ram >= 8.0 and config_name != "lightning":
            console.print("[yellow]Low VRAM detected, using CPU-optimized lightning config")
            config_name = "lightning"
        
        config = cls.CONFIGS[config_name]
        console.print(f"[green]Selected config: {config_name}")
        console.print(f"[green]Estimated parameters: {config.estimated_params:.1f}M (active ~{config.estimated_active_params:.1f}M)")
        
        return config
    
    @classmethod
    def get_batch_size(cls, config: ModelConfig, available_vram: Optional[float] = None) -> int:
        """Get appropriate batch size based on model config and VRAM"""
        
        if available_vram is None:
            available_vram = cls.get_available_vram() or 0.0  # Handle None case
        
        # Optimized baseline based on profile and model width
        if config.model_size == "expert" or config.d_model >= 512:
            base_batch_size = 64
        elif config.model_size == "base" or config.d_model >= 256:
            base_batch_size = 128
        else:
            base_batch_size = 256

        # Adjust based on available VRAM (more aggressive scaling for high VRAM)
        if available_vram < 4.0:
            batch_size = max(32, int(base_batch_size // 2))
        elif available_vram < 12.0:
            batch_size = max(64, int(base_batch_size // 1.5))
        elif available_vram < 16.0:
            batch_size = base_batch_size
        elif available_vram < 24.0:
            batch_size = int(base_batch_size * 1.5)
        else:
            batch_size = int(base_batch_size * 2)
        # Round to nearest multiple of 16 for better kernel efficiency
        batch_size = max(32, (batch_size // 16) * 16)
        
        console.print(f"[blue]Selected batch size: {batch_size}")
        return batch_size
    
    @classmethod
    def get_device(cls) -> torch.device:
        """Get the appropriate device (CUDA or CPU)"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            console.print(f"[green]Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            console.print("[yellow]Using CPU (CUDA not available)")
        
        return device 

    @classmethod
    def get_default_learning_rate(cls, config: ModelConfig) -> float:
        """Return a sane default learning rate for PPO based on model profile.

        Rationale:
        - lightning/base are compact enough to train stably at 3e-4.
        - expert has substantially more parameters; a slightly lower LR (2e-4)
          improves stability without materially slowing convergence.
        """
        profile = getattr(config, "model_size", None) or "base"
        if profile == "expert":
            return 2e-4
        # Default for lightning and base
        return 3e-4