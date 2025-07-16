"""
Create a test checkpoint for debugging device compatibility issues
"""

import torch
import os
import sys
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.models.game_transformer import GameTransformer
from app.models.model_config import ModelConfig

def create_test_checkpoint():
    """Create a simple test checkpoint"""
    
    # Create a simple model configuration
    config = ModelConfig(
        d_model=256,
        n_heads=8,
        n_layers=4,
        n_experts=4,
        d_ff=1024,
        top_k=2,
        dropout=0.1,
        attention_dropout=0.1
    )
    
    # Create model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GameTransformer(config)
    model.to(device)
    
    # Create optimizer (required for checkpoint compatibility)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    
    # Create dummy training data
    dummy_board = torch.randint(0, 16, (1, 4, 4), device=device)
    
    # Test forward pass
    print(f"Testing model on device: {device}")
    print(f"Model parameters on device: {next(model.parameters()).device}")
    print(f"Input tensor device: {dummy_board.device}")
    
    with torch.no_grad():
        policy_logits, value = model(dummy_board)
        print(f"Policy logits device: {policy_logits.device}")
        print(f"Value device: {value.device}")
    
    # Save checkpoint
    checkpoint_dir = Path(os.getenv('CHECKPOINTS_DIR', '../backend/checkpoints'))
    print(f"[create_test_checkpoint] Using checkpoint_dir: {checkpoint_dir}")
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),  # Add missing optimizer state
        'config': config,
        'episode_count': 1,
        'total_steps': 0,  # Add missing total_steps
        'best_score': 1000,
        'loss_history': [(0, 0.5)],
        'score_history': [(0, 1000)],
        'training_duration': 60.0,
        'device': str(device)
    }
    
    checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
    torch.save(checkpoint_data, checkpoint_path)
    print(f"OK Test checkpoint saved to {checkpoint_path}")
    
    return checkpoint_path

if __name__ == "__main__":
    create_test_checkpoint() 