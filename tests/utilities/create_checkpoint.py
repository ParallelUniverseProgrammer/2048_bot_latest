#!/usr/bin/env python3
"""
Create Checkpoint Utility
========================

This utility creates test checkpoints for debugging device compatibility issues.
It generates a simple checkpoint with known state that can be used for testing
checkpoint loading, device compatibility, and training resumption.

This utility is critical for creating reproducible test scenarios.
"""

import torch
import os
import sys
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from tests.utilities.test_utils import TestLogger
from app.models.game_transformer import GameTransformer
from app.models.model_config import ModelConfig

class CheckpointCreator:
    """Utility class for creating test checkpoints"""
    
    def __init__(self):
        self.logger = TestLogger()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def create_test_checkpoint(self) -> Path:
        """Create a simple test checkpoint"""
        try:
            self.logger.info("Creating test checkpoint...")
            
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
            model = GameTransformer(config)
            model.to(self.device)
            
            # Create optimizer (required for checkpoint compatibility)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
            
            # Create dummy training data
            dummy_board = torch.randint(0, 16, (1, 4, 4), device=self.device)
            
            # Test forward pass
            self.logger.info(f"Testing model on device: {self.device}")
            self.logger.info(f"Model parameters on device: {next(model.parameters()).device}")
            self.logger.info(f"Input tensor device: {dummy_board.device}")
            
            with torch.no_grad():
                policy_logits, value = model(dummy_board)
                self.logger.info(f"Policy logits device: {policy_logits.device}")
                self.logger.info(f"Value device: {value.device}")
            
            # Save checkpoint
            checkpoint_dir = Path(os.getenv('CHECKPOINTS_DIR', '../backend/checkpoints'))
            self.logger.info(f"Using checkpoint_dir: {checkpoint_dir}")
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
                'device': str(self.device)
            }
            
            checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.ok(f"Test checkpoint saved to {checkpoint_path}")
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Failed to create test checkpoint: {e}")
            raise
    
    def create_multiple_checkpoints(self, count: int = 3) -> list[Path]:
        """Create multiple test checkpoints with different states"""
        try:
            self.logger.info(f"Creating {count} test checkpoints...")
            
            checkpoint_paths = []
            for i in range(count):
                # Create checkpoint with different episode counts
                checkpoint_path = self.create_test_checkpoint_with_state(
                    episode_count=i + 1,
                    best_score=1000 + i * 100,
                    total_steps=i * 1000
                )
                checkpoint_paths.append(checkpoint_path)
            
            self.logger.ok(f"Created {count} test checkpoints")
            return checkpoint_paths
            
        except Exception as e:
            self.logger.error(f"Failed to create multiple checkpoints: {e}")
            raise
    
    def create_test_checkpoint_with_state(self, episode_count: int = 1, 
                                         best_score: int = 1000, 
                                         total_steps: int = 0) -> Path:
        """Create a test checkpoint with specific state values"""
        try:
            self.logger.info(f"Creating checkpoint with episode_count={episode_count}, best_score={best_score}")
            
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
            model = GameTransformer(config)
            model.to(self.device)
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
            
            # Save checkpoint
            checkpoint_dir = Path(os.getenv('CHECKPOINTS_DIR', '../backend/checkpoints'))
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'episode_count': episode_count,
                'total_steps': total_steps,
                'best_score': best_score,
                'loss_history': [(i, 0.5 - i * 0.01) for i in range(episode_count)],
                'score_history': [(i, best_score - i * 10) for i in range(episode_count)],
                'training_duration': 60.0 + episode_count * 10,
                'device': str(self.device)
            }
            
            checkpoint_path = checkpoint_dir / f"test_checkpoint_ep{episode_count}.pt"
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.ok(f"Checkpoint saved to {checkpoint_path}")
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint with state: {e}")
            raise

def main():
    """Main entry point for checkpoint creation utility"""
    logger = TestLogger()
    logger.banner("Checkpoint Creation Utility", 60)
    
    try:
        creator = CheckpointCreator()
        
        # Create a single test checkpoint
        checkpoint_path = creator.create_test_checkpoint()
        logger.ok(f"Created test checkpoint: {checkpoint_path}")
        
        # Create multiple checkpoints for testing
        logger.info("Creating additional test checkpoints...")
        additional_paths = creator.create_multiple_checkpoints(2)
        
        logger.success("CHECKPOINT CREATION COMPLETED!")
        logger.info(f"Total checkpoints created: {len(additional_paths) + 1}")
        
    except Exception as e:
        logger.error(f"Checkpoint creation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 