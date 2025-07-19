from tests.utilities.backend_manager import requires_real_backend
#!/usr/bin/env python3
"""
Checkpoint Loading Verification Test
===================================

This test verifies that the checkpoint loading fix works correctly by:
- Creating a test checkpoint with known state
- Loading it using the new load_checkpoint_trainer method
- Verifying all state is properly restored
- Testing that training can start from the loaded checkpoint

This test is critical for ensuring checkpoint loading reliability.
"""

import sys
import os
import torch
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tests.utilities.test_utils import TestLogger
from app.training.training_manager import TrainingManager
from app.api.websocket_manager import WebSocketManager
from app.models.model_config import DynamicModelConfig
from app.training.ppo_trainer import PPOTrainer

class CheckpointVerificationTester:
    """Test class for verifying checkpoint loading functionality"""
    
    def __init__(self):
        self.logger = TestLogger()
        self.training_manager = None
        self.checkpoint_path = None
@requires_real_backend
    
    def test_checkpoint_loading_fix(self) -> bool:
        """Test that checkpoint loading properly updates trainer configuration"""
        try:
            self.logger.banner("Testing Checkpoint Loading Fix", 60)
            
            # Create a mock WebSocket manager
            ws_manager = WebSocketManager()
            
            # Create training manager
            self.training_manager = TrainingManager(ws_manager, n_envs=2)
            
            # Create a test config
            config = DynamicModelConfig.select_config(target_vram=2.0)  # Small model
            
            # Create a test trainer and save a checkpoint
            test_trainer = PPOTrainer(config=config, learning_rate=0.0003)
            
            # Simulate some training progress
            test_trainer.episode_count = 150
            test_trainer.best_score = 2048
            test_trainer.total_steps = 5000
            
            # Create a temporary checkpoint file
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
                self.checkpoint_path = tmp_file.name
            
            # Save checkpoint
            test_trainer.save_checkpoint(self.checkpoint_path)
            self.logger.ok(f"Created test checkpoint at: {self.checkpoint_path}")
            
            # Test the new load_checkpoint_trainer method
            self.logger.info("Testing load_checkpoint_trainer method...")
            
            # Load checkpoint using the new method
            self.training_manager.load_checkpoint_trainer(config, self.checkpoint_path)
            
            # Verify the results
            self.logger.info(f"Current episode: {self.training_manager.current_episode}")
            self.logger.info(f"Trainer episode count: {self.training_manager.trainer.episode_count}")
            self.logger.info(f"Best score: {self.training_manager.trainer.best_score}")
            self.logger.info(f"Total steps: {self.training_manager.trainer.total_steps}")
            self.logger.info(f"Number of env trainers: {len(self.training_manager.env_trainers)}")
            self.logger.info(f"All env trainers reference same trainer: {all(env_trainer is self.training_manager.trainer for env_trainer in self.training_manager.env_trainers)}")
            
            # Verify that the episode count was properly restored
            assert self.training_manager.current_episode == 150, f"Expected episode 150, got {self.training_manager.current_episode}"
            assert self.training_manager.trainer.episode_count == 150, f"Expected trainer episode 150, got {self.training_manager.trainer.episode_count}"
            assert self.training_manager.trainer.best_score == 2048, f"Expected best score 2048, got {self.training_manager.trainer.best_score}"
            assert len(self.training_manager.env_trainers) == 2, f"Expected 2 env trainers, got {len(self.training_manager.env_trainers)}"
            assert all(env_trainer is self.training_manager.trainer for env_trainer in self.training_manager.env_trainers), "Not all env trainers reference the same trainer"
            
            self.logger.ok("All assertions passed! Checkpoint loading fix is working correctly.")
            
            # Test that training can start from the loaded checkpoint
            self.logger.info("Testing training start from loaded checkpoint...")
            self.training_manager.start()
            
            # Verify training state
            assert self.training_manager.is_training == True, "Training should be active"
            assert self.training_manager.current_episode == 150, "Should start from episode 150"
            
            self.logger.ok("Training started successfully from loaded checkpoint!")
            
            # Stop training
            self.training_manager.stop_sync()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Checkpoint loading verification failed: {e}")
            return False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Clean up checkpoint file
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                os.unlink(self.checkpoint_path)
            
            # Clean up training manager
            if self.training_manager:
                self.training_manager.cleanup()
                
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
@requires_real_backend

def main():
    """Main entry point for checkpoint loading verification test"""
    logger = TestLogger()
    logger.banner("Checkpoint Loading Verification Test Suite", 60)
    
    tester = CheckpointVerificationTester()
    success = tester.test_checkpoint_loading_fix()
    
    if success:
        logger.success("CHECKPOINT LOADING VERIFICATION TEST PASSED!")
    else:
        logger.error("CHECKPOINT LOADING VERIFICATION TEST FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main() 