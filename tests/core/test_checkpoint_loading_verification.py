#!/usr/bin/env python3
"""
Test script to verify checkpoint loading fix works correctly
"""

import sys
import os
import torch
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.training.training_manager import TrainingManager
from app.api.websocket_manager import WebSocketManager
from app.models.model_config import DynamicModelConfig
from app.training.ppo_trainer import PPOTrainer

def test_checkpoint_loading_fix():
    """Test that checkpoint loading properly updates trainer configuration"""
    print("🧪 Testing checkpoint loading fix...")
    
    # Create a mock WebSocket manager
    ws_manager = WebSocketManager()
    
    # Create training manager
    training_manager = TrainingManager(ws_manager, n_envs=2)
    
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
        checkpoint_path = tmp_file.name
    
    try:
        # Save checkpoint
        test_trainer.save_checkpoint(checkpoint_path)
        print(f"✅ Created test checkpoint at: {checkpoint_path}")
        
        # Test the new load_checkpoint_trainer method
        print("\n🔄 Testing load_checkpoint_trainer method...")
        
        # Load checkpoint using the new method
        training_manager.load_checkpoint_trainer(config, checkpoint_path)
        
        # Verify the results
        print(f"   Current episode: {training_manager.current_episode}")
        print(f"   Trainer episode count: {training_manager.trainer.episode_count}")
        print(f"   Best score: {training_manager.trainer.best_score}")
        print(f"   Total steps: {training_manager.trainer.total_steps}")
        print(f"   Number of env trainers: {len(training_manager.env_trainers)}")
        print(f"   All env trainers reference same trainer: {all(env_trainer is training_manager.trainer for env_trainer in training_manager.env_trainers)}")
        
        # Verify that the episode count was properly restored
        assert training_manager.current_episode == 150, f"Expected episode 150, got {training_manager.current_episode}"
        assert training_manager.trainer.episode_count == 150, f"Expected trainer episode 150, got {training_manager.trainer.episode_count}"
        assert training_manager.trainer.best_score == 2048, f"Expected best score 2048, got {training_manager.trainer.best_score}"
        assert len(training_manager.env_trainers) == 2, f"Expected 2 env trainers, got {len(training_manager.env_trainers)}"
        assert all(env_trainer is training_manager.trainer for env_trainer in training_manager.env_trainers), "Not all env trainers reference the same trainer"
        
        print("✅ All assertions passed! Checkpoint loading fix is working correctly.")
        
        # Test that training can start from the loaded checkpoint
        print("\n🚀 Testing training start from loaded checkpoint...")
        training_manager.start()
        
        # Verify training state
        assert training_manager.is_training == True, "Training should be active"
        assert training_manager.current_episode == 150, "Should start from episode 150"
        
        print("✅ Training started successfully from loaded checkpoint!")
        
        # Stop training
        training_manager.stop_sync()
        
    finally:
        # Clean up
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)
        training_manager.cleanup()
    
    print("\n🎉 Checkpoint loading fix verification completed successfully!")

if __name__ == "__main__":
    test_checkpoint_loading_fix() 