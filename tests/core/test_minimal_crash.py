#!/usr/bin/env python3
"""
Minimal Crash Test Suite
========================

This test suite verifies that the core system components can be created and run
without crashing. It tests the minimal functionality required for the system to work.

The test covers:
- Model creation and configuration
- Environment initialization
- Trainer creation and setup
- Load balancing functionality
- Single episode execution
"""

import sys
import os
import time
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from tests.utilities.test_utils import TestLogger

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend')
sys.path.insert(0, backend_path)

try:
    from app.models.model_config import ModelConfig
    from app.environment.gym_2048_env import Gym2048Env
    from app.training.ppo_trainer import PPOTrainer
    from app.utils.action_selection import select_action_with_fallback
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info(f"Backend path: {backend_path}")
    logger.info(f"Available in backend: {os.listdir(backend_path) if os.path.exists(backend_path) else 'Path does not exist'}")
    raise

def test_model_creation():
    """Test that model can be created without crashing"""
    logger = TestLogger()
    logger.testing("Testing model creation")
    
    try:
        # Create minimal model config
        config = ModelConfig(
            input_size=16,
            hidden_size=64,
            num_layers=2,
            output_size=4,
            dropout=0.1
        )
        
        # Create model
        model = config.create_model()
        
        # Test basic model operations
        import torch
        test_input = torch.randn(1, 16)
        output = model(test_input)
        
        assert output.shape == (1, 4)
        logger.ok("Model creation and forward pass successful")
        return True
        
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        return False

def test_environment():
    """Test that environment can be created and used without crashing"""
    logger = TestLogger()
    logger.testing("Testing environment creation")
    
    try:
        # Create environment
        env = Gym2048Env()
        
        # Test basic environment operations
        state = env.reset()
        assert state is not None
        
        # Test action selection
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        
        assert next_state is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        
        logger.ok("Environment creation and basic operations successful")
        return True
        
    except Exception as e:
        logger.error(f"Environment creation failed: {e}")
        return False

def test_trainer_creation():
    """Test that trainer can be created without crashing"""
    logger = TestLogger()
    logger.testing("Testing trainer creation")
    
    try:
        # Create minimal components
        config = ModelConfig(
            input_size=16,
            hidden_size=64,
            num_layers=2,
            output_size=4,
            dropout=0.1
        )
        
        env = Gym2048Env()
        
        # Create trainer
        trainer = PPOTrainer(
            model_config=config,
            env=env,
            learning_rate=1e-4,
            batch_size=32,
            epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01
        )
        
        logger.ok("Trainer creation successful")
        return True
        
    except Exception as e:
        logger.error(f"Trainer creation failed: {e}")
        return False

def test_load_balancing():
    """Test that load balancing functionality works without crashing"""
    logger = TestLogger()
    logger.testing("Testing load balancing functionality")
    
    try:
        # Test action selection with fallback
        import torch
        import numpy as np
        
        # Create mock model output
        action_probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        legal_actions = [0, 1, 2, 3]
        
        # Test action selection
        action, probs, attention = select_action_with_fallback(
            action_probs, legal_actions, device='cpu'
        )
        
        assert action in legal_actions
        assert probs is not None
        assert attention is not None
        
        logger.ok("Load balancing functionality successful")
        return True
        
    except Exception as e:
        logger.error(f"Load balancing test failed: {e}")
        return False

def test_single_episode():
    """Test that a single episode can be run without crashing"""
    logger = TestLogger()
    logger.testing("Testing single episode execution")
    
    try:
        # Create minimal setup
        config = ModelConfig(
            input_size=16,
            hidden_size=64,
            num_layers=2,
            output_size=4,
            dropout=0.1
        )
        
        env = Gym2048Env()
        model = config.create_model()
        
        # Run single episode
        state = env.reset()
        done = False
        steps = 0
        max_steps = 100  # Prevent infinite loops
        
        while not done and steps < max_steps:
            # Get model prediction
            import torch
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = model(state_tensor)
            
            # Select action
            action = action_probs.argmax().item()
            
            # Take step
            state, reward, done, truncated, info = env.step(action)
            steps += 1
        
        logger.ok(f"Single episode completed in {steps} steps")
        return True
        
    except Exception as e:
        logger.error(f"Single episode test failed: {e}")
        return False

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("Minimal Crash Test Suite", 60)
    
    # Run all tests
    tests = [
        ("Model Creation", test_model_creation),
        ("Environment", test_environment),
        ("Trainer Creation", test_trainer_creation),
        ("Load Balancing", test_load_balancing),
        ("Single Episode", test_single_episode),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.testing(f"Running {test_name} test...")
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"{test_name} test failed with exception: {e}")
            failed += 1
    
    logger.separator(60)
    if failed == 0:
        logger.success(f"All {passed} tests passed! System is stable.")
        sys.exit(0)
    else:
        logger.error(f"{failed} tests failed, {passed} tests passed")
        sys.exit(1)

if __name__ == "__main__":
    main() 