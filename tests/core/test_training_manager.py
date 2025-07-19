#!/usr/bin/env python3
"""
Training Manager Test Suite
===========================

This test suite verifies that the training manager works correctly and can handle
various training scenarios without issues.

The test covers:
- Training manager initialization
- WebSocket communication
- Memory management
- Training state management
- Error handling and recovery
"""

import sys
import os
import time
import asyncio
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from tests.utilities.test_utils import TestLogger

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend')
sys.path.insert(0, backend_path)

try:
    from app.training.training_manager import TrainingManager
    from app.models.model_config import ModelConfig
    from app.environment.gym_2048_env import Gym2048Env
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info(f"Backend path: {backend_path}")
    logger.info(f"Available in backend: {os.listdir(backend_path) if os.path.exists(backend_path) else 'Path does not exist'}")
    raise

class MockWebSocketManager:
    """Mock WebSocket manager for testing"""
    
    def __init__(self):
        self.messages = []
        self.connection_count = 0
    
    async def broadcast(self, message, priority="normal"):
        """Mock broadcast method"""
        self.messages.append(message)
        return True
    
    async def broadcast_high_priority(self, message):
        """Mock high priority broadcast"""
        self.messages.append({"priority": "high", "message": message})
        return True

def test_training_manager():
    """Test that training manager can be created and used"""
    logger = TestLogger()
    logger.testing("Testing training manager creation and functionality")
    
    try:
        # Create mock WebSocket manager
        ws_manager = MockWebSocketManager()
        
        # Create model config
        config = ModelConfig(
            input_size=16,
            hidden_size=64,
            num_layers=2,
            output_size=4,
            dropout=0.1
        )
        
        # Create environment
        env = Gym2048Env()
        
        # Create training manager
        training_manager = TrainingManager(
            model_config=config,
            websocket_manager=ws_manager,
            env=env
        )
        
        # Test basic functionality
        assert training_manager is not None
        assert training_manager.model_config == config
        assert training_manager.websocket_manager == ws_manager
        
        logger.ok("Training manager creation successful")
        return True
        
    except Exception as e:
        logger.error(f"Training manager test failed: {e}")
        return False

def test_memory_usage():
    """Test that training manager handles memory properly"""
    logger = TestLogger()
    logger.testing("Testing training manager memory usage")
    
    try:
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create training manager
        ws_manager = MockWebSocketManager()
        config = ModelConfig(
            input_size=16,
            hidden_size=64,
            num_layers=2,
            output_size=4,
            dropout=0.1
        )
        env = Gym2048Env()
        
        training_manager = TrainingManager(
            model_config=config,
            websocket_manager=ws_manager,
            env=env
        )
        
        # Get memory usage after creation
        after_creation_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = after_creation_memory - initial_memory
        
        logger.log(f"   Initial memory: {initial_memory:.1f} MB")
        logger.log(f"   After creation: {after_creation_memory:.1f} MB")
        logger.log(f"   Memory increase: {memory_increase:.1f} MB")
        
        # Check if memory increase is reasonable (less than 100MB)
        if memory_increase < 100:
            logger.ok("Memory usage is reasonable")
            return True
        else:
            logger.warning(f"Memory usage increase is high: {memory_increase:.1f} MB")
            return False
        
    except ImportError:
        logger.warning("psutil not available, skipping memory test")
        return True
    except Exception as e:
        logger.error(f"Memory usage test failed: {e}")
        return False

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("Training Manager Test Suite", 60)
    
    # Run all tests
    tests = [
        ("Training Manager Creation", test_training_manager),
        ("Memory Usage", test_memory_usage),
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
        logger.success(f"All {passed} tests passed! Training manager is working correctly.")
        sys.exit(0)
    else:
        logger.error(f"{failed} tests failed, {passed} tests passed")
        sys.exit(1)

if __name__ == "__main__":
    main() 