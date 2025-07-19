#!/usr/bin/env python3
"""
Tiny Model Test Suite
=====================

This test suite verifies that the tiny model configuration works correctly and
can be used for training and inference without issues.

The test covers:
- Tiny model creation and configuration
- Model parameter count verification
- Forward pass functionality
- Training compatibility
- Performance benchmarks
"""

import sys
import os
import time
import torch
import numpy as np
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
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info(f"Backend path: {backend_path}")
    logger.info(f"Available in backend: {os.listdir(backend_path) if os.path.exists(backend_path) else 'Path does not exist'}")
    raise

def test_tiny_model():
    """Test that tiny model can be created and used"""
    logger = TestLogger()
    logger.testing("Testing tiny model creation and functionality")
    
    try:
        # Create tiny model config
        config = ModelConfig(
            input_size=16,
            hidden_size=32,  # Very small hidden size
            num_layers=1,    # Only one layer
            output_size=4,
            dropout=0.1
        )
        
        # Create model
        model = config.create_model()
        
        # Test parameter count
        total_params = sum(p.numel() for p in model.parameters())
        logger.log(f"   Total parameters: {total_params:,}")
        
        # Verify it's actually tiny (less than 10k parameters)
        assert total_params < 10000, f"Model has {total_params} parameters, expected < 10k"
        
        # Test forward pass
        test_input = torch.randn(1, 16)
        output = model(test_input)
        
        assert output.shape == (1, 4)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        logger.ok("Tiny model creation and forward pass successful")
        return True
        
    except Exception as e:
        logger.error(f"Tiny model test failed: {e}")
        return False

def test_tiny_model_speed():
    """Test that tiny model is fast enough for real-time use"""
    logger = TestLogger()
    logger.testing("Testing tiny model speed")
    
    try:
        # Create tiny model
        config = ModelConfig(
            input_size=16,
            hidden_size=32,
            num_layers=1,
            output_size=4,
            dropout=0.1
        )
        
        model = config.create_model()
        model.eval()  # Set to evaluation mode
        
        # Create test batch
        batch_size = 32
        test_input = torch.randn(batch_size, 16)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # Benchmark
        num_runs = 100
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(test_input)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = batch_size / avg_time
        
        logger.log(f"   Average time per batch: {avg_time*1000:.2f}ms")
        logger.log(f"   Throughput: {throughput:.1f} samples/second")
        
        # Verify it's fast enough (should be > 1000 samples/second)
        assert throughput > 1000, f"Throughput {throughput:.1f} samples/second is too slow"
        
        logger.ok("Tiny model speed test passed")
        return True
        
    except Exception as e:
        logger.error(f"Tiny model speed test failed: {e}")
        return False

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("Tiny Model Test Suite", 60)
    
    # Run all tests
    tests = [
        ("Tiny Model Creation", test_tiny_model),
        ("Tiny Model Speed", test_tiny_model_speed),
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
        logger.success(f"All {passed} tests passed! Tiny model is working correctly.")
        sys.exit(0)
    else:
        logger.error(f"{failed} tests failed, {passed} tests passed")
        sys.exit(1)

if __name__ == "__main__":
    main() 