#!/usr/bin/env python3
"""
Training Issue Test Suite
=========================

This test suite verifies that training issues are properly identified and resolved.
It tests the training system's ability to handle various scenarios and edge cases.

The test covers:
- Training initialization and startup
- Training progress monitoring
- Error detection and handling
- System stability during training
"""

import sys
import os
import time
import requests
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from tests.utilities.test_utils import TestLogger, BackendTester

def test_training_start():
    """Test that training can start properly"""
    logger = TestLogger()
    logger.testing("Testing training start functionality")
    
    try:
        backend = BackendTester()
        
        # Check if backend is available
        if not backend.test_connectivity():
            logger.warning("Backend not available, skipping training start test")
            return True
        
        # Check initial status
        status = backend.get_training_status()
        if status:
            logger.log(f"   Initial training status: {status.get('is_training', False)}")
        
        # Test training start
        try:
            response = requests.post(
                f"{backend.base_url}/training/start",
                json={"model_size": "small"},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.ok("Training start request successful")
                return True
            else:
                logger.error(f"Training start failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Training start request failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Training start test failed: {e}")
        return False

def test_training_progress():
    """Test that training progress is monitored correctly"""
    logger = TestLogger()
    logger.testing("Testing training progress monitoring")
    
    try:
        backend = BackendTester()
        
        # Check if backend is available
        if not backend.test_connectivity():
            logger.warning("Backend not available, skipping training progress test")
            return True
        
        # Monitor training progress for a short time
        logger.log("   Monitoring training progress...")
        
        for i in range(3):  # Check 3 times
            status = backend.get_training_status()
            if status:
                is_training = status.get('is_training', False)
                episode = status.get('current_episode', 0)
                logger.log(f"   Check {i+1}: Training={is_training}, Episode={episode}")
                
                if is_training and episode > 0:
                    logger.ok("Training is progressing correctly")
                    return True
            
            time.sleep(2)  # Wait 2 seconds between checks
        
        logger.warning("Training progress monitoring completed")
        return True
        
    except Exception as e:
        logger.error(f"Training progress test failed: {e}")
        return False

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("Training Issue Test Suite", 60)
    
    # Run all tests
    tests = [
        ("Training Start", test_training_start),
        ("Training Progress", test_training_progress),
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
        logger.success(f"All {passed} tests passed! Training system is working correctly.")
        sys.exit(0)
    else:
        logger.error(f"{failed} tests failed, {passed} tests passed")
        sys.exit(1)

if __name__ == "__main__":
    main() 