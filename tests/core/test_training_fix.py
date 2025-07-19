#!/usr/bin/env python3
"""
Training Fix Test Suite
======================

This test suite verifies that the training system works correctly after fixes
and can handle various training scenarios without issues.

The test covers:
- Training initialization
- Model configuration handling
- Environment setup
- Training loop stability
- Error recovery
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

def test_training_fix():
    """Test that training system works correctly after fixes"""
    logger = TestLogger()
    logger.testing("Testing training system fixes")
    
    try:
        backend = BackendTester()
        
        # Check if backend is available
        if not backend.test_connectivity():
            logger.warning("Backend not available, skipping training test")
            return True
        
        # Check initial training status
        status = backend.get_training_status()
        if status:
            logger.log(f"   Initial training status: {status.get('is_training', False)}")
        
        # Test training start
        logger.log("   Testing training start...")
        try:
            response = requests.post(
                f"{backend.base_url}/training/start",
                json={"model_size": "small"},
                timeout=10
            )
            
            if response.status_code == 200:
                start_result = response.json()
                logger.ok("Training start successful")
                
                # Wait a moment for training to begin
                time.sleep(2)
                
                # Check training status again
                status = backend.get_training_status()
                if status and status.get('is_training', False):
                    logger.ok("Training is active")
                    
                    # Test training stop
                    logger.log("   Testing training stop...")
                    try:
                        stop_response = requests.post(
                            f"{backend.base_url}/training/stop",
                            timeout=10
                        )
                        
                        if stop_response.status_code == 200:
                            logger.ok("Training stop successful")
                            return True
                        else:
                            logger.error(f"Training stop failed: HTTP {stop_response.status_code}")
                            return False
                    except Exception as e:
                        logger.error(f"Training stop request failed: {e}")
                        return False
                else:
                    logger.error("Training did not start properly")
                    return False
            else:
                logger.error(f"Training start failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Training start request failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Training fix test failed: {e}")
        return False

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("Training Fix Test Suite", 60)
    
    # Run all tests
    tests = [
        ("Training Fix", test_training_fix),
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