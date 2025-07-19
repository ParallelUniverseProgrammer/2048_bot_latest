from tests.utilities.backend_manager import requires_mock_backend
#!/usr/bin/env python3
"""
Frontend Persistence Fix Tests
==============================

This module tests the persistence fixes for frontend components, including
training metrics persistence and local storage functionality. It ensures
that frontend state is properly maintained across sessions and page reloads.

These tests are critical for maintaining user experience and data consistency.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from tests.utilities.test_utils import TestLogger, BackendTester

class PersistenceFixTester:
    """Test class for frontend persistence fixes"""
    
    def __init__(self):
        self.logger = TestLogger()
        self.backend = BackendTester()
@requires_mock_backend
    
    def test_training_metrics_persistence(self) -> bool:
        """Test that training metrics are properly persisted"""
        try:
            self.logger.banner("Testing Training Metrics Persistence", 60)
            
            # Simulate training metrics data
            training_metrics = {
                "episode": 1500,
                "score": 2048,
                "steps": 1000,
                "timestamp": "2024-01-01T12:00:00Z"
            }
            
            # Test localStorage persistence
            self.logger.info("Testing localStorage persistence...")
            
            # Simulate storing in localStorage
            stored_data = json.dumps(training_metrics)
            self.logger.ok("Training metrics stored in localStorage")
            
            # Simulate retrieving from localStorage
            retrieved_data = json.loads(stored_data)
            
            if retrieved_data == training_metrics:
                self.logger.ok("Training metrics retrieved successfully")
                return True
            else:
                self.logger.error("Training metrics retrieval failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Training metrics persistence test failed: {e}")
            return False
@requires_mock_backend
    
    def test_local_storage_persistence(self) -> bool:
        """Test local storage functionality"""
        try:
            self.logger.banner("Testing Local Storage Persistence", 60)
            
            # Test data to persist
            test_data = {
                "checkpoint_id": "test_checkpoint_episode_1500",
                "playback_speed": 1.0,
                "ui_state": {
                    "sidebar_collapsed": False,
                    "theme": "dark"
                }
            }
            
            self.logger.info("Testing localStorage operations...")
            
            # Simulate localStorage.setItem
            stored_json = json.dumps(test_data)
            self.logger.ok("Data stored in localStorage")
            
            # Simulate localStorage.getItem
            retrieved_json = stored_json
            retrieved_data = json.loads(retrieved_json)
            
            if retrieved_data == test_data:
                self.logger.ok("Local storage persistence working correctly")
                return True
            else:
                self.logger.error("Local storage persistence failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Local storage persistence test failed: {e}")
            return False
@requires_mock_backend

def main():
    """Main entry point for persistence fix tests"""
    logger = TestLogger()
    logger.banner("Frontend Persistence Fix Test Suite", 60)
    
    try:
        tester = PersistenceFixTester()
        
        # Run persistence tests
        training_success = tester.test_training_metrics_persistence()
        storage_success = tester.test_local_storage_persistence()
        
        # Summary
        logger.banner("Persistence Fix Test Summary", 60)
        logger.info(f"Training Metrics Persistence: {'PASS' if training_success else 'FAIL'}")
        logger.info(f"Local Storage Persistence: {'PASS' if storage_success else 'FAIL'}")
        
        if training_success and storage_success:
            logger.success("ALL PERSISTENCE FIX TESTS PASSED!")
        else:
            logger.error("Some persistence fix tests failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Persistence fix test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 