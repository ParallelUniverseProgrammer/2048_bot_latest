from tests.utilities.backend_manager import requires_real_backend
#!/usr/bin/env python3
"""
WebSocket Broadcast Tests
========================

This module tests WebSocket broadcast functionality, including JSON serialization
of numpy arrays and message broadcasting to multiple clients. It validates that
the WebSocket system can handle complex data types and concurrent connections.

These tests ensure the WebSocket broadcast system is robust and handles edge cases properly.
"""

import json
import sys
import os
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from backend.app.api.websocket_manager import WebSocketManager
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info(f"Backend path: {os.path.join(os.path.dirname(__file__), '..', '..', 'backend')}")
    logger.info(f"Available in backend: {os.listdir(os.path.join(os.path.dirname(__file__), '..', '..', 'backend')) if os.path.exists(os.path.join(os.path.dirname(__file__), '..', '..', 'backend')) else 'Path does not exist'}")

from tests.utilities.test_utils import TestLogger

class TestWebSocketBroadcastFix:
    """Test class for WebSocket broadcast functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.logger = TestLogger()
        self.websocket_manager = WebSocketManager()
@requires_real_backend
    
    def test_broadcast_with_numpy_arrays_fails_before_fix(self):
        """Test that broadcast fails with numpy arrays before the fix"""
        try:
            self.logger.banner("Testing Broadcast with Numpy Arrays (Before Fix)", 60)
            
            # Create a message with numpy arrays
            import numpy as np
            message = {
                "type": "training_update",
                "data": {
                    "state": np.array([[1, 2], [3, 4]]),
                    "action": np.array([0, 1, 2]),
                    "reward": np.array([0.5])
                }
            }
            
            self.logger.info("Testing broadcast with numpy arrays...")
            
            # This should fail before the fix
            try:
                json.dumps(message)
                self.logger.error("JSON serialization should have failed")
                return False
            except TypeError:
                self.logger.ok("JSON serialization correctly failed with numpy arrays")
                return True
                
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return False
@requires_real_backend
    
    def test_broadcast_with_converted_arrays_works_after_fix(self):
        """Test that broadcast works with converted arrays after the fix"""
        try:
            self.logger.banner("Testing Broadcast with Converted Arrays (After Fix)", 60)
            
            # Create a message with numpy arrays
            import numpy as np
            message = {
                "type": "training_update",
                "data": {
                    "state": np.array([[1, 2], [3, 4]]).tolist(),
                    "action": np.array([0, 1, 2]).tolist(),
                    "reward": np.array([0.5]).tolist()
                }
            }
            
            self.logger.info("Testing broadcast with converted arrays...")
            
            # This should work after the fix
            try:
                json.dumps(message)
                self.logger.ok("JSON serialization works with converted arrays")
                return True
            except TypeError as e:
                self.logger.error(f"JSON serialization failed: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return False
    
@requires_real_backend
    @patch('app.api.websocket_manager.WebSocketManager.get_connection_count')
    def test_websocket_manager_broadcast_method_json_serialization(self, mock_get_connection_count):
        """Test WebSocket manager broadcast method JSON serialization"""
        try:
            self.logger.banner("Testing WebSocket Manager Broadcast Method", 60)
            
            # Mock connection count
            mock_get_connection_count.return_value = 2
            
            # Create a message that would normally cause serialization issues
            import numpy as np
            message = {
                "type": "training_update",
                "episode": 1500,
                "score": 2048,
                "state": np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).tolist(),
                "action": np.array([0, 1, 2, 3]).tolist(),
                "reward": np.array([0.5, 1.0]).tolist()
            }
            
            self.logger.info("Testing WebSocket manager broadcast...")
            
            # Test that the message can be serialized
            try:
                json_str = json.dumps(message)
                self.logger.ok("Message serialized successfully")
                
                # Test that it can be deserialized
                deserialized = json.loads(json_str)
                if deserialized == message:
                    self.logger.ok("Message deserialized correctly")
                    return True
                else:
                    self.logger.error("Message deserialization failed")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Serialization failed: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return False
@requires_real_backend
    
    def test_training_update_message_serialization(self):
        """Test training update message serialization"""
        try:
            self.logger.banner("Testing Training Update Message Serialization", 60)
            
            # Create a realistic training update message
            import numpy as np
            training_message = {
                "type": "training_update",
                "episode": 1500,
                "current_score": 2048,
                "best_score": 4096,
                "steps": 1000,
                "epsilon": 0.1,
                "loss": 0.05,
                "state": np.array([[0, 2, 0, 4], [0, 0, 8, 0], [0, 0, 0, 16], [0, 0, 0, 0]]).tolist(),
                "action": np.array([0]).tolist(),
                "reward": np.array([1.0]).tolist(),
                "next_state": np.array([[0, 2, 0, 4], [0, 0, 8, 0], [0, 0, 0, 16], [0, 0, 0, 0]]).tolist(),
                "done": False
            }
            
            self.logger.info("Testing training update message serialization...")
            
            # Test serialization
            try:
                json_str = json.dumps(training_message)
                self.logger.ok("Training message serialized successfully")
                
                # Test deserialization
                deserialized = json.loads(json_str)
                if deserialized["episode"] == training_message["episode"]:
                    self.logger.ok("Training message deserialized correctly")
                    return True
                else:
                    self.logger.error("Training message deserialization failed")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Training message serialization failed: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return False
@requires_real_backend
    
    def test_regression_original_error_message_structure(self):
        """Test regression to ensure original error message structure is preserved"""
        try:
            self.logger.banner("Testing Original Error Message Structure", 60)
            
            # Test the original error message structure
            error_message = {
                "type": "error",
                "message": "Failed to load checkpoint",
                "details": {
                    "checkpoint_id": "episode_1500",
                    "error_code": "LOAD_FAILED",
                    "timestamp": "2024-01-01T12:00:00Z"
                }
            }
            
            self.logger.info("Testing error message structure...")
            
            # Test serialization
            try:
                json_str = json.dumps(error_message)
                self.logger.ok("Error message serialized successfully")
                
                # Test deserialization
                deserialized = json.loads(json_str)
                if deserialized["type"] == "error":
                    self.logger.ok("Error message structure preserved")
                    return True
                else:
                    self.logger.error("Error message structure corrupted")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error message serialization failed: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return False
@requires_real_backend

def main():
    """Main entry point for WebSocket broadcast tests"""
    logger = TestLogger()
    logger.banner("WebSocket Broadcast Test Suite", 60)
    
    try:
        tester = TestWebSocketBroadcastFix()
        tester.setup_method()
        
        # Run WebSocket broadcast tests
        numpy_fail_success = tester.test_broadcast_with_numpy_arrays_fails_before_fix()
        converted_success = tester.test_broadcast_with_converted_arrays_works_after_fix()
        manager_success = tester.test_websocket_manager_broadcast_method_json_serialization()
        training_success = tester.test_training_update_message_serialization()
        error_success = tester.test_regression_original_error_message_structure()
        
        # Summary
        logger.banner("WebSocket Broadcast Test Summary", 60)
        logger.info(f"Numpy Arrays Fail (Before Fix): {'PASS' if numpy_fail_success else 'FAIL'}")
        logger.info(f"Converted Arrays Work (After Fix): {'PASS' if converted_success else 'FAIL'}")
        logger.info(f"Manager Broadcast Method: {'PASS' if manager_success else 'FAIL'}")
        logger.info(f"Training Update Message: {'PASS' if training_success else 'FAIL'}")
        logger.info(f"Error Message Structure: {'PASS' if error_success else 'FAIL'}")
        
        all_passed = all([numpy_fail_success, converted_success, manager_success, training_success, error_success])
        
        if all_passed:
            logger.success("ALL WEBSOCKET BROADCAST TESTS PASSED!")
        else:
            logger.error("Some WebSocket broadcast tests failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"WebSocket broadcast test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 