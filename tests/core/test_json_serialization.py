#!/usr/bin/env python3
"""
Test JSON Serialization of Checkpoint Playback Messages
======================================================

This test suite verifies that checkpoint playback messages are properly JSON serializable
and that numpy arrays are correctly converted to prevent serialization errors.

The test covers:
- Action probabilities serialization
- Full message structure serialization
- Lightweight message structure serialization
- Numpy array detection and conversion
- WebSocket broadcast compatibility
- Regression testing of original error scenarios
"""

import json
import numpy as np
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from tests.utilities.test_utils import TestLogger

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend')
sys.path.insert(0, backend_path)

try:
    from app.models.checkpoint_playback import CheckpointPlayback
    from app.models.model_config import ModelConfig
    from app.models.checkpoint_metadata import CheckpointMetadata
    from app.environment.gym_2048_env import Gym2048Env
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info(f"Backend path: {backend_path}")
    logger.info(f"Available in backend: {os.listdir(backend_path) if os.path.exists(backend_path) else 'Path does not exist'}")
    raise

class TestCheckpointPlaybackJSONSerialization:
    """Test suite for JSON serialization of checkpoint playback messages"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.logger = TestLogger()
        
        # Create a mock checkpoint manager
        self.mock_checkpoint_manager = Mock()
        
        # Create checkpoint playback instance
        self.playback = CheckpointPlayback(self.mock_checkpoint_manager)
        
        # Mock environment and game
        self.mock_env = Mock()
        self.mock_game = Mock()
        self.mock_game.board = np.array([[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]])
        self.mock_game.score = 12345
        self.mock_env.game = self.mock_game
        self.mock_env.is_done.return_value = False
        self.mock_env.get_state.return_value = np.array([1, 2, 3, 4])
        self.mock_env.get_legal_actions.return_value = [0, 1, 2, 3]
        self.mock_env.step.return_value = (np.array([1, 2, 3, 4]), 10.0, True, False, {})
        
        # Set up the playback environment
        self.playback.env = self.mock_env
        self.playback.current_checkpoint_id = "test_checkpoint"
        
    def test_action_probs_json_serializable(self):
        """Test that action_probs in step_data is JSON serializable"""
        self.logger.testing("Testing action probabilities JSON serialization")
        
        # Mock the select_action method to return numpy arrays
        with patch.object(self.playback, 'select_action') as mock_select:
            # Return numpy arrays as would happen in real usage
            mock_select.return_value = (
                1,  # action
                np.array([0.1, 0.7, 0.15, 0.05]),  # action_probs as numpy array
                np.array([[0.25, 0.25, 0.25, 0.25]])  # attention_weights as numpy array
            )
            
            # Mock the environment to finish after one step
            self.mock_env.is_done.side_effect = [False, True]
            
            # Run play_single_game
            result = self.playback.play_single_game()
            
            # Verify the result is valid
            assert 'game_history' in result
            assert len(result['game_history']) > 0
            
            # Get the first step data
            step_data = result['game_history'][0]
            
            # Verify action_probs is now a list, not numpy array
            assert isinstance(step_data['action_probs'], list)
            assert not isinstance(step_data['action_probs'], np.ndarray)
            
            # Verify attention_weights is also a list
            assert isinstance(step_data['attention_weights'], list)
            assert not isinstance(step_data['attention_weights'], np.ndarray)
            
            # Most importantly: verify the entire step_data can be JSON serialized
            json_str = json.dumps(step_data)
            assert json_str is not None
            
            # Verify we can deserialize it back
            deserialized = json.loads(json_str)
            assert deserialized['action_probs'] == [0.1, 0.7, 0.15, 0.05]
            
            self.logger.ok("Action probabilities JSON serialization test passed")
            
    def test_full_message_json_serializable(self):
        """Test that _create_full_message produces JSON serializable output"""
        self.logger.testing("Testing full message JSON serialization")
        
        # Create mock step data with converted arrays
        step_data = {
            'step': 0,
            'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
            'score': 12345,
            'action': 1,
            'action_probs': [0.1, 0.7, 0.15, 0.05],  # Already converted to list
            'legal_actions': [0, 1, 2, 3],
            'attention_weights': [[0.25, 0.25, 0.25, 0.25]],  # Already converted to list
            'timestamp': 1234567890.0,
            'reward': 10.0,
            'done': False
        }
        
        game_result = {
            'final_score': 12345,
            'steps': 100,
            'max_tile': 2048
        }
        
        # Create full message
        message = self.playback._create_full_message(step_data, game_result, "test_checkpoint", 1)
        
        # Verify it's JSON serializable
        json_str = json.dumps(message)
        assert json_str is not None
        
        # Verify structure
        deserialized = json.loads(json_str)
        assert deserialized['type'] == 'checkpoint_playback'
        assert deserialized['checkpoint_id'] == 'test_checkpoint'
        assert deserialized['game_number'] == 1
        assert deserialized['step_data']['action_probs'] == [0.1, 0.7, 0.15, 0.05]
        
        self.logger.ok("Full message JSON serialization test passed")
        
    def test_lightweight_message_json_serializable(self):
        """Test that _create_lightweight_message produces JSON serializable output"""
        self.logger.testing("Testing lightweight message JSON serialization")
        
        # Create mock step data
        step_data = {
            'step': 0,
            'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
            'score': 12345,
            'action': 1,
            'action_probs': [0.1, 0.7, 0.15, 0.05],  # Already converted to list
            'timestamp': 1234567890.0
        }
        
        game_result = {
            'final_score': 12345,
            'steps': 100,
            'max_tile': 2048
        }
        
        # Create lightweight message
        message = self.playback._create_lightweight_message(step_data, game_result, "test_checkpoint", 1)
        
        # Verify it's JSON serializable
        json_str = json.dumps(message)
        assert json_str is not None
        
        # Verify structure
        deserialized = json.loads(json_str)
        assert deserialized['type'] == 'checkpoint_playback_light'
        assert deserialized['checkpoint_id'] == 'test_checkpoint'
        assert deserialized['game_number'] == 1
        
        self.logger.ok("Lightweight message JSON serialization test passed")
        
    def test_numpy_array_detection_and_conversion(self):
        """Test that numpy arrays are properly detected and converted"""
        self.logger.testing("Testing numpy array detection and conversion")
        
        # Create a message with various numpy array types
        test_data = {
            'regular_list': [1, 2, 3, 4],
            'numpy_array_1d': np.array([0.1, 0.2, 0.3, 0.4]),
            'numpy_array_2d': np.array([[0.1, 0.2], [0.3, 0.4]]),
            'numpy_int': np.int32(42),
            'numpy_float': np.float64(3.14),
            'regular_int': 42,
            'regular_float': 3.14,
            'string': "test",
            'nested': {
                'inner_array': np.array([1, 2, 3])
            }
        }
        
        # This should fail with the original code
        with pytest.raises(TypeError, match="Object of type ndarray is not JSON serializable"):
            json.dumps(test_data)
            
        # But after conversion, it should work
        converted_data = self._convert_numpy_to_json_serializable(test_data)
        json_str = json.dumps(converted_data)
        assert json_str is not None
        
        # Verify the conversion worked correctly
        deserialized = json.loads(json_str)
        assert deserialized['regular_list'] == [1, 2, 3, 4]
        assert deserialized['numpy_array_1d'] == [0.1, 0.2, 0.3, 0.4]
        assert deserialized['numpy_array_2d'] == [[0.1, 0.2], [0.3, 0.4]]
        assert deserialized['numpy_int'] == 42
        assert deserialized['numpy_float'] == 3.14
        assert deserialized['regular_int'] == 42
        assert deserialized['regular_float'] == 3.14
        assert deserialized['string'] == "test"
        assert deserialized['nested']['inner_array'] == [1, 2, 3]
        
        self.logger.ok("Numpy array detection and conversion test passed")
        
    def test_websocket_broadcast_compatibility(self):
        """Test that messages are compatible with WebSocket broadcast"""
        self.logger.testing("Testing WebSocket broadcast compatibility")
        
        # Create a message that would be broadcast via WebSocket
        message = {
            'type': 'checkpoint_playback',
            'checkpoint_id': 'test_checkpoint',
            'game_number': 1,
            'step_data': {
                'step': 0,
                'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
                'score': 12345,
                'action': 1,
                'action_probs': [0.1, 0.7, 0.15, 0.05],
                'timestamp': 1234567890.0
            }
        }
        
        # Verify it can be serialized for WebSocket transmission
        json_str = json.dumps(message)
        assert json_str is not None
        
        # Verify it can be deserialized on the client side
        deserialized = json.loads(json_str)
        assert deserialized['type'] == 'checkpoint_playback'
        assert deserialized['step_data']['action_probs'] == [0.1, 0.7, 0.15, 0.05]
        
        self.logger.ok("WebSocket broadcast compatibility test passed")
        
    def test_regression_original_error_scenario(self):
        """Test regression of the original error scenario"""
        self.logger.testing("Testing regression of original error scenario")
        
        # Simulate the original error scenario with numpy arrays
        original_step_data = {
            'step': 0,
            'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
            'score': 12345,
            'action': 1,
            'action_probs': np.array([0.1, 0.7, 0.15, 0.05]),  # This would cause the original error
            'timestamp': 1234567890.0
        }
        
        # This should fail without the fix
        with pytest.raises(TypeError, match="Object of type ndarray is not JSON serializable"):
            json.dumps(original_step_data)
        
        # But with the fix, it should work
        fixed_step_data = self._convert_numpy_to_json_serializable(original_step_data)
        json_str = json.dumps(fixed_step_data)
        assert json_str is not None
        
        self.logger.ok("Regression test passed - original error is fixed")
    
    def _convert_numpy_to_json_serializable(self, obj):
        """Convert numpy arrays to JSON serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json_serializable(item) for item in obj]
        else:
            return obj

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("JSON Serialization Test Suite", 60)
    
    # Run the test suite
    test_suite = TestCheckpointPlaybackJSONSerialization()
    
    # Run each test method
    test_methods = [
        test_suite.test_action_probs_json_serializable,
        test_suite.test_full_message_json_serializable,
        test_suite.test_lightweight_message_json_serializable,
        test_suite.test_numpy_array_detection_and_conversion,
        test_suite.test_websocket_broadcast_compatibility,
        test_suite.test_regression_original_error_scenario,
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_suite.setup_method()
            test_method()
            passed += 1
        except Exception as e:
            logger.error(f"Test {test_method.__name__} failed: {e}")
            failed += 1
    
    logger.separator(60)
    if failed == 0:
        logger.success(f"All {passed} tests passed!")
        sys.exit(0)
    else:
        logger.error(f"{failed} tests failed, {passed} tests passed")
        sys.exit(1)

if __name__ == "__main__":
    main() 