"""
Test JSON serialization of checkpoint playback messages to prevent ndarray serialization errors.
"""
import json
import numpy as np
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)

try:
    from app.models.checkpoint_playback import CheckpointPlayback
    from app.models.model_config import ModelConfig
    from app.models.checkpoint_metadata import CheckpointMetadata
    from app.environment.gym_2048_env import Gym2048Env
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Backend path: {backend_path}")
    print(f"Available in backend: {os.listdir(backend_path) if os.path.exists(backend_path) else 'Path does not exist'}")
    raise


class TestCheckpointPlaybackJSONSerialization:
    """Test suite for JSON serialization of checkpoint playback messages"""
    
    def setup_method(self):
        """Set up test fixtures"""
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
            
    def test_full_message_json_serializable(self):
        """Test that _create_full_message produces JSON serializable output"""
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
        
    def test_lightweight_message_json_serializable(self):
        """Test that _create_lightweight_message produces JSON serializable output"""
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
        
    def test_numpy_array_detection_and_conversion(self):
        """Test that numpy arrays are properly detected and converted"""
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
        
    def test_websocket_broadcast_compatibility(self):
        """Test that messages are compatible with WebSocket broadcast"""
        # Create a realistic step data as would be generated
        with patch.object(self.playback, 'select_action') as mock_select:
            mock_select.return_value = (
                2,  # action
                np.array([0.05, 0.15, 0.75, 0.05]),  # action_probs as numpy array
                np.array([[0.1, 0.2, 0.3, 0.4]])  # attention_weights as numpy array
            )
            
            self.mock_env.is_done.side_effect = [False, True]
            
            # Run play_single_game
            result = self.playback.play_single_game()
            step_data = result['game_history'][0]
            
            # Create both message types
            full_message = self.playback._create_full_message(step_data, result, "test_checkpoint", 1)
            light_message = self.playback._create_lightweight_message(step_data, result, "test_checkpoint", 1)
            
            # Verify both can be JSON serialized (this is what websocket_manager.broadcast does)
            json.dumps(full_message)  # Should not raise
            json.dumps(light_message)  # Should not raise
            
            # Verify the messages have the expected structure
            assert full_message['type'] == 'checkpoint_playback'
            assert light_message['type'] == 'checkpoint_playback_light'
            
            # Verify action_probs is a list in both
            assert isinstance(full_message['step_data']['action_probs'], list)
            assert isinstance(light_message['action_probs'], list)
            
    def test_regression_original_error_scenario(self):
        """Regression test for the original 'ndarray is not JSON serializable' error"""
        # This test simulates the exact scenario that caused the original error
        
        # Mock the select_action to return numpy arrays (as it would in real usage)
        with patch.object(self.playback, 'select_action') as mock_select:
            mock_select.return_value = (
                1,  # action
                np.array([0.1, 0.7, 0.15, 0.05], dtype=np.float32),  # numpy array
                np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)  # numpy array
            )
            
            # Set up environment for one step
            self.mock_env.is_done.side_effect = [False, True]
            
            # Run the game
            result = self.playback.play_single_game()
            
            # Get step data
            step_data = result['game_history'][0]
            
            # Create the full message (this is what was failing before)
            message = self.playback._create_full_message(step_data, result, "test_checkpoint", 1)
            
            # This should NOT raise "TypeError: Object of type ndarray is not JSON serializable"
            try:
                json_str = json.dumps(message)
                # If we get here, the fix worked
                assert json_str is not None
            except TypeError as e:
                if "ndarray is not JSON serializable" in str(e):
                    pytest.fail("The original ndarray JSON serialization error still occurs!")
                else:
                    # Re-raise if it's a different TypeError
                    raise
                    
    def _convert_numpy_to_json_serializable(self, obj):
        """Helper function to recursively convert numpy objects to JSON serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json_serializable(item) for item in obj]
        else:
            return obj


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 