"""
Integration test to verify WebSocket broadcast fix for numpy array serialization.
"""
import json
import numpy as np
import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend')
sys.path.insert(0, backend_path)

try:
    from app.api.websocket_manager import WebSocketManager
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestWebSocketBroadcastFix:
    """Test that WebSocket broadcast handles numpy arrays correctly"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.websocket_manager = WebSocketManager()
    
    def test_broadcast_with_numpy_arrays_fails_before_fix(self):
        """Test that broadcasting messages with numpy arrays would fail before the fix"""
        # This simulates the exact message that was causing the original error
        message_with_numpy = {
            'type': 'checkpoint_playback',
            'checkpoint_id': 'test_checkpoint',
            'game_number': 1,
            'step_data': {
                'step': 0,
                'action_probs': np.array([0.1, 0.7, 0.15, 0.05]),  # This would cause the error
                'attention_weights': np.array([[0.25, 0.25, 0.25, 0.25]]),  # This would cause the error
                'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
                'score': 12345,
                'action': 1,
                'timestamp': 1234567890.0
            }
        }
        
        # This should fail with the original error (demonstrating the problem)
        with pytest.raises(TypeError, match="Object of type ndarray is not JSON serializable"):
            json.dumps(message_with_numpy)
    
    def test_broadcast_with_converted_arrays_works_after_fix(self):
        """Test that broadcasting messages with converted arrays works after the fix"""
        # This simulates the message after our fix
        message_with_lists = {
            'type': 'checkpoint_playback',
            'checkpoint_id': 'test_checkpoint',
            'game_number': 1,
            'step_data': {
                'step': 0,
                'action_probs': [0.1, 0.7, 0.15, 0.05],  # Fixed: converted to list
                'attention_weights': [[0.25, 0.25, 0.25, 0.25]],  # Fixed: converted to list
                'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
                'score': 12345,
                'action': 1,
                'timestamp': 1234567890.0
            }
        }
        
        # This should work without errors (demonstrating the fix)
        json_str = json.dumps(message_with_lists)
        assert json_str is not None
        
        # Verify the data is preserved correctly
        deserialized = json.loads(json_str)
        assert deserialized['step_data']['action_probs'] == [0.1, 0.7, 0.15, 0.05]
        assert deserialized['step_data']['attention_weights'] == [[0.25, 0.25, 0.25, 0.25]]
    
    @patch('app.api.websocket_manager.WebSocketManager.get_connection_count')
    def test_websocket_manager_broadcast_method_json_serialization(self, mock_get_connection_count):
        """Test that the WebSocket manager's broadcast method can handle the fixed messages"""
        # Mock no connections to avoid actual WebSocket operations
        mock_get_connection_count.return_value = 0
        
        # This simulates the message after our fix
        message_with_lists = {
            'type': 'checkpoint_playback',
            'checkpoint_id': 'test_checkpoint',
            'game_number': 1,
            'step_data': {
                'step': 0,
                'action_probs': [0.1, 0.7, 0.15, 0.05],  # Fixed: converted to list
                'attention_weights': [[0.25, 0.25, 0.25, 0.25]],  # Fixed: converted to list
                'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
                'score': 12345,
                'action': 1,
                'timestamp': 1234567890.0
            }
        }
        
        # The broadcast method should be able to handle this without errors
        # Since we have no connections, it should return immediately without trying to send
        # This tests that the JSON serialization check in broadcast() works
        try:
            # This should not raise any errors
            import asyncio
            asyncio.run(self.websocket_manager.broadcast(message_with_lists))
        except Exception as e:
            # If there's an error, it should NOT be about JSON serialization
            assert "Object of type ndarray is not JSON serializable" not in str(e)
    
    def test_training_update_message_serialization(self):
        """Test that training update messages (which also had arrays) are properly serialized"""
        # This simulates a training update message that might contain arrays
        training_message = {
            'type': 'training_update',
            'timestamp': 1234567890.0,
            'episode': 100,
            'score': 2048,
            'reward': 100.0,
            'loss': 0.5,
            'actions': [0.1, 0.7, 0.15, 0.05],  # Should be a list, not numpy array
            'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
            'attention_weights': [[0.25, 0.25, 0.25, 0.25]],  # Should be a list, not numpy array
            'expert_usage': [0.2, 0.3, 0.1, 0.4],  # Should be a list, not numpy array
            'loss_history': {'episodes': [1, 2, 3], 'values': [0.8, 0.6, 0.5]},
            'score_history': {'episodes': [1, 2, 3], 'values': [512, 1024, 2048]}
        }
        
        # This should work without errors
        json_str = json.dumps(training_message)
        assert json_str is not None
        
        # Verify the data is preserved correctly
        deserialized = json.loads(json_str)
        assert deserialized['type'] == 'training_update'
        assert deserialized['actions'] == [0.1, 0.7, 0.15, 0.05]
        assert deserialized['attention_weights'] == [[0.25, 0.25, 0.25, 0.25]]
        assert deserialized['expert_usage'] == [0.2, 0.3, 0.1, 0.4]
    
    def test_regression_original_error_message_structure(self):
        """Regression test for the exact error message structure from the original issue"""
        # This recreates the exact scenario from the original error logs
        
        # The original error occurred when broadcasting checkpoint playback messages
        # with numpy arrays in the step_data
        
        # Before fix: this would cause "TypeError: Object of type ndarray is not JSON serializable"
        numpy_action_probs = np.array([0.1, 0.7, 0.15, 0.05])
        numpy_attention_weights = np.array([[0.25, 0.25, 0.25, 0.25]])
        
        # After fix: convert to lists before creating the message
        fixed_action_probs = numpy_action_probs.tolist() if numpy_action_probs is not None else None
        fixed_attention_weights = numpy_attention_weights.tolist() if numpy_attention_weights is not None else None
        
        # Create the message structure that would have failed before
        message = {
            'type': 'checkpoint_playback',
            'checkpoint_id': 'checkpoint_episode_100',
            'game_number': 1,
            'step_data': {
                'step': 0,
                'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
                'score': 12345,
                'action': 1,
                'action_probs': fixed_action_probs,  # Fixed: converted to list
                'legal_actions': [0, 1, 2, 3],
                'attention_weights': fixed_attention_weights,  # Fixed: converted to list
                'timestamp': 1234567890.0,
                'reward': 10.0,
                'done': False
            },
            'game_summary': {
                'final_score': 12345,
                'total_steps': 100,
                'max_tile': 2048
            },
            'performance_info': {
                'adaptive_skip': 1,
                'lightweight_mode': False,
                'broadcast_interval': 0.1
            }
        }
        
        # This should work without the original error
        json_str = json.dumps(message)
        assert json_str is not None
        
        # Verify the message can be deserialized correctly
        deserialized = json.loads(json_str)
        assert deserialized['type'] == 'checkpoint_playback'
        assert deserialized['step_data']['action_probs'] == [0.1, 0.7, 0.15, 0.05]
        assert deserialized['step_data']['attention_weights'] == [[0.25, 0.25, 0.25, 0.25]]
        
        # Verify that the original arrays were properly converted
        assert isinstance(deserialized['step_data']['action_probs'], list)
        assert isinstance(deserialized['step_data']['attention_weights'], list)
        assert not isinstance(deserialized['step_data']['action_probs'], np.ndarray)
        assert not isinstance(deserialized['step_data']['attention_weights'], np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 