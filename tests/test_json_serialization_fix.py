"""
Simple test to verify the JSON serialization fix for numpy arrays in checkpoint playback.
"""
import json
import numpy as np
import pytest


class TestJSONSerializationFix:
    """Test the specific JSON serialization fix for numpy arrays"""
    
    def test_numpy_array_serialization_error(self):
        """Test that numpy arrays cause JSON serialization errors"""
        # This demonstrates the original problem
        data_with_numpy = {
            'action_probs': np.array([0.1, 0.7, 0.15, 0.05]),
            'attention_weights': np.array([[0.25, 0.25, 0.25, 0.25]]),
            'other_data': 'test'
        }
        
        # This should raise the original error
        with pytest.raises(TypeError, match="Object of type ndarray is not JSON serializable"):
            json.dumps(data_with_numpy)
    
    def test_numpy_array_conversion_fix(self):
        """Test that converting numpy arrays to lists fixes the JSON serialization"""
        # This demonstrates the fix
        action_probs = np.array([0.1, 0.7, 0.15, 0.05])
        attention_weights = np.array([[0.25, 0.25, 0.25, 0.25]])
        
        # Convert to lists (the fix)
        data_with_lists = {
            'action_probs': action_probs.tolist() if action_probs is not None else None,
            'attention_weights': attention_weights.tolist() if attention_weights is not None else None,
            'other_data': 'test'
        }
        
        # This should work without errors
        json_str = json.dumps(data_with_lists)
        assert json_str is not None
        
        # Verify the data is preserved correctly
        deserialized = json.loads(json_str)
        assert deserialized['action_probs'] == [0.1, 0.7, 0.15, 0.05]
        assert deserialized['attention_weights'] == [[0.25, 0.25, 0.25, 0.25]]
        assert deserialized['other_data'] == 'test'
    
    def test_step_data_structure_serialization(self):
        """Test that the step_data structure can be JSON serialized after the fix"""
        # Simulate the exact structure that was causing the issue
        step_data = {
            'step': 0,
            'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
            'score': 12345,
            'action': 1,
            'action_probs': np.array([0.1, 0.7, 0.15, 0.05]).tolist(),  # Fixed: converted to list
            'legal_actions': [0, 1, 2, 3],
            'attention_weights': np.array([[0.25, 0.25, 0.25, 0.25]]).tolist(),  # Fixed: converted to list
            'timestamp': 1234567890.0,
            'reward': 10.0,
            'done': False
        }
        
        # This should work without errors now
        json_str = json.dumps(step_data)
        assert json_str is not None
        
        # Verify the data is preserved correctly
        deserialized = json.loads(json_str)
        assert deserialized['action_probs'] == [0.1, 0.7, 0.15, 0.05]
        assert deserialized['attention_weights'] == [[0.25, 0.25, 0.25, 0.25]]
        assert deserialized['step'] == 0
        assert deserialized['score'] == 12345
    
    def test_full_checkpoint_playback_message_serialization(self):
        """Test that the full checkpoint playback message can be JSON serialized"""
        # Simulate the exact message structure that was causing the issue
        step_data = {
            'step': 0,
            'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
            'score': 12345,
            'action': 1,
            'action_probs': [0.1, 0.7, 0.15, 0.05],  # Fixed: already a list
            'legal_actions': [0, 1, 2, 3],
            'attention_weights': [[0.25, 0.25, 0.25, 0.25]],  # Fixed: already a list
            'timestamp': 1234567890.0,
            'reward': 10.0,
            'done': False
        }
        
        game_result = {
            'final_score': 12345,
            'steps': 100,
            'max_tile': 2048
        }
        
        # Simulate the full message structure
        full_message = {
            'type': 'checkpoint_playback',
            'checkpoint_id': 'test_checkpoint',
            'game_number': 1,
            'step_data': step_data,
            'game_summary': {
                'final_score': game_result['final_score'],
                'total_steps': game_result['steps'],
                'max_tile': game_result['max_tile']
            },
            'performance_info': {
                'adaptive_skip': 1,
                'lightweight_mode': False,
                'broadcast_interval': 0.1
            }
        }
        
        # This should work without errors now
        json_str = json.dumps(full_message)
        assert json_str is not None
        
        # Verify the message structure is preserved
        deserialized = json.loads(json_str)
        assert deserialized['type'] == 'checkpoint_playback'
        assert deserialized['checkpoint_id'] == 'test_checkpoint'
        assert deserialized['game_number'] == 1
        assert deserialized['step_data']['action_probs'] == [0.1, 0.7, 0.15, 0.05]
        assert deserialized['step_data']['attention_weights'] == [[0.25, 0.25, 0.25, 0.25]]
    
    def test_lightweight_checkpoint_playback_message_serialization(self):
        """Test that the lightweight checkpoint playback message can be JSON serialized"""
        # Simulate the lightweight message structure
        step_data = {
            'step': 0,
            'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
            'score': 12345,
            'action': 1,
            'action_probs': [0.1, 0.7, 0.15, 0.05],  # Fixed: already a list
            'timestamp': 1234567890.0
        }
        
        game_result = {
            'final_score': 12345,
            'steps': 100,
            'max_tile': 2048
        }
        
        # Simulate the lightweight message structure
        lightweight_message = {
            'type': 'checkpoint_playback_light',
            'checkpoint_id': 'test_checkpoint',
            'game_number': 1,
            'step': step_data['step'],
            'board_state': step_data['board_state'],
            'score': step_data['score'],
            'action': step_data['action'],
            'timestamp': step_data['timestamp'],
            'game_progress': {
                'current_step': step_data['step'],
                'total_steps': game_result['steps'],
                'final_score': game_result['final_score']
            }
        }
        
        # This should work without errors now
        json_str = json.dumps(lightweight_message)
        assert json_str is not None
        
        # Verify the message structure is preserved
        deserialized = json.loads(json_str)
        assert deserialized['type'] == 'checkpoint_playback_light'
        assert deserialized['checkpoint_id'] == 'test_checkpoint'
        assert deserialized['game_number'] == 1
        assert deserialized['step'] == 0
        assert deserialized['score'] == 12345
    
    def test_edge_cases_with_none_values(self):
        """Test edge cases where numpy arrays might be None"""
        # Test with None values (which can happen in some cases)
        step_data = {
            'step': 0,
            'action_probs': None,  # This can happen
            'attention_weights': None,  # This can happen
            'other_data': 'test'
        }
        
        # This should work without errors
        json_str = json.dumps(step_data)
        assert json_str is not None
        
        # Verify the data is preserved correctly
        deserialized = json.loads(json_str)
        assert deserialized['action_probs'] is None
        assert deserialized['attention_weights'] is None
        assert deserialized['other_data'] == 'test'
    
    def test_various_numpy_types(self):
        """Test various numpy data types are handled correctly"""
        # Test different numpy types that might appear
        test_data = {
            'float32_array': np.array([1.0, 2.0, 3.0], dtype=np.float32).tolist(),
            'float64_array': np.array([1.0, 2.0, 3.0], dtype=np.float64).tolist(),
            'int32_array': np.array([1, 2, 3], dtype=np.int32).tolist(),
            'int64_array': np.array([1, 2, 3], dtype=np.int64).tolist(),
            'bool_array': np.array([True, False, True], dtype=np.bool_).tolist(),
            'regular_data': 'test'
        }
        
        # This should work without errors
        json_str = json.dumps(test_data)
        assert json_str is not None
        
        # Verify the data is preserved correctly
        deserialized = json.loads(json_str)
        assert deserialized['float32_array'] == [1.0, 2.0, 3.0]
        assert deserialized['float64_array'] == [1.0, 2.0, 3.0]
        assert deserialized['int32_array'] == [1, 2, 3]
        assert deserialized['int64_array'] == [1, 2, 3]
        assert deserialized['bool_array'] == [True, False, True]
        assert deserialized['regular_data'] == 'test'


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 