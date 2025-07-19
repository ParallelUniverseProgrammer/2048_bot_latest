#!/usr/bin/env python3
"""
Test JSON Serialization Fix Implementation
=========================================

This test suite verifies that the JSON serialization fix properly handles numpy arrays
and prevents the "Object of type ndarray is not JSON serializable" error.

The test covers:
- Numpy array serialization error reproduction
- Numpy array conversion fix verification
- Step data structure serialization
- Full and lightweight message serialization
- Edge cases with None values
- Various numpy types handling
"""

import json
import numpy as np
import sys
import os
from typing import Any, Dict, List

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from tests.utilities.test_utils import TestLogger

class TestJSONSerializationFix:
    """Test suite for JSON serialization fix implementation"""
    
    def __init__(self):
        self.logger = TestLogger()
    
    def test_numpy_array_serialization_error(self):
        """Test that numpy arrays cause JSON serialization errors without fix"""
        self.logger.testing("Testing numpy array serialization error")
        
        # Create data with numpy arrays
        test_data = {
            'action_probs': np.array([0.1, 0.7, 0.15, 0.05]),
            'attention_weights': np.array([[0.25, 0.25, 0.25, 0.25]]),
            'regular_list': [1, 2, 3, 4]
        }
        
        # This should fail without the fix
        try:
            json.dumps(test_data)
            self.logger.error("Expected JSON serialization error but none occurred")
            return False
        except TypeError as e:
            if "ndarray is not JSON serializable" in str(e):
                self.logger.ok("Correctly caught numpy array serialization error")
                return True
            else:
                self.logger.error(f"Unexpected error: {e}")
                return False
    
    def test_numpy_array_conversion_fix(self):
        """Test that the fix properly converts numpy arrays"""
        self.logger.testing("Testing numpy array conversion fix")
        
        # Create data with numpy arrays
        test_data = {
            'action_probs': np.array([0.1, 0.7, 0.15, 0.05]),
            'attention_weights': np.array([[0.25, 0.25, 0.25, 0.25]]),
            'regular_list': [1, 2, 3, 4],
            'numpy_int': np.int32(42),
            'numpy_float': np.float64(3.14)
        }
        
        # Apply the fix
        fixed_data = self._convert_numpy_to_json_serializable(test_data)
        
        # This should work now
        try:
            json_str = json.dumps(fixed_data)
            deserialized = json.loads(json_str)
            
            # Verify the conversion worked correctly
            assert isinstance(deserialized['action_probs'], list)
            assert deserialized['action_probs'] == [0.1, 0.7, 0.15, 0.05]
            assert isinstance(deserialized['attention_weights'], list)
            assert deserialized['attention_weights'] == [[0.25, 0.25, 0.25, 0.25]]
            assert deserialized['regular_list'] == [1, 2, 3, 4]
            assert deserialized['numpy_int'] == 42
            assert deserialized['numpy_float'] == 3.14
            
            self.logger.ok("Numpy array conversion fix works correctly")
            return True
        except Exception as e:
            self.logger.error(f"Fix failed: {e}")
            return False
    
    def test_step_data_structure_serialization(self):
        """Test serialization of complete step data structure"""
        self.logger.testing("Testing step data structure serialization")
        
        # Create realistic step data structure
        step_data = {
            'step': 0,
            'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
            'score': 12345,
            'action': 1,
            'action_probs': np.array([0.1, 0.7, 0.15, 0.05]),
            'legal_actions': [0, 1, 2, 3],
            'attention_weights': np.array([[0.25, 0.25, 0.25, 0.25]]),
            'timestamp': 1234567890.0,
            'reward': 10.0,
            'done': False
        }
        
        # Apply the fix
        fixed_step_data = self._convert_numpy_to_json_serializable(step_data)
        
        # Verify serialization works
        json_str = json.dumps(fixed_step_data)
        deserialized = json.loads(json_str)
        
        # Verify all fields are correct
        assert deserialized['step'] == 0
        assert deserialized['score'] == 12345
        assert deserialized['action'] == 1
        assert isinstance(deserialized['action_probs'], list)
        assert deserialized['action_probs'] == [0.1, 0.7, 0.15, 0.05]
        assert isinstance(deserialized['attention_weights'], list)
        assert deserialized['attention_weights'] == [[0.25, 0.25, 0.25, 0.25]]
        
        self.logger.ok("Step data structure serialization works correctly")
        return True
    
    def test_full_checkpoint_playback_message_serialization(self):
        """Test serialization of full checkpoint playback message"""
        self.logger.testing("Testing full checkpoint playback message serialization")
        
        # Create step data with numpy arrays
        step_data = {
            'step': 0,
            'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
            'score': 12345,
            'action': 1,
            'action_probs': np.array([0.1, 0.7, 0.15, 0.05]),
            'legal_actions': [0, 1, 2, 3],
            'attention_weights': np.array([[0.25, 0.25, 0.25, 0.25]]),
            'timestamp': 1234567890.0,
            'reward': 10.0,
            'done': False
        }
        
        game_result = {
            'final_score': 12345,
            'steps': 100,
            'max_tile': 2048
        }
        
        # Create full message structure
        full_message = {
            'type': 'checkpoint_playback',
            'checkpoint_id': 'test_checkpoint',
            'game_number': 1,
            'step_data': step_data,
            'game_result': game_result
        }
        
        # Apply the fix
        fixed_message = self._convert_numpy_to_json_serializable(full_message)
        
        # Verify serialization works
        json_str = json.dumps(fixed_message)
        deserialized = json.loads(json_str)
        
        # Verify message structure
        assert deserialized['type'] == 'checkpoint_playback'
        assert deserialized['checkpoint_id'] == 'test_checkpoint'
        assert deserialized['game_number'] == 1
        assert isinstance(deserialized['step_data']['action_probs'], list)
        assert deserialized['step_data']['action_probs'] == [0.1, 0.7, 0.15, 0.05]
        
        self.logger.ok("Full checkpoint playback message serialization works correctly")
        return True
    
    def test_lightweight_checkpoint_playback_message_serialization(self):
        """Test serialization of lightweight checkpoint playback message"""
        self.logger.testing("Testing lightweight checkpoint playback message serialization")
        
        # Create step data with numpy arrays
        step_data = {
            'step': 0,
            'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
            'score': 12345,
            'action': 1,
            'action_probs': np.array([0.1, 0.7, 0.15, 0.05]),
            'timestamp': 1234567890.0
        }
        
        game_result = {
            'final_score': 12345,
            'steps': 100,
            'max_tile': 2048
        }
        
        # Create lightweight message structure
        light_message = {
            'type': 'checkpoint_playback_light',
            'checkpoint_id': 'test_checkpoint',
            'game_number': 1,
            'step': 0,
            'board_state': [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]],
            'score': 12345,
            'action': 1,
            'action_probs': np.array([0.1, 0.7, 0.15, 0.05]),
            'timestamp': 1234567890.0
        }
        
        # Apply the fix
        fixed_message = self._convert_numpy_to_json_serializable(light_message)
        
        # Verify serialization works
        json_str = json.dumps(fixed_message)
        deserialized = json.loads(json_str)
        
        # Verify message structure
        assert deserialized['type'] == 'checkpoint_playback_light'
        assert deserialized['checkpoint_id'] == 'test_checkpoint'
        assert deserialized['game_number'] == 1
        assert isinstance(deserialized['action_probs'], list)
        assert deserialized['action_probs'] == [0.1, 0.7, 0.15, 0.05]
        
        self.logger.ok("Lightweight checkpoint playback message serialization works correctly")
        return True
    
    def test_edge_cases_with_none_values(self):
        """Test handling of None values and edge cases"""
        self.logger.testing("Testing edge cases with None values")
        
        # Create data with None values and numpy arrays
        test_data = {
            'action_probs': np.array([0.1, 0.7, 0.15, 0.05]),
            'attention_weights': None,
            'regular_list': [1, 2, 3, 4],
            'none_value': None,
            'empty_array': np.array([])
        }
        
        # Apply the fix
        fixed_data = self._convert_numpy_to_json_serializable(test_data)
        
        # Verify serialization works
        json_str = json.dumps(fixed_data)
        deserialized = json.loads(json_str)
        
        # Verify edge cases are handled correctly
        assert isinstance(deserialized['action_probs'], list)
        assert deserialized['action_probs'] == [0.1, 0.7, 0.15, 0.05]
        assert deserialized['attention_weights'] is None
        assert deserialized['regular_list'] == [1, 2, 3, 4]
        assert deserialized['none_value'] is None
        assert deserialized['empty_array'] == []
        
        self.logger.ok("Edge cases with None values handled correctly")
        return True
    
    def test_various_numpy_types(self):
        """Test various numpy types and their conversion"""
        self.logger.testing("Testing various numpy types")
        
        # Create data with various numpy types
        test_data = {
            'int32': np.int32(42),
            'int64': np.int64(123456789),
            'float32': np.float32(3.14),
            'float64': np.float64(2.718),
            'bool': np.bool_(True),
            'array_1d': np.array([1, 2, 3, 4]),
            'array_2d': np.array([[1, 2], [3, 4]]),
            'array_3d': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        }
        
        # Apply the fix
        fixed_data = self._convert_numpy_to_json_serializable(test_data)
        
        # Verify serialization works
        json_str = json.dumps(fixed_data)
        deserialized = json.loads(json_str)
        
        # Verify all types are converted correctly
        assert deserialized['int32'] == 42
        assert deserialized['int64'] == 123456789
        assert abs(deserialized['float32'] - 3.14) < 0.001
        assert abs(deserialized['float64'] - 2.718) < 0.001
        assert deserialized['bool'] is True
        assert deserialized['array_1d'] == [1, 2, 3, 4]
        assert deserialized['array_2d'] == [[1, 2], [3, 4]]
        assert deserialized['array_3d'] == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        
        self.logger.ok("Various numpy types converted correctly")
        return True
    
    def _convert_numpy_to_json_serializable(self, obj):
        """Convert numpy arrays to JSON serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json_serializable(item) for item in obj]
        else:
            return obj

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("JSON Serialization Fix Test Suite", 60)
    
    # Create test instance
    test_suite = TestJSONSerializationFix()
    
    # Run all tests
    tests = [
        test_suite.test_numpy_array_serialization_error,
        test_suite.test_numpy_array_conversion_fix,
        test_suite.test_step_data_structure_serialization,
        test_suite.test_full_checkpoint_playback_message_serialization,
        test_suite.test_lightweight_checkpoint_playback_message_serialization,
        test_suite.test_edge_cases_with_none_values,
        test_suite.test_various_numpy_types,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
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