"""
Test for device compatibility in action selection
"""

import torch
import numpy as np
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.utils.action_selection import select_action_with_fallback_for_playback, select_action_with_fallback
from app.models.game_transformer import GameTransformer
from app.models.model_config import ModelConfig


class TestDeviceCompatibility(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Testing on device: {self.device}")
        
        # Create a simple mock model
        self.model = MagicMock()
        self.model.to.return_value = self.model
        
        # Mock model output (policy_logits, value)
        policy_logits = torch.randn(1, 4, device=self.device)
        value = torch.randn(1, 1, device=self.device)
        self.model.return_value = (policy_logits, value)
        
        # Test state
        self.state = np.random.rand(4, 4)
        
        # Mock game environment
        self.env_game = MagicMock()
        self.env_game._move.return_value = (True, 10)  # Valid move
        
        # Legal actions
        self.legal_actions = [0, 1, 2, 3]
    
    def test_tensor_creation_with_device(self):
        """Test correct tensor creation with device"""
        action = 1
        
        # Test correct ways to create tensor on device
        correct_ways = [
            torch.tensor(action).to(self.device),
            torch.tensor(action, device=self.device),  # This is the correct way
        ]
        
        for tensor in correct_ways:
            self.assertEqual(tensor.device, self.device)
        
        # Test incorrect way (this should fail)
        try:
            # This is what was causing the problem
            incorrect_tensor = torch.tensor(action, device=self.device)
            # If it doesn't fail, at least verify it has the right device
            self.assertEqual(incorrect_tensor.device, self.device)
        except Exception as e:
            print(f"Error with incorrect tensor creation: {e}")
    
    def test_action_selection_device_compatibility(self):
        """Test that action selection works with GPU/CPU device compatibility"""
        try:
            # Test the playback function
            action, action_probs_list, attention_weights = select_action_with_fallback_for_playback(
                self.model,
                self.state,
                self.legal_actions,
                self.env_game,
                self.device,
                deterministic=True
            )
            
            # Verify results
            self.assertIsInstance(action, int)
            self.assertIn(action, self.legal_actions)
            self.assertIsInstance(action_probs_list, list)
            self.assertEqual(len(action_probs_list), 4)
            
            print(f"✓ Action selection succeeded: action={action}")
            
        except Exception as e:
            self.fail(f"Device compatibility error in action selection: {e}")
    
    def test_action_selection_with_fallback_device_compatibility(self):
        """Test that action selection with fallback works with device compatibility"""
        try:
            # Test the training function
            action, log_prob, attention_weights = select_action_with_fallback(
                self.model,
                self.state,
                self.legal_actions,
                self.env_game,
                self.device,
                sample_action=False,
                max_attempts=2
            )
            
            # Verify results
            self.assertIsInstance(action, int)
            self.assertIn(action, self.legal_actions)
            self.assertIsInstance(log_prob, float)
            
            print(f"✓ Action selection with fallback succeeded: action={action}, log_prob={log_prob}")
            
        except Exception as e:
            self.fail(f"Device compatibility error in action selection with fallback: {e}")
    
    def test_categorical_distribution_device_compatibility(self):
        """Test that categorical distribution works with tensor device compatibility"""
        try:
            # Create action probabilities on the correct device
            action_probs = torch.tensor([0.25, 0.25, 0.25, 0.25], device=self.device)
            action_dist = torch.distributions.Categorical(action_probs)
            
            # Test both ways of creating action tensor
            action = 1
            
            # Correct way 1: create then move to device
            action_tensor_1 = torch.tensor(action).to(self.device)
            log_prob_1 = action_dist.log_prob(action_tensor_1)
            self.assertEqual(log_prob_1.device, self.device)
            
            # Correct way 2: create directly on device
            action_tensor_2 = torch.tensor(action, device=self.device)
            log_prob_2 = action_dist.log_prob(action_tensor_2)
            self.assertEqual(log_prob_2.device, self.device)
            
            print(f"✓ Categorical distribution device compatibility verified")
            
        except Exception as e:
            self.fail(f"Device compatibility error in categorical distribution: {e}")


if __name__ == '__main__':
    unittest.main() 