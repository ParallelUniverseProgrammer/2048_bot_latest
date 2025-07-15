#!/usr/bin/env python3
"""
Test script to demonstrate the device compatibility fix
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import torch
import numpy as np
from app.utils.action_selection import select_action_with_fallback_for_playback, select_action_with_fallback
from app.models.game_transformer import GameTransformer
from app.models.model_config import ModelConfig
from app.environment.gym_2048_env import Gym2048Env
from copy import deepcopy

def test_device_compatibility_fix():
    """Test that demonstrates the device compatibility fix"""
    
    # Create device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create simple model config
    config = ModelConfig(
        d_model=256,
        n_heads=8,
        n_layers=4,
        n_experts=4,
        d_ff=1024,
        top_k=2,
        dropout=0.1,
        attention_dropout=0.1
    )
    
    # Create and load model
    model = GameTransformer(config)
    model.to(device)
    model.eval()
    
    print(f"Model device: {next(model.parameters()).device}")
    
    # Create environment
    env = Gym2048Env()
    env.reset()
    
    # Get state and legal actions
    state = env.get_state()
    legal_actions = env.get_legal_actions()
    
    print(f"State shape: {state.shape}")
    print(f"State dtype: {state.dtype}")
    print(f"Legal actions: {legal_actions}")
    
    # Test action selection with fallback (playback version)
    print("\nTesting playback action selection...")
    
    try:
        action, action_probs, attention_weights = select_action_with_fallback_for_playback(
            model=model,
            state=state,
            legal_actions=legal_actions,
            env_game=env.game,
            device=device,
            deterministic=True
        )
        
        print(f"✓ Playback action selection successful!")
        print(f"  Selected action: {action}")
        print(f"  Action probabilities: {action_probs}")
        print(f"  Attention weights shape: {attention_weights.shape if attention_weights is not None else None}")
        
    except Exception as e:
        print(f"❌ Playback action selection error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test action selection with fallback (training version)
    print("\nTesting training action selection...")
    
    try:
        action, log_prob, attention_weights = select_action_with_fallback(
            model=model,
            state=state,
            legal_actions=legal_actions,
            env_game=env.game,
            device=device,
            sample_action=False,
            max_attempts=2
        )
        
        print(f"✓ Training action selection successful!")
        print(f"  Selected action: {action}")
        print(f"  Log probability: {log_prob}")
        print(f"  Attention weights shape: {attention_weights.shape if attention_weights is not None else None}")
        
        # Test multiple selections to see if device errors occur
        print("\nTesting multiple training selections...")
        for i in range(5):
            try:
                action, log_prob, attention_weights = select_action_with_fallback(
                    model=model,
                    state=state,
                    legal_actions=legal_actions,
                    env_game=env.game,
                    device=device,
                    sample_action=False,
                    max_attempts=2
                )
                print(f"  Selection {i+1}: action={action}, log_prob={log_prob:.4f} ✓")
            except Exception as e:
                print(f"  Selection {i+1}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print("\n✅ All device compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Training action selection error: {e}")
        print(f"Error type: {type(e)}")
        
        # Show more detailed error information
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return False

if __name__ == "__main__":
    success = test_device_compatibility_fix()
    if not success:
        sys.exit(1) 