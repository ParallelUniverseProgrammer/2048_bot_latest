#!/usr/bin/env python3
"""
Comprehensive test for device compatibility in the entire checkpoint playback pipeline
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import torch
import numpy as np
from app.models.checkpoint_playback import CheckpointPlayback
from app.models.checkpoint_metadata import CheckpointManager
from app.models.game_transformer import GameTransformer
from app.models.model_config import ModelConfig
from app.environment.gym_2048_env import Gym2048Env
from app.utils.action_selection import select_action_with_fallback_for_playback, select_action_with_fallback
from copy import deepcopy
import traceback

def test_device_compatibility_pipeline():
    """Test the entire device compatibility pipeline"""
    
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
    
    # Test 1: Direct model creation and forward pass
    print("\n=== Test 1: Direct model creation and forward pass ===")
    try:
        model = GameTransformer(config)
        model.to(device)
        model.eval()
        
        print(f"Model parameters device: {next(model.parameters()).device}")
        
        # Test forward pass with proper device placement
        dummy_board = torch.randint(0, 16, (1, 4, 4), device=device)
        print(f"Input tensor device: {dummy_board.device}")
        
        with torch.no_grad():
            policy_logits, value = model(dummy_board)
            print(f"Policy logits device: {policy_logits.device}")
            print(f"Value device: {value.device}")
        
        print("OK Direct model test passed")
    except Exception as e:
        print(f"ERROR: Direct model test failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 2: Checkpoint manager and playback system
    print("\n=== Test 2: Checkpoint manager and playback system ===")
    
    # Create checkpoint manager
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'backend', 'checkpoints')
    cm = CheckpointManager(checkpoint_dir)
    
    # Check available checkpoints
    checkpoints = cm.list_checkpoints()
    print(f"Available checkpoints: {[cp.id for cp in checkpoints]}")
    
    if not checkpoints:
        print("ERROR: No checkpoints available for testing")
        return False
    
    # Use the first checkpoint
    checkpoint_id = checkpoints[0].id
    print(f"Using checkpoint: {checkpoint_id}")
    
    # Create playback system
    playback = CheckpointPlayback(cm)
    
    # Test checkpoint loading
    print("\n=== Test 3: Checkpoint loading ===")
    try:
        success = playback.load_checkpoint(checkpoint_id)
        if not success:
            print("ERROR: Failed to load checkpoint")
            return False
        
        print(f"OK Checkpoint loaded successfully")
        print(f"Model device: {next(playback.current_model.parameters()).device}")
        print(f"Playback device: {playback.device}")
        
        # Verify devices match
        model_device = next(playback.current_model.parameters()).device
        if model_device != playback.device:
            print(f"ERROR: Device mismatch: model on {model_device}, playback expects {playback.device}")
            return False
        
        print("OK Device consistency verified")
        
    except Exception as e:
        print(f"ERROR: Checkpoint loading failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 4: Environment and state handling
    print("\n=== Test 4: Environment and state handling ===")
    try:
        env = Gym2048Env()
        env.reset()
        
        state = env.get_state()
        legal_actions = env.get_legal_actions()
        
        print(f"State type: {type(state)}")
        print(f"State dtype: {state.dtype}")
        print(f"State shape: {state.shape}")
        print(f"Legal actions: {legal_actions}")
        
        # Test numpy to tensor conversion
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        print(f"State tensor device: {state_tensor.device}")
        print(f"State tensor shape: {state_tensor.shape}")
        
        print("OK Environment and state handling test passed")
        
    except Exception as e:
        print(f"ERROR: Environment and state handling failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 5: Action selection with loaded model
    print("\n=== Test 5: Action selection with loaded model ===")
    try:
        # Test the actual action selection methods used in playback
        action, action_probs, attention_weights = playback.select_action(state, legal_actions, env.game)
        
        print(f"OK Action selection successful")
        print(f"Selected action: {action}")
        print(f"Action probabilities: {action_probs}")
        
        # Test multiple calls to see if device errors occur
        print("\nTesting multiple action selections...")
        for i in range(10):
            try:
                action, action_probs, attention_weights = playback.select_action(state, legal_actions, env.game)
                print(f"  Call {i+1}: action={action} OK")
            except Exception as e:
                print(f"  Call {i+1}: ERROR - {e}")
                traceback.print_exc()
                return False
        
        print("OK Multiple action selections test passed")
        
    except Exception as e:
        print(f"ERROR: Action selection failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 6: Game loop simulation
    print("\n=== Test 6: Game loop simulation ===")
    try:
        env.reset()
        step_count = 0
        max_steps = 20  # Just test a few steps
        
        while not env.is_done() and step_count < max_steps:
            try:
                # Get current state
                state = env.get_state()
                legal_actions = env.get_legal_actions()
                
                if not legal_actions:
                    break
                
                # Select action
                action, action_probs, attention_weights = playback.select_action(state, legal_actions, env.game)
                
                # Take action
                obs, reward, done, truncated, info = env.step(action)
                step_count += 1
                
                print(f"  Step {step_count}: action={action}, reward={reward:.2f}, done={done}")
                
                if done or truncated:
                    break
                    
            except Exception as e:
                print(f"ERROR: Game loop error at step {step_count}: {e}")
                traceback.print_exc()
                return False
        
        print(f"OK Game loop simulation completed {step_count} steps")
        
    except Exception as e:
        print(f"ERROR: Game loop simulation failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 7: Direct function calls to check device compatibility
    print("\n=== Test 7: Direct function calls ===")
    try:
        # Test direct calls to action selection functions
        env.reset()
        state = env.get_state()
        legal_actions = env.get_legal_actions()
        
        # Test playback function
        action, action_probs, attention_weights = select_action_with_fallback_for_playback(
            model=playback.current_model,
            state=state,
            legal_actions=legal_actions,
            env_game=env.game,
            device=playback.device,
            deterministic=True
        )
        print(f"OK Direct playback function call: action={action}")
        
        # Test training function
        action, log_prob, attention_weights = select_action_with_fallback(
            model=playback.current_model,
            state=state,
            legal_actions=legal_actions,
            env_game=env.game,
            device=playback.device,
            sample_action=False,
            max_attempts=2
        )
        print(f"OK Direct training function call: action={action}, log_prob={log_prob:.4f}")
        
    except Exception as e:
        print(f"ERROR: Direct function calls failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 8: Single game playback
    print("\n=== Test 8: Single game playback ===")
    try:
        result = playback.play_single_game()
        if isinstance(result, dict) and 'error' in result:
            print(f"ERROR: Single game playback failed: {result['error']}")
            return False
        
        print(f"OK Single game playback completed")
        print(f"  Final score: {result.get('final_score', 'unknown')}")
        print(f"  Steps: {result.get('steps', 'unknown')}")
        print(f"  Max tile: {result.get('max_tile', 'unknown')}")
        
    except Exception as e:
        print(f"ERROR: Single game playback failed: {e}")
        traceback.print_exc()
        return False
    
    print("\nOK: All comprehensive device compatibility tests passed!")
    return True

if __name__ == "__main__":
    success = test_device_compatibility_pipeline()
    if not success:
        sys.exit(1)
    else:
        print("\nSUCCESS: Device compatibility is working correctly!")
        print("The issue might be in a different part of the system or already fixed.") 