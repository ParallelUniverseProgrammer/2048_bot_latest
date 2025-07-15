#!/usr/bin/env python3
"""
Test script to verify GPU usage during model training
"""

import torch
import time
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from app.models.model_config import DynamicModelConfig
from app.models.game_transformer import GameTransformer
from app.environment.gym_2048_env import Gym2048Env

def test_gpu_usage():
    """Test that the model is actually using the GPU"""
    
    print("=== GPU Usage Test ===")
    
    # Check initial GPU memory
    if torch.cuda.is_available():
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Initial GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Get device and config
    device = DynamicModelConfig.get_device()
    config = DynamicModelConfig.select_config()
    
    print(f"\nDevice: {device}")
    print(f"Config: {config}")
    
    # Create model
    model = GameTransformer(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Check GPU memory after model creation
    if torch.cuda.is_available():
        print(f"GPU memory after model creation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Create environment
    env = Gym2048Env()
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    
    # Get initial state
    state_tuple = env.reset()
    if isinstance(state_tuple, tuple):
        state = state_tuple[0]  # Gymnasium returns (state, info)
    else:
        state = state_tuple
    
    print(f"Initial state shape: {state.shape}")
    
    # Convert to tensor and move to device
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    print(f"State tensor device: {state_tensor.device}")
    
    # Forward pass
    with torch.no_grad():
        start_time = time.time()
        policy_logits, value = model(state_tensor)
        forward_time = time.time() - start_time
    
    print(f"Forward pass time: {forward_time:.4f} seconds")
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")
    
    # Check GPU memory after forward pass
    if torch.cuda.is_available():
        print(f"GPU memory after forward pass: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Test training step
    print("\n=== Testing Training Step ===")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy target
    target_value = torch.tensor([[1.0]]).to(device)
    target_policy = torch.tensor([[0.25, 0.25, 0.25, 0.25]]).to(device)
    
    # Training step
    optimizer.zero_grad()
    
    start_time = time.time()
    policy_logits, value = model(state_tensor)
    
    # Compute losses
    value_loss = torch.nn.functional.mse_loss(value, target_value)
    policy_loss = torch.nn.functional.cross_entropy(policy_logits, torch.argmax(target_policy, dim=1))
    total_loss = value_loss + policy_loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    training_time = time.time() - start_time
    
    print(f"Training step time: {training_time:.4f} seconds")
    print(f"Value loss: {value_loss.item():.6f}")
    print(f"Policy loss: {policy_loss.item():.6f}")
    print(f"Total loss: {total_loss.item():.6f}")
    
    # Check GPU memory after training
    if torch.cuda.is_available():
        print(f"GPU memory after training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Test multiple steps to see if GPU usage increases
    print("\n=== Testing Multiple Training Steps ===")
    
    for i in range(5):
        optimizer.zero_grad()
        
        # Get new state
        action = env.action_space.sample()
        step_result = env.step(action)
        if isinstance(step_result, tuple) and len(step_result) >= 4:
            state, reward, done, info = step_result[:4]
        else:
            state, reward, done, info = step_result, 0, False, {}
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Forward and backward
        policy_logits, value = model(state_tensor)
        value_loss = torch.nn.functional.mse_loss(value, target_value)
        policy_loss = torch.nn.functional.cross_entropy(policy_logits, torch.argmax(target_policy, dim=1))
        total_loss = value_loss + policy_loss
        
        total_loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"Step {i+1}: GPU memory = {gpu_memory:.2f} GB, Loss = {total_loss.item():.6f}")
    
    print("\n=== Test Complete ===")
    
    # Final GPU memory check
    if torch.cuda.is_available():
        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Final GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Check if we're actually using GPU
        if torch.cuda.memory_allocated() > 0:
            print("✅ GPU is being used!")
        else:
            print("❌ GPU is NOT being used!")

if __name__ == "__main__":
    test_gpu_usage() 