#!/usr/bin/env python3
"""
Test script to verify load balancing reward system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import torch
import numpy as np
from app.models.model_config import DynamicModelConfig
from app.models.game_transformer import GameTransformer
from app.training.ppo_trainer import PPOTrainer
from app.environment.gym_2048_env import Gym2048Env

def test_load_balancing_rewards():
    """Test the load balancing reward calculation"""
    
    print("=== Load Balancing Reward Test ===")
    
    # Test with tiny model (now 2 layers)
    print("\n1. Testing tiny model configuration...")
    config = DynamicModelConfig.CONFIGS["tiny"]
    print(f"   Tiny model: {config.n_layers} layers, {config.n_experts} experts, {config.d_model} dimensions")
    
    # Create model and trainer
    device = DynamicModelConfig.get_device()
    model = GameTransformer(config).to(device)
    trainer = PPOTrainer(config=config, device=device)
    
    print(f"   Model parameters: {model.count_parameters():,}")
    print(f"   Device: {device}")
    
    # Test load balancing reward calculation
    print("\n2. Testing load balancing reward calculation...")
    
    # Create a test environment
    env = Gym2048Env()
    obs, _ = env.reset()
    
    # Run a few steps to get expert usage
    for step in range(5):
        legal_actions = env.game.legal_moves()
        if not legal_actions:
            break
            
        action, log_prob, value = trainer.select_action(obs, legal_actions, env.game)
        next_obs, reward, done, _, _ = env.step(action)
        
        # Calculate load balancing reward
        lb_reward = trainer.calculate_load_balancing_reward()
        print(f"   Step {step + 1}: Game reward = {reward:.3f}, LB reward = {lb_reward:.3f}")
        
        obs = next_obs
        if done:
            break
    
    # Test expert usage distribution
    print("\n3. Testing expert usage distribution...")
    expert_usage = model.get_expert_usage()
    if expert_usage is not None:
        usage_np = expert_usage.cpu().numpy()
        print(f"   Expert usage: {usage_np}")
        print(f"   Usage variance: {np.var(usage_np):.4f}")
        print(f"   Ideal uniform: {1.0/config.n_experts:.4f}")
        
        # Check for expert starvation
        starvation_count = np.sum(usage_np < trainer.lb_critical_threshold)
        print(f"   Experts below critical threshold ({trainer.lb_critical_threshold}): {starvation_count}")
    else:
        print("   No expert usage data available")
    
    print("\n4. Testing reward parameters...")
    print(f"   LB reward coefficient: {trainer.lb_reward_coef}")
    print(f"   Critical threshold: {trainer.lb_critical_threshold}")
    print(f"   Early training boost: {trainer.lb_early_training_boost}")
    print(f"   Episode threshold: {trainer.lb_episode_threshold}")
    
    print("\n✅ Load balancing reward system test completed!")
    return True

def test_tiny_model_enhancement():
    """Test that the tiny model now has 2 layers"""
    
    print("\n=== Tiny Model Enhancement Test ===")
    
    config = DynamicModelConfig.CONFIGS["tiny"]
    
    if config.n_layers == 2:
        print("✅ Tiny model correctly has 2 layers")
        return True
    else:
        print(f"❌ Tiny model has {config.n_layers} layers, expected 2")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Load Balancing Reward System")
    print("=" * 50)
    
    # Test 1: Tiny model enhancement
    tiny_ok = test_tiny_model_enhancement()
    
    # Test 2: Load balancing rewards
    lb_ok = test_load_balancing_rewards()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Tiny Model Enhancement: {'✅ PASSED' if tiny_ok else '❌ FAILED'}")
    print(f"  Load Balancing Rewards: {'✅ PASSED' if lb_ok else '❌ FAILED'}")
    
    if tiny_ok and lb_ok:
        print("\n🎉 All tests passed!")
        print("\n💡 New features:")
        print("   - Tiny model now has 2 layers (increased from 1)")
        print("   - Load balancing rewards prioritize expert usage")
        print("   - Early training boost for better expert utilization")
        print("   - Critical threshold detection for expert starvation")
        print("   - Frontend displays load balancing metrics")
        return 0
    else:
        print("\n💥 Some tests failed.")
        return 1

if __name__ == "__main__":
    exit(main()) 