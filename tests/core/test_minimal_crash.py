#!/usr/bin/env python3
"""
Minimal test script to isolate the crash issue
"""

import torch
import numpy as np
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_model_creation():
    """Test if model creation works"""
    print("Testing model creation...")
    try:
        from app.models.model_config import DynamicModelConfig
        from app.models.game_transformer import GameTransformer
        
        config = DynamicModelConfig.select_config(target_vram=1.0)  # Use tiny config
        print(f"Config: {config}")
        
        model = GameTransformer(config)
        print(f"Model created successfully with {model.count_parameters():,} parameters")
        
        # Test forward pass
        batch_size = 1
        board = torch.randint(0, 17, (batch_size, 4, 4), dtype=torch.float32)
        print(f"Input board shape: {board.shape}")
        
        with torch.no_grad():
            policy_logits, value = model(board)
            print(f"Policy logits shape: {policy_logits.shape}")
            print(f"Value shape: {value.shape}")
        
        print("✅ Model forward pass successful")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment():
    """Test if environment works"""
    print("\nTesting environment...")
    try:
        from app.environment.gym_2048_env import Gym2048Env
        
        env = Gym2048Env()
        obs, _ = env.reset()
        print(f"Environment reset successful, obs shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            legal_actions = env.game.legal_moves()
            if not legal_actions:
                break
            action = legal_actions[0]  # Take first legal action
            obs, reward, done, _, _ = env.step(action)
            print(f"Step {i}: action={action}, reward={reward}, done={done}")
            if done:
                break
        
        print("✅ Environment test successful")
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer_creation():
    """Test if trainer creation works"""
    print("\nTesting trainer creation...")
    try:
        from app.training.ppo_trainer import PPOTrainer
        from app.models.model_config import DynamicModelConfig
        
        config = DynamicModelConfig.select_config(target_vram=1.0)
        trainer = PPOTrainer(config=config)
        print(f"Trainer created successfully")
        
        # Test action selection
        env = Gym2048Env()
        obs, _ = env.reset()
        legal_actions = env.game.legal_moves()
        
        action, log_prob, value = trainer.select_action(obs, legal_actions, env.game)
        print(f"Action selection successful: action={action}, log_prob={log_prob}, value={value}")
        
        print("✅ Trainer test successful")
        return True
        
    except Exception as e:
        print(f"❌ Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_load_balancing():
    """Test load balancing calculations"""
    print("\nTesting load balancing calculations...")
    try:
        from app.training.ppo_trainer import PPOTrainer
        from app.models.model_config import DynamicModelConfig
        
        config = DynamicModelConfig.select_config(target_vram=1.0)
        trainer = PPOTrainer(config=config)
        
        # Test load balancing reward calculation
        lb_reward = trainer.calculate_load_balancing_reward()
        print(f"Load balancing reward: {lb_reward}")
        
        print("✅ Load balancing test successful")
        return True
        
    except Exception as e:
        print(f"❌ Load balancing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_episode():
    """Test a single training episode"""
    print("\nTesting single training episode...")
    try:
        from app.training.ppo_trainer import PPOTrainer
        from app.models.model_config import DynamicModelConfig
        from app.environment.gym_2048_env import Gym2048Env
        
        config = DynamicModelConfig.select_config(target_vram=1.0)
        trainer = PPOTrainer(config=config)
        env = Gym2048Env()
        
        result = trainer.train_episode(env)
        print(f"Episode result: {result}")
        
        print("✅ Single episode test successful")
        return True
        
    except Exception as e:
        print(f"❌ Single episode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔍 Running minimal crash tests...")
    
    tests = [
        test_model_creation,
        test_environment,
        test_trainer_creation,
        test_load_balancing,
        test_single_episode
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print(f"\n📊 Test Results:")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i+1}. {test.__name__}: {status}")
    
    if all(results):
        print("\n🎉 All tests passed! The issue might be in the training loop or threading.")
    else:
        print("\n💥 Some tests failed. Check the specific test that failed above.")

if __name__ == "__main__":
    main() 