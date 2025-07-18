#!/usr/bin/env python3
"""
Test script to isolate training manager issues
"""

import torch
import numpy as np
import sys
import os
import asyncio
import time

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

class MockWebSocketManager:
    """Mock WebSocket manager for testing"""
    def __init__(self):
        self.messages = []
    
    async def broadcast(self, message, priority="normal"):
        self.messages.append(message)
        print(f"Broadcast: {message.get('type', 'unknown')}")
    
    async def broadcast_high_priority(self, message):
        self.messages.append(message)
        print(f"High priority broadcast: {message.get('type', 'unknown')}")

def test_training_manager():
    """Test the training manager for extended periods"""
    print("Testing training manager...")
    
    try:
        from app.training.training_manager import TrainingManager
        
        # Create mock WebSocket manager
        ws_manager = MockWebSocketManager()
        
        # Create training manager with fewer environments for testing
        manager = TrainingManager(ws_manager, n_envs=2)
        
        print("Training manager created successfully")
        
        # Test multiple episodes
        print("Starting training for 10 episodes...")
        manager.start()
        
        # Let it run for a few episodes
        for i in range(10):
            if not manager.is_training:
                print(f"Training stopped at episode {i}")
                break
            
            # Wait a bit for each episode
            time.sleep(0.1)
            
            print(f"Episode {i}: current_episode={manager.current_episode}, is_training={manager.is_training}")
            
            # Check if we've made progress
            if manager.current_episode > 0:
                print(f"✅ Training is progressing: episode {manager.current_episode}")
            else:
                print(f"⚠️ No progress yet...")
        
        # Stop training
        manager.stop_sync()
        print("✅ Training manager test completed successfully")
        return True
        
    except Exception as e:
        print(f"Training manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage over time"""
    print("\nTesting memory usage...")
    
    try:
        import psutil
        import torch
        
        initial_memory = psutil.virtual_memory().used / (1024**3)  # GB
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        else:
            initial_gpu_memory = 0
        
        print(f"Initial RAM usage: {initial_memory:.2f} GB")
        print(f"Initial GPU memory: {initial_gpu_memory:.2f} GB")
        
        # Create and run some training
        from app.training.training_manager import TrainingManager
        ws_manager = MockWebSocketManager()
        manager = TrainingManager(ws_manager, n_envs=1)
        
        manager.start()
        
        # Run for a few episodes
        for i in range(5):
            time.sleep(0.2)
            current_memory = psutil.virtual_memory().used / (1024**3)
            if torch.cuda.is_available():
                current_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            else:
                current_gpu_memory = 0
            
            print(f"Episode {i}: RAM={current_memory:.2f}GB, GPU={current_gpu_memory:.2f}GB")
        
        manager.stop_sync()
        
        final_memory = psutil.virtual_memory().used / (1024**3)
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        else:
            final_gpu_memory = 0
        
        print(f"Final RAM usage: {final_memory:.2f} GB (change: {final_memory - initial_memory:+.2f} GB)")
        print(f"Final GPU memory: {final_gpu_memory:.2f} GB (change: {final_gpu_memory - initial_gpu_memory:+.2f} GB)")
        
        print("✅ Memory usage test completed")
        return True
        
    except Exception as e:
        print(f"Memory usage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run training manager tests"""
    print("Running training manager tests...")
    
    tests = [
        test_training_manager,
        test_memory_usage
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print(f"\n📊 Test Results:")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"  {i+1}. {test.__name__}: {status}")
    
    if all(results):
        print("\nAll tests passed! The issue might be in the full application setup.")
    else:
        print("\nSome tests failed. Check the specific test that failed above.")

if __name__ == "__main__":
    main() 