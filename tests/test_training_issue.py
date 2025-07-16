#!/usr/bin/env python3
"""
Test to identify training issues
"""

import requests
import time
import json

def test_training_start():
    """Test if training starts properly"""
    print("🧪 Testing training start...")
    
    try:
        # Start training
        response = requests.post(
            "http://localhost:8000/training/start",
            json={"model_size": "small"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training started: {result}")
            return True
        else:
            print(f"❌ Failed to start training: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting training: {e}")
        return False

def test_training_progress():
    """Test if training progresses"""
    print("\n🧪 Testing training progress...")
    
    try:
        # Check initial status
        response = requests.get("http://localhost:8000/training/status", timeout=5)
        if response.status_code != 200:
            print(f"❌ Failed to get training status: {response.status_code}")
            return False
            
        initial_status = response.json()
        print(f"📊 Initial status: {initial_status}")
        
        # Wait a bit and check again
        print("⏳ Waiting 10 seconds for training to progress...")
        time.sleep(10)
        
        response = requests.get("http://localhost:8000/training/status", timeout=5)
        if response.status_code != 200:
            print(f"❌ Failed to get training status after wait: {response.status_code}")
            return False
            
        final_status = response.json()
        print(f"📊 Final status: {final_status}")
        
        # Check if episode count increased
        if final_status.get('current_episode', 0) > initial_status.get('current_episode', 0):
            print("✅ Training is progressing!")
            return True
        else:
            print("❌ Training is not progressing - episode count didn't increase")
            return False
            
    except Exception as e:
        print(f"❌ Error testing training progress: {e}")
        return False

def main():
    """Run training tests"""
    print("🚀 Testing Training Issues")
    print("=" * 40)
    
    # Test 1: Start training
    start_ok = test_training_start()
    
    # Test 2: Check progress
    progress_ok = test_training_progress()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"  Training Start: {'✅ PASSED' if start_ok else '❌ FAILED'}")
    print(f"  Training Progress: {'✅ PASSED' if progress_ok else '❌ FAILED'}")
    
    if start_ok and progress_ok:
        print("\n🎉 Training is working correctly!")
        return 0
    else:
        print("\n💥 Training has issues that need to be fixed.")
        return 1

if __name__ == "__main__":
    exit(main()) 