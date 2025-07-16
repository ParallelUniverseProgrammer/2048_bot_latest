#!/usr/bin/env python3
"""
Test to verify training fix works
"""

import requests
import time
import json

def test_training_fix():
    """Test if training starts and progresses properly after the fix"""
    print("🧪 Testing training fix...")
    
    try:
        # Check initial status
        response = requests.get("http://localhost:8000/training/status", timeout=5)
        if response.status_code != 200:
            print(f"❌ Failed to get initial status: {response.status_code}")
            return False
            
        initial_status = response.json()
        print(f"📊 Initial status: {initial_status}")
        
        # Start training
        print("🚀 Starting training...")
        response = requests.post(
            "http://localhost:8000/training/start",
            json={"model_size": "small"},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to start training: {response.status_code}")
            return False
            
        start_result = response.json()
        print(f"✅ Training started: {start_result}")
        
        # Wait a bit and check progress
        print("⏳ Waiting 15 seconds for training to progress...")
        time.sleep(15)
        
        response = requests.get("http://localhost:8000/training/status", timeout=5)
        if response.status_code != 200:
            print(f"❌ Failed to get status after wait: {response.status_code}")
            return False
            
        final_status = response.json()
        print(f"📊 Final status: {final_status}")
        
        # Check if training is progressing
        if final_status.get('is_training', False) and final_status.get('current_episode', 0) > 0:
            print("✅ Training is working! Episode count increased.")
            return True
        else:
            print("❌ Training is not progressing properly.")
            print(f"  is_training: {final_status.get('is_training', False)}")
            print(f"  current_episode: {final_status.get('current_episode', 0)}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing training fix: {e}")
        return False

def main():
    """Run the training fix test"""
    print("🚀 Testing Training Fix")
    print("=" * 40)
    
    success = test_training_fix()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Training fix is working correctly!")
        print("\n💡 The training should now:")
        print("   - Start properly when you click 'Start Training'")
        print("   - Show episode count increasing in the frontend")
        print("   - Display training metrics and progress")
        return 0
    else:
        print("💥 Training fix needs more work.")
        return 1

if __name__ == "__main__":
    exit(main()) 