#!/usr/bin/env python3
"""
Test to verify the new 'tiny' model configuration
"""

import requests
import time
import json

def test_tiny_model():
    """Test if the tiny model starts and works correctly"""
    print("🧪 Testing tiny model configuration...")
    
    try:
        # Check initial status
        response = requests.get("http://localhost:8000/training/status", timeout=5)
        if response.status_code != 200:
            print(f"❌ Failed to get initial status: {response.status_code}")
            return False
            
        initial_status = response.json()
        print(f"📊 Initial status: {initial_status}")
        
        # Start training with tiny model
        print("🚀 Starting training with tiny model...")
        response = requests.post(
            "http://localhost:8000/training/start",
            json={"model_size": "tiny"},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to start training with tiny model: {response.status_code}")
            return False
            
        start_result = response.json()
        print(f"✅ Training started with tiny model: {start_result}")
        
        # Wait a bit and check progress
        print("⏳ Waiting 10 seconds for tiny model training to progress...")
        time.sleep(10)
        
        response = requests.get("http://localhost:8000/training/status", timeout=5)
        if response.status_code != 200:
            print(f"❌ Failed to get status after wait: {response.status_code}")
            return False
            
        final_status = response.json()
        print(f"📊 Final status: {final_status}")
        
        # Check if training is progressing
        if final_status.get('is_training', False) and final_status.get('current_episode', 0) > 0:
            print("✅ Tiny model training is working! Episode count increased.")
            return True
        else:
            print("❌ Tiny model training is not progressing properly.")
            print(f"  is_training: {final_status.get('is_training', False)}")
            print(f"  current_episode: {final_status.get('current_episode', 0)}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing tiny model: {e}")
        return False

def test_tiny_model_speed():
    """Test if the tiny model is faster than other models"""
    print("\n🧪 Testing tiny model speed...")
    
    try:
        # Start training with tiny model
        print("🚀 Starting tiny model training for speed test...")
        response = requests.post(
            "http://localhost:8000/training/start",
            json={"model_size": "tiny"},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to start tiny model training: {response.status_code}")
            return False
        
        # Monitor progress for 20 seconds
        print("⏳ Monitoring tiny model progress for 20 seconds...")
        start_time = time.time()
        episode_counts = []
        timestamps = []
        
        for i in range(4):  # Check every 5 seconds for 20 seconds total
            time.sleep(5)
            elapsed = time.time() - start_time
            
            response = requests.get("http://localhost:8000/training/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                episode_count = status.get('current_episode', 0)
                episode_counts.append(episode_count)
                timestamps.append(elapsed)
                print(f"  {elapsed:.1f}s: Episode {episode_count}")
            else:
                print(f"  {elapsed:.1f}s: Failed to get status")
        
        # Calculate speed metrics
        if len(episode_counts) >= 2:
            total_episodes = episode_counts[-1] - episode_counts[0]
            total_time = timestamps[-1] - timestamps[0]
            
            if total_time > 0:
                episodes_per_second = total_episodes / total_time
                episodes_per_minute = episodes_per_second * 60
                
                print(f"\n📈 Tiny Model Speed Metrics:")
                print(f"  Total episodes: {total_episodes}")
                print(f"  Total time: {total_time:.1f}s")
                print(f"  Episodes/second: {episodes_per_second:.2f}")
                print(f"  Episodes/minute: {episodes_per_minute:.1f}")
                
                # Tiny model should be faster than previous tests
                if episodes_per_second > 0.5:  # Should be faster than the 0.23 we saw before
                    print("✅ Tiny model is faster than expected!")
                    return True
                else:
                    print("⚠️ Tiny model speed could be better")
                    return False
            else:
                print("❌ No time elapsed")
                return False
        else:
            print("❌ Not enough data points to measure speed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing tiny model speed: {e}")
        return False

def main():
    """Run the tiny model tests"""
    print("🚀 Testing Tiny Model Configuration")
    print("=" * 50)
    
    # Test 1: Basic functionality
    basic_ok = test_tiny_model()
    
    # Test 2: Speed comparison
    speed_ok = test_tiny_model_speed()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Tiny Model Test Results:")
    print(f"  Basic Functionality: {'✅ PASSED' if basic_ok else '❌ FAILED'}")
    print(f"  Speed Performance: {'✅ PASSED' if speed_ok else '❌ FAILED'}")
    
    if basic_ok and speed_ok:
        print("\n🎉 Tiny model is working correctly!")
        print("\n💡 The tiny model features:")
        print("   - ~51k parameters (much smaller than other models)")
        print("   - 1 layer, 4 experts, 60 dimensions")
        print("   - Should be faster for development and testing")
        print("   - Available in the frontend dropdown")
        return 0
    else:
        print("\n💥 Tiny model needs more work.")
        return 1

if __name__ == "__main__":
    exit(main()) 