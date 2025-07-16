#!/usr/bin/env python3
"""
Test to verify training speed optimizations
"""

import requests
import time
import json

def test_training_speed():
    """Test if training is faster with the optimizations"""
    print("🧪 Testing training speed optimizations...")
    
    try:
        # Check initial status
        response = requests.get("http://localhost:8000/training/status", timeout=5)
        if response.status_code != 200:
            print(f"❌ Failed to get initial status: {response.status_code}")
            return False
            
        initial_status = response.json()
        print(f"📊 Initial status: {initial_status}")
        
        # Start training
        print("🚀 Starting training with speed optimizations...")
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
        
        # Wait and check progress multiple times to measure speed
        print("⏳ Monitoring training progress for 30 seconds...")
        start_time = time.time()
        episode_counts = []
        timestamps = []
        
        for i in range(6):  # Check every 5 seconds for 30 seconds total
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
                
                print(f"\n📈 Speed Metrics:")
                print(f"  Total episodes: {total_episodes}")
                print(f"  Total time: {total_time:.1f}s")
                print(f"  Episodes/second: {episodes_per_second:.2f}")
                print(f"  Episodes/minute: {episodes_per_minute:.1f}")
                
                # Check if speed is reasonable (should be > 1 episode per second with optimizations)
                if episodes_per_second > 1.0:
                    print("✅ Training speed is good!")
                    return True
                else:
                    print("⚠️ Training speed could be better")
                    return False
            else:
                print("❌ No time elapsed")
                return False
        else:
            print("❌ Not enough data points to measure speed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing training speed: {e}")
        return False

def main():
    """Run the training speed test"""
    print("🚀 Testing Training Speed Optimizations")
    print("=" * 50)
    
    success = test_training_speed()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Training speed optimizations are working!")
        print("\n💡 The optimizations include:")
        print("   - Increased parallel environments (4 → 8)")
        print("   - Reduced buffer size (2048 → 1024)")
        print("   - Reduced PPO epochs (4 → 2)")
        print("   - Increased batch sizes")
        print("   - More frequent checkpoints (100 → 50 episodes)")
        print("   - Faster training loop (reduced sleep intervals)")
        return 0
    else:
        print("💥 Training speed needs more optimization.")
        return 1

if __name__ == "__main__":
    exit(main()) 