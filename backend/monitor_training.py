#!/usr/bin/env python3
"""Monitor training progress"""

import requests
import time
import json

def monitor_training():
    """Monitor training progress for a few minutes"""
    print("Monitoring training progress...")
    print("=" * 50)
    
    for i in range(15):  # Monitor for ~30 seconds
        try:
            response = requests.get('http://localhost:8000/training/status')
            status = response.json()
            
            print(f"Episode {status['current_episode']}/{status['total_episodes']} - Training: {status['is_training']}")
            
            if not status['is_training']:
                print("Training completed!")
                break
                
            time.sleep(2)
            
        except Exception as e:
            print(f"Error monitoring: {e}")
            break
    
    print("Monitoring complete!")

if __name__ == "__main__":
    monitor_training() 