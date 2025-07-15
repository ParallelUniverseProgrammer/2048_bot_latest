#!/usr/bin/env python3
"""
Simple test to reproduce the device compatibility error
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.models.checkpoint_playback import CheckpointPlayback
from app.models.checkpoint_metadata import CheckpointManager

def test_device_compatibility():
    """Test device compatibility with existing checkpoint"""
    
    # Create checkpoint manager and playback
    cm = CheckpointManager()
    playback = CheckpointPlayback(cm)
    
    print("Loading checkpoint_episode_1400...")
    success = playback.load_checkpoint('checkpoint_episode_1400')
    print(f"Load success: {success}")
    
    if success:
        print("Playing one game...")
        result = playback.play_single_game()
        if isinstance(result, dict):
            print(f"Result keys: {list(result.keys())}")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Final score: {result.get('final_score', 'unknown')}")
                print(f"Steps: {result.get('steps', 'unknown')}")
        else:
            print(f"Result: {result}")
    
    return success

if __name__ == "__main__":
    test_device_compatibility() 