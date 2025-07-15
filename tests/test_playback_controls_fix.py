#!/usr/bin/env python3
"""
Test script to verify playback control issues:
1. Pause button doesn't turn into resume button
2. Reset briefly shows idle state
"""

import asyncio
import requests
import time
import json
from typing import Dict, Any

class PlaybackControlTest:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_checkpoint_id = None
        
    def get_playback_status(self) -> Dict[str, Any]:
        """Get current playback status"""
        try:
            response = requests.get(f"{self.base_url}/checkpoints/playback/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get status: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error getting status: {e}")
            return {}
    
    def find_test_checkpoint(self) -> bool:
        """Find a checkpoint to test with"""
        try:
            response = requests.get(f"{self.base_url}/checkpoints", timeout=5)
            if response.status_code == 200:
                checkpoints = response.json()
                if checkpoints and len(checkpoints) > 0:
                    self.test_checkpoint_id = checkpoints[0]['id']
                    print(f"Using test checkpoint: {self.test_checkpoint_id}")
                    return True
                else:
                    print("No checkpoints available")
                    return False
            else:
                print(f"Failed to get checkpoints: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error getting checkpoints: {e}")
            return False
    
    def start_playback(self) -> bool:
        """Start playback of test checkpoint"""
        if not self.test_checkpoint_id:
            print("No test checkpoint available")
            return False
            
        try:
            response = requests.post(
                f"{self.base_url}/checkpoints/{self.test_checkpoint_id}/playback/start",
                timeout=10
            )
            if response.status_code == 200:
                print("Playback started successfully")
                return True
            else:
                print(f"Failed to start playback: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error starting playback: {e}")
            return False
    
    def pause_playback(self) -> bool:
        """Pause playback"""
        try:
            response = requests.post(f"{self.base_url}/checkpoints/playback/pause", timeout=5)
            if response.status_code == 200:
                print("Playback paused successfully")
                return True
            else:
                print(f"Failed to pause playback: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error pausing playback: {e}")
            return False
    
    def resume_playback(self) -> bool:
        """Resume playback"""
        try:
            response = requests.post(f"{self.base_url}/checkpoints/playback/resume", timeout=5)
            if response.status_code == 200:
                print("Playback resumed successfully")
                return True
            else:
                print(f"Failed to resume playback: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error resuming playback: {e}")
            return False
    
    def stop_playback(self) -> bool:
        """Stop playback"""
        try:
            response = requests.post(f"{self.base_url}/checkpoints/playback/stop", timeout=5)
            if response.status_code == 200:
                print("Playback stopped successfully")
                return True
            else:
                print(f"Failed to stop playback: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error stopping playback: {e}")
            return False
    
    def test_pause_resume_button_logic(self):
        """Test the pause/resume button logic that should be used in frontend"""
        print("\n=== Testing Pause/Resume Button Logic ===")
        
        # Get initial status
        status = self.get_playback_status()
        print(f"Initial status: is_playing={status.get('is_playing')}, is_paused={status.get('is_paused')}")
        
        # Test button logic
        is_playing = status.get('is_playing', False)
        is_paused = status.get('is_paused', False)
        
        # Current logic (incorrect)
        current_logic = is_playing
        print(f"Current logic (is_playing only): {current_logic} -> {'Pause' if current_logic else 'Resume'}")
        
        # Correct logic
        correct_logic = is_playing and not is_paused
        print(f"Correct logic (is_playing AND NOT is_paused): {correct_logic} -> {'Pause' if correct_logic else 'Resume'}")
        
        if current_logic != correct_logic:
            print("❌ ISSUE DETECTED: Button logic is incorrect!")
            return False
        else:
            print("✅ Button logic is correct")
            return True
    
    def test_pause_resume_cycle(self):
        """Test full pause/resume cycle"""
        print("\n=== Testing Pause/Resume Cycle ===")
        
        # Start playback
        if not self.start_playback():
            return False
        
        # Wait for playback to start
        time.sleep(2)
        
        # Check initial state
        status = self.get_playback_status()
        print(f"After start: is_playing={status.get('is_playing')}, is_paused={status.get('is_paused')}")
        
        # Pause playback
        if not self.pause_playback():
            return False
        
        time.sleep(1)
        
        # Check paused state
        status = self.get_playback_status()
        print(f"After pause: is_playing={status.get('is_playing')}, is_paused={status.get('is_paused')}")
        
        # Test button logic at this point
        is_playing = status.get('is_playing', False)
        is_paused = status.get('is_paused', False)
        
        current_logic = is_playing
        correct_logic = is_playing and not is_paused
        
        print(f"Button should show: {'Pause' if correct_logic else 'Resume'}")
        print(f"Current logic shows: {'Pause' if current_logic else 'Resume'}")
        
        if current_logic != correct_logic:
            print("❌ ISSUE: Button shows wrong state when paused!")
            issue_found = True
        else:
            print("✅ Button shows correct state when paused")
            issue_found = False
        
        # Resume playback
        if not self.resume_playback():
            return False
        
        time.sleep(1)
        
        # Check resumed state
        status = self.get_playback_status()
        print(f"After resume: is_playing={status.get('is_playing')}, is_paused={status.get('is_paused')}")
        
        # Stop playback
        self.stop_playback()
        
        return issue_found
    
    def test_reset_behavior(self):
        """Test reset behavior - should not briefly show idle"""
        print("\n=== Testing Reset Behavior ===")
        
        # Start playback
        if not self.start_playback():
            return False
        
        # Wait for playback to start
        time.sleep(2)
        
        # Check initial state
        status = self.get_playback_status()
        print(f"Initial state: is_playing={status.get('is_playing')}, is_paused={status.get('is_paused')}")
        
        # Start new game (reset)
        print("Starting new game...")
        start_time = time.time()
        
        if not self.start_playback():
            return False
        
        # Monitor status for brief idle state
        idle_detected = False
        for i in range(10):  # Check for 1 second
            time.sleep(0.1)
            status = self.get_playback_status()
            is_playing = status.get('is_playing', False)
            is_paused = status.get('is_paused', False)
            
            if not is_playing and not is_paused:
                idle_detected = True
                print(f"❌ IDLE STATE DETECTED at {time.time() - start_time:.2f}s")
                break
        
        if not idle_detected:
            print("✅ No idle state detected during reset")
        
        # Wait for playback to stabilize
        time.sleep(2)
        
        # Check final state
        status = self.get_playback_status()
        print(f"Final state: is_playing={status.get('is_playing')}, is_paused={status.get('is_paused')}")
        
        # Stop playback
        self.stop_playback()
        
        return idle_detected
    
    def run_all_tests(self):
        """Run all tests"""
        print("Starting Playback Control Tests...")
        
        # Check if backend is available
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                print("❌ Backend not available")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to backend: {e}")
            return False
        
        print("✅ Backend is available")
        
        # Find test checkpoint
        if not self.find_test_checkpoint():
            print("❌ No test checkpoint available")
            return False
        
        # Run tests
        issues_found = []
        
        # Test 1: Button logic
        if not self.test_pause_resume_button_logic():
            issues_found.append("Button logic issue")
        
        # Test 2: Pause/resume cycle
        if self.test_pause_resume_cycle():
            issues_found.append("Pause/resume cycle issue")
        
        # Test 3: Reset behavior
        if self.test_reset_behavior():
            issues_found.append("Reset idle state issue")
        
        # Summary
        print(f"\n=== Test Summary ===")
        if issues_found:
            print(f"❌ Issues found: {', '.join(issues_found)}")
            return False
        else:
            print("✅ All tests passed")
            return True

def main():
    """Main test function"""
    test = PlaybackControlTest()
    success = test.run_all_tests()
    
    if success:
        print("\n🎉 All playback control tests passed!")
        exit(0)
    else:
        print("\n💥 Playback control tests failed!")
        exit(1)

if __name__ == "__main__":
    main() 