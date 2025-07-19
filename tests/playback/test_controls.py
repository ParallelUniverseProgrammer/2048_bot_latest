#!/usr/bin/env python3
"""
Playback Controls Test
=====================

This test verifies playback control functionality and identifies UI issues:
- Pause button doesn't turn into resume button correctly
- Reset briefly shows idle state instead of smooth transition
- Button logic validation for proper state management
- Full pause/resume cycle testing
- Reset behavior verification

This test is critical for ensuring smooth user experience during playback.
"""

import asyncio
import requests
import time
import json
from typing import Dict, Any
import sys

from tests.utilities.test_utils import TestLogger

class PlaybackControlTester:
    """Test class for playback control verification"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.logger = TestLogger()
        self.base_url = base_url
        self.test_checkpoint_id = None
        
    def get_playback_status(self) -> Dict[str, Any]:
        """Get current playback status"""
        try:
            response = requests.get(f"{self.base_url}/checkpoints/playback/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get status: {response.status_code}")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {}
    
    def find_test_checkpoint(self) -> bool:
        """Find a checkpoint to test with"""
        try:
            response = requests.get(f"{self.base_url}/checkpoints", timeout=5)
            if response.status_code == 200:
                checkpoints = response.json()
                if checkpoints and len(checkpoints) > 0:
                    self.test_checkpoint_id = checkpoints[0]['id']
                    self.logger.ok(f"Using test checkpoint: {self.test_checkpoint_id}")
                    return True
                else:
                    self.logger.error("No checkpoints available")
                    return False
            else:
                self.logger.error(f"Failed to get checkpoints: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Error getting checkpoints: {e}")
            return False
    
    def start_playback(self) -> bool:
        """Start playback of test checkpoint"""
        if not self.test_checkpoint_id:
            self.logger.error("No test checkpoint available")
            return False
            
        try:
            response = requests.post(
                f"{self.base_url}/checkpoints/{self.test_checkpoint_id}/playback/start",
                timeout=10
            )
            if response.status_code == 200:
                self.logger.ok("Playback started successfully")
                return True
            else:
                self.logger.error(f"Failed to start playback: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Error starting playback: {e}")
            return False
    
    def pause_playback(self) -> bool:
        """Pause playback"""
        try:
            response = requests.post(f"{self.base_url}/checkpoints/playback/pause", timeout=5)
            if response.status_code == 200:
                self.logger.ok("Playback paused successfully")
                return True
            else:
                self.logger.error(f"Failed to pause playback: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Error pausing playback: {e}")
            return False
    
    def resume_playback(self) -> bool:
        """Resume playback"""
        try:
            response = requests.post(f"{self.base_url}/checkpoints/playback/resume", timeout=5)
            if response.status_code == 200:
                self.logger.ok("Playback resumed successfully")
                return True
            else:
                self.logger.error(f"Failed to resume playback: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Error resuming playback: {e}")
            return False
    
    def stop_playback(self) -> bool:
        """Stop playback"""
        try:
            response = requests.post(f"{self.base_url}/checkpoints/playback/stop", timeout=5)
            if response.status_code == 200:
                self.logger.ok("Playback stopped successfully")
                return True
            else:
                self.logger.error(f"Failed to stop playback: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Error stopping playback: {e}")
            return False
    
    def test_pause_resume_button_logic(self) -> bool:
        """Test the pause/resume button logic that should be used in frontend"""
        self.logger.banner("Testing Pause/Resume Button Logic", 60)
        
        # Get initial status
        status = self.get_playback_status()
        self.logger.info(f"Initial status: is_playing={status.get('is_playing')}, is_paused={status.get('is_paused')}")
        
        # Test button logic
        is_playing = status.get('is_playing', False)
        is_paused = status.get('is_paused', False)
        
        # Current logic (incorrect)
        current_logic = is_playing
        self.logger.info(f"Current logic (is_playing only): {current_logic} -> {'Pause' if current_logic else 'Resume'}")
        
        # Correct logic
        correct_logic = is_playing and not is_paused
        self.logger.info(f"Correct logic (is_playing AND NOT is_paused): {correct_logic} -> {'Pause' if correct_logic else 'Resume'}")
        
        if current_logic != correct_logic:
            self.logger.error("Button logic is incorrect!")
            return False
        else:
            self.logger.ok("Button logic is correct")
            return True
    
    def test_pause_resume_cycle(self) -> bool:
        """Test full pause/resume cycle"""
        self.logger.banner("Testing Pause/Resume Cycle", 60)
        
        try:
            # Start playback
            if not self.start_playback():
                return False
            
            # Wait for playback to start
            time.sleep(2)
            
            # Check initial state
            status = self.get_playback_status()
            self.logger.info(f"After start: is_playing={status.get('is_playing')}, is_paused={status.get('is_paused')}")
            
            # Pause playback
            if not self.pause_playback():
                return False
            
            time.sleep(1)
            
            # Check paused state
            status = self.get_playback_status()
            self.logger.info(f"After pause: is_playing={status.get('is_playing')}, is_paused={status.get('is_paused')}")
            
            # Test button logic at this point
            is_playing = status.get('is_playing', False)
            is_paused = status.get('is_paused', False)
            
            current_logic = is_playing
            correct_logic = is_playing and not is_paused
            
            self.logger.info(f"Button should show: {'Pause' if correct_logic else 'Resume'}")
            self.logger.info(f"Current logic shows: {'Pause' if current_logic else 'Resume'}")
            
            if current_logic != correct_logic:
                self.logger.error("Button shows wrong state when paused!")
                issue_found = True
            else:
                self.logger.ok("Button shows correct state when paused")
                issue_found = False
            
            # Resume playback
            if not self.resume_playback():
                return False
            
            time.sleep(1)
            
            # Check resumed state
            status = self.get_playback_status()
            self.logger.info(f"After resume: is_playing={status.get('is_playing')}, is_paused={status.get('is_paused')}")
            
            # Stop playback
            self.stop_playback()
            
            return issue_found
            
        except Exception as e:
            self.logger.error(f"Pause/resume cycle test failed: {e}")
            return False
    
    def test_reset_behavior(self) -> bool:
        """Test reset behavior - should not briefly show idle"""
        self.logger.banner("Testing Reset Behavior", 60)
        
        try:
            # Start playback
            if not self.start_playback():
                return False
            
            # Wait for playback to start
            time.sleep(2)
            
            # Check initial state
            status = self.get_playback_status()
            self.logger.info(f"Initial state: is_playing={status.get('is_playing')}, is_paused={status.get('is_paused')}")
            
            # Start new game (reset)
            self.logger.info("Starting new game...")
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
                    self.logger.error(f"IDLE STATE DETECTED at {time.time() - start_time:.2f}s")
                    break
            
            if not idle_detected:
                self.logger.ok("No idle state detected during reset")
            
            # Wait for playback to stabilize
            time.sleep(2)
            
            # Check final state
            status = self.get_playback_status()
            self.logger.info(f"Final state: is_playing={status.get('is_playing')}, is_paused={status.get('is_paused')}")
            
            # Stop playback
            self.stop_playback()
            
            return idle_detected
            
        except Exception as e:
            self.logger.error(f"Reset behavior test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests"""
        self.logger.banner("Playback Control Tests", 60)
        
        try:
            # Check if backend is available
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code != 200:
                    self.logger.error("Backend not available")
                    return False
            except Exception as e:
                self.logger.error(f"Cannot connect to backend: {e}")
                return False
            
            self.logger.ok("Backend is available")
            
            # Find test checkpoint
            if not self.find_test_checkpoint():
                self.logger.error("No test checkpoint available")
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
            self.logger.banner("Test Summary", 60)
            if issues_found:
                self.logger.error(f"Issues found: {', '.join(issues_found)}")
                return False
            else:
                self.logger.ok("All tests passed")
                return True
                
        except Exception as e:
            self.logger.error(f"Playback control test failed: {e}")
            return False

def main():
    """Main entry point for playback control tests"""
    logger = TestLogger()
    logger.banner("Playback Control Test Suite", 60)
    
    tester = PlaybackControlTester()
    success = tester.run_all_tests()
    
    if success:
        logger.success("PLAYBACK CONTROL TESTS PASSED!")
    else:
        logger.error("PLAYBACK CONTROL TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main() 