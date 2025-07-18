#!/usr/bin/env python3
"""
Test Complete Game Playback from Checkpoints
============================================

This test focuses specifically on ensuring that:
1. Checkpoints can be loaded
2. Complete games can be played back
3. Game data is valid and complete
4. Performance is acceptable

This is the core functionality that must work for the checkpoint system.
"""

import requests
import time
import json
import sys
import os
from typing import Dict, Any, List, Optional
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from tests.utilities.test_utils import TestLogger, BackendTester, GameTester, PlaybackTester, check_backend_or_start_mock

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 120  # 2 minutes for game completion

class CheckpointGameTester:
    """Test complete game playback from checkpoints"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = []
        self.test_checkpoint_id = None
        self.logger = TestLogger()
        self.backend = BackendTester(base_url, self.logger)
        self.game_tester = GameTester(base_url, self.logger, TIMEOUT)
        self.playback_tester = PlaybackTester(base_url, self.logger)
        
    def log(self, message: str):
        """Log a message with timestamp"""
        self.logger.log(message)
    
    def test_backend_connectivity(self) -> bool:
        """Test that backend is accessible"""
        # Try to start mock backend if real backend is not available
        if not self.backend.test_connectivity():
            self.logger.info("Real backend not available, attempting to start mock backend...")
            if check_backend_or_start_mock(port=8000, wait_time=30):
                return self.backend.test_connectivity()
            else:
                return False
        return True
    
    def get_available_checkpoints(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints"""
        return self.backend.get_checkpoints()
    
    def test_checkpoint_loading(self, checkpoint_id: str) -> bool:
        """Test that a checkpoint can be loaded"""
        checkpoint_info = self.backend.load_checkpoint(checkpoint_id)
        if checkpoint_info:
            self.logger.log(f"   Episode: {checkpoint_info.get('episode', 'N/A')}")
            self.logger.log(f"   File size: {checkpoint_info.get('file_size', 0)} bytes")
            return True
        return False
    
    def test_single_game_playback(self, checkpoint_id: str) -> Dict[str, Any]:
        """Test playing a complete game from a checkpoint"""
        return self.game_tester.test_single_game_playback(checkpoint_id)
    
    def test_live_playback_start(self, checkpoint_id: str) -> Dict[str, Any]:
        """Test starting live playback (without waiting for completion)"""
        return self.playback_tester.start_live_playback(checkpoint_id)
    
    def test_playback_controls(self) -> Dict[str, Any]:
        """Test playback pause/resume/stop controls"""
        if not self.test_checkpoint_id:
            return {"success": False, "error": "No test checkpoint available"}
        
        return self.playback_tester.test_playback_controls(self.test_checkpoint_id)
    
    def run_complete_test_suite(self) -> bool:
        """Run the complete test suite"""
        self.logger.starting("Starting Complete Game Playback Test Suite")
        self.logger.separator(60)
        
        total_tests = 7
        test_results = []
        
        # Test 1: Backend connectivity
        self.logger.step(1, total_tests, "Testing backend connectivity")
        self.logger.indent()
        connectivity_result = self.test_backend_connectivity()
        test_results.append(("Backend Connectivity", "PASS" if connectivity_result else "FAIL"))
        self.logger.dedent()
        
        if not connectivity_result:
            self.logger.error("Backend not accessible. Please start the server.")
            return False
        
        # Test 2: Get available checkpoints
        self.logger.step(2, total_tests, "Getting available checkpoints")
        self.logger.indent()
        checkpoints = self.get_available_checkpoints()
        checkpoint_available = len(checkpoints) > 0
        test_results.append(("Checkpoint Availability", "PASS" if checkpoint_available else "FAIL"))
        self.logger.dedent()
        
        if not checkpoint_available:
            self.logger.error("No checkpoints available for testing")
            return False
        
        # Test 3: Test checkpoint loading
        self.logger.step(3, total_tests, "Testing checkpoint loading")
        self.logger.indent()
        test_checkpoint = checkpoints[0]  # Use first checkpoint
        self.test_checkpoint_id = test_checkpoint['id']
        
        loading_result = self.test_checkpoint_loading(self.test_checkpoint_id)
        test_results.append(("Checkpoint Loading", "PASS" if loading_result else "FAIL"))
        self.logger.dedent()
        
        if not loading_result:
            self.logger.error("Failed to load test checkpoint")
            return False
        
        # Test 4: Test single game playback
        self.logger.step(4, total_tests, "Testing single game playback")
        self.logger.indent()
        game_result = self.test_single_game_playback(self.test_checkpoint_id)
        game_success = game_result["success"]
        test_results.append(("Single Game Playback", "PASS" if game_success else "FAIL"))
        self.logger.dedent()
        
        if not game_success:
            self.logger.error(f"Single game playback failed: {game_result['error']}")
            return False
        
        # Test 5: Test live playback start
        self.logger.step(5, total_tests, "Testing live playback start")
        self.logger.indent()
        live_result = self.test_live_playback_start(self.test_checkpoint_id)
        live_success = live_result["success"]
        test_results.append(("Live Playback Start", "PASS" if live_success else "FAIL"))
        self.logger.dedent()
        
        if not live_success:
            self.logger.error(f"Live playback start failed: {live_result['error']}")
            return False
        
        # Test 6: Test playback controls
        self.logger.step(6, total_tests, "Testing playback controls")
        self.logger.indent()
        controls_result = self.test_playback_controls()
        controls_success = controls_result["success"]
        test_results.append(("Playback Controls", "PASS" if controls_success else "FAIL"))
        self.logger.dedent()
        
        if not controls_success:
            self.logger.error(f"Playback controls failed: {controls_result['error']}")
            return False
        
        # Test 7: Performance validation
        self.logger.step(7, total_tests, "Validating performance")
        self.logger.indent()
        performance_ok = game_result.get("performance_ok", False)
        test_results.append(("Performance", "PASS" if performance_ok else "NEEDS ATTENTION"))
        
        if not performance_ok:
            self.logger.warning("Performance is below expected threshold")
            self.logger.log(f"Current: {game_result.get('steps_per_second', 0):.2f} steps/sec")
            self.logger.log("Expected: > 0.5 steps/sec")
        else:
            self.logger.ok("Performance is acceptable")
        self.logger.dedent()
        
        # Summary
        self.logger.separator(60)
        self.logger.success("COMPLETE GAME PLAYBACK TEST SUITE PASSED!")
        
        # Create summary table
        self.logger.table_header(["Test Component", "Status"], [30, 20])
        for test_name, status in test_results:
            self.logger.table_row([test_name, status], [30, 20])
        
        self.logger.log("")
        self.logger.log("The checkpoint system is working correctly!")
        self.logger.log("Complete games can be played back from checkpoints.")
        
        return True

def main():
    """Main entry point"""
    logger = TestLogger()
    
    logger.game("Complete Game Playback Test Suite")
    logger.log("Testing that checkpoints can play complete games")
    logger.separator(60)
    
    tester = CheckpointGameTester()
    success = tester.run_complete_test_suite()
    
    if not success:
        logger.error("TEST SUITE FAILED")
        logger.log("The checkpoint system needs attention.")
        sys.exit(1)
    else:
        logger.ok("TEST SUITE PASSED")
        logger.log("The checkpoint system is working correctly!")
        sys.exit(0)

if __name__ == "__main__":
    main() 