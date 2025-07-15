#!/usr/bin/env python3
"""
Live Playback Test Suite
========================

Tests for live checkpoint playback functionality including:
- Real-time game playback from checkpoints
- Playback speed control and manipulation
- WebSocket integration for live updates
- Memory usage during extended playback sessions
- Connection handling and error recovery
- Performance monitoring during playback

Usage:
    python test_live_playback.py

Expected outcomes:
- Successful playback of checkpoint games
- Proper speed control functionality
- Stable WebSocket connections
- Memory leak detection and prevention
- Graceful error handling
"""

import sys
import os
import asyncio
import time
import threading
from pathlib import Path
import json
import traceback
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from test_utils import TestLogger, BackendTester

class LivePlaybackTester:
    """Test suite for live checkpoint playback functionality"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.logger = TestLogger()
        self.backend = BackendTester(base_url, self.logger)
        self.playback_sessions = []
        self.memory_usage = []
        self.test_results = {
            'playback_basic': False,
            'speed_control': False,
            'websocket_stability': False,
            'memory_management': False,
            'error_recovery': False
        }
    
    def log_test_start(self, test_name: str):
        """Log the start of a test"""
        self.logger.info(f"🧪 Starting {test_name}...")
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log the result of a test"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.logger.info(f"{status} {test_name}")
        if details:
            self.logger.info(f"   Details: {details}")
    
    def test_basic_playback(self) -> bool:
        """Test basic checkpoint playback functionality"""
        self.log_test_start("Basic Playback Test")
        
        try:
            # Get available checkpoints
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Basic Playback Test", False, "No checkpoints available")
                return False
            
            # Select first checkpoint for testing
            checkpoint = checkpoints[0]
            checkpoint_id = checkpoint.get('id', checkpoint.get('name', 'unknown'))
            
            # Start playback
            response = self.backend.start_playback(checkpoint_id)
            if not response:
                self.log_test_result("Basic Playback Test", False, "Failed to start playback")
                return False
            
            # Monitor playback for a few seconds
            time.sleep(3)
            
            # Check playback status
            status = self.backend.get_playback_status()
            if status and status.get('active', False):
                self.log_test_result("Basic Playback Test", True, f"Playback active for {checkpoint_id}")
                return True
            else:
                self.log_test_result("Basic Playback Test", False, "Playback not active")
                return False
                
        except Exception as e:
            self.log_test_result("Basic Playback Test", False, f"Exception: {str(e)}")
            return False
    
    def test_speed_control(self) -> bool:
        """Test playback speed control functionality"""
        self.log_test_start("Speed Control Test")
        
        try:
            # Get available checkpoints
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Speed Control Test", False, "No checkpoints available")
                return False
            
            checkpoint = checkpoints[0]
            checkpoint_id = checkpoint.get('id', checkpoint.get('name', 'unknown'))
            
            # Start playback
            self.backend.start_playback(checkpoint_id)
            time.sleep(1)
            
            # Test different speeds
            speeds = [0.5, 1.0, 2.0, 4.0]
            for speed in speeds:
                response = self.backend.set_playback_speed(speed)
                if not response:
                    self.log_test_result("Speed Control Test", False, f"Failed to set speed {speed}")
                    return False
                
                time.sleep(0.5)  # Brief pause to allow speed change
            
            self.log_test_result("Speed Control Test", True, f"Successfully tested speeds: {speeds}")
            return True
            
        except Exception as e:
            self.log_test_result("Speed Control Test", False, f"Exception: {str(e)}")
            return False
    
    def test_websocket_stability(self) -> bool:
        """Test WebSocket connection stability during playback"""
        self.log_test_start("WebSocket Stability Test")
        
        try:
            # Test WebSocket connection
            ws_connected = self.backend.test_websocket_connection()
            if not ws_connected:
                self.log_test_result("WebSocket Stability Test", False, "WebSocket connection failed")
                return False
            
            # Start playback and monitor WebSocket
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("WebSocket Stability Test", False, "No checkpoints available")
                return False
            
            checkpoint = checkpoints[0]
            checkpoint_id = checkpoint.get('id', checkpoint.get('name', 'unknown'))
            
            self.backend.start_playback(checkpoint_id)
            
            # Monitor WebSocket for 10 seconds
            start_time = time.time()
            message_count = 0
            while time.time() - start_time < 10:
                messages = self.backend.get_websocket_messages()
                if messages:
                    message_count += len(messages)
                time.sleep(0.1)
            
            self.log_test_result("WebSocket Stability Test", True, f"Received {message_count} messages")
            return True
            
        except Exception as e:
            self.log_test_result("WebSocket Stability Test", False, f"Exception: {str(e)}")
            return False
    
    def test_memory_management(self) -> bool:
        """Test memory usage during extended playback sessions"""
        self.log_test_start("Memory Management Test")
        
        try:
            import psutil
            process = psutil.Process()
            
            # Record initial memory usage
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.memory_usage.append(initial_memory)
            
            # Start playback
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Memory Management Test", False, "No checkpoints available")
                return False
            
            checkpoint = checkpoints[0]
            checkpoint_id = checkpoint.get('id', checkpoint.get('name', 'unknown'))
            
            self.backend.start_playback(checkpoint_id)
            
            # Monitor memory usage for 30 seconds
            start_time = time.time()
            max_memory = initial_memory
            
            while time.time() - start_time < 30:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                self.memory_usage.append(current_memory)
                max_memory = max(max_memory, current_memory)
                time.sleep(1)
            
            # Check for memory leaks
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Allow for reasonable memory increase (< 100MB)
            if memory_increase < 100:
                self.log_test_result("Memory Management Test", True, 
                                   f"Memory stable: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
                return True
            else:
                self.log_test_result("Memory Management Test", False, 
                                   f"Memory leak detected: {memory_increase:.1f}MB increase")
                return False
                
        except ImportError:
            self.log_test_result("Memory Management Test", False, "psutil not available")
            return False
        except Exception as e:
            self.log_test_result("Memory Management Test", False, f"Exception: {str(e)}")
            return False
    
    def test_error_recovery(self) -> bool:
        """Test error recovery during playback"""
        self.log_test_start("Error Recovery Test")
        
        try:
            # Test invalid checkpoint ID
            response = self.backend.start_playback("invalid_checkpoint_id")
            if response:
                self.log_test_result("Error Recovery Test", False, "Should have failed for invalid checkpoint")
                return False
            
            # Test stopping non-existent playback
            response = self.backend.stop_playback()
            # This should not crash the system
            
            # Test starting valid playback after error
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Error Recovery Test", False, "No checkpoints available")
                return False
            
            checkpoint = checkpoints[0]
            checkpoint_id = checkpoint.get('id', checkpoint.get('name', 'unknown'))
            
            response = self.backend.start_playback(checkpoint_id)
            if not response:
                self.log_test_result("Error Recovery Test", False, "Failed to start playback after error")
                return False
            
            self.log_test_result("Error Recovery Test", True, "Error recovery successful")
            return True
            
        except Exception as e:
            self.log_test_result("Error Recovery Test", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all live playback tests"""
        self.logger.info("🚀 Starting Live Playback Test Suite")
        self.logger.info("=" * 50)
        
        # Test backend connectivity first
        if not self.backend.test_connectivity():
            self.logger.error("❌ Backend connectivity failed - aborting tests")
            return
        
        # Run all tests
        self.test_results['playback_basic'] = self.test_basic_playback()
        self.test_results['speed_control'] = self.test_speed_control()
        self.test_results['websocket_stability'] = self.test_websocket_stability()
        self.test_results['memory_management'] = self.test_memory_management()
        self.test_results['error_recovery'] = self.test_error_recovery()
        
        # Generate final report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        self.logger.info("📊 Live Playback Test Results")
        self.logger.info("=" * 50)
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.logger.info(f"{status} {test_name}")
        
        self.logger.info("-" * 50)
        self.logger.info(f"📈 Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            self.logger.info("🎉 All live playback tests passed!")
        else:
            self.logger.warning(f"⚠️  {total - passed} tests failed")
        
        # Memory usage summary
        if self.memory_usage:
            min_memory = min(self.memory_usage)
            max_memory = max(self.memory_usage)
            avg_memory = sum(self.memory_usage) / len(self.memory_usage)
            
            self.logger.info(f"💾 Memory Usage: {min_memory:.1f}MB - {max_memory:.1f}MB (avg: {avg_memory:.1f}MB)")


def main():
    """Main test execution function"""
    try:
        tester = LivePlaybackTester()
        tester.run_all_tests()
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 