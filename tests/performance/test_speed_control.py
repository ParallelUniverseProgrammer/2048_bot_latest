#!/usr/bin/env python3
"""
Test script for playback speed control functionality
Uses the existing test utilities for backend management and testing
"""

import sys
import os
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from tests.utilities.test_utils import BackendTester, TestLogger, check_backend_or_start_mock, requires_backend

class SpeedControlTester:
    """Test suite for playback speed control functionality"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.logger = TestLogger()
        self.backend = BackendTester(base_url, self.logger)
        self.test_results = {
            'speed_get': False,
            'speed_set': False,
            'invalid_speeds': False,
            'status_integration': False
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
    
    @requires_backend("Speed Control Tests")
    def test_get_current_speed(self) -> bool:
        """Test getting current playback speed"""
        self.log_test_start("Get Current Speed Test")
        
        try:
            response = self.backend.get_playback_speed()
            if response and 'speed' in response:
                self.log_test_result("Get Current Speed Test", True, f"Current speed: {response['speed']}x")
                return True
            else:
                self.log_test_result("Get Current Speed Test", False, "Failed to get speed or invalid response")
                return False
        except Exception as e:
            self.log_test_result("Get Current Speed Test", False, f"Exception: {str(e)}")
            return False
    
    @requires_backend("Speed Control Tests")
    def test_set_speed_values(self) -> bool:
        """Test setting different speed values"""
        self.log_test_start("Set Speed Values Test")
        
        test_speeds = [1.0, 2.0, 2.5, 3.0, 4.0]
        success_count = 0
        
        for speed in test_speeds:
            try:
                response = self.backend.set_playback_speed(speed)
                if response and response.get('speed') == speed:
                    success_count += 1
                    self.logger.ok(f"   Speed {speed}x set successfully")
                else:
                    self.logger.error(f"   Failed to set speed {speed}x")
            except Exception as e:
                self.logger.error(f"   Error setting speed {speed}x: {e}")
        
        success = success_count == len(test_speeds)
        self.log_test_result("Set Speed Values Test", success, f"{success_count}/{len(test_speeds)} speeds set successfully")
        return success
    
    @requires_backend("Speed Control Tests")
    def test_invalid_speed_values(self) -> bool:
        """Test that invalid speed values are properly rejected"""
        self.log_test_start("Invalid Speed Values Test")
        
        invalid_speeds = [-1, 0, 15, "invalid"]
        rejected_count = 0
        
        for speed in invalid_speeds:
            try:
                response = self.backend.set_playback_speed(speed)
                # Should return None or error for invalid speeds
                if response is None:
                    rejected_count += 1
                    self.logger.ok(f"   Correctly rejected invalid speed: {speed}")
                else:
                    self.logger.error(f"   Should have rejected invalid speed: {speed}")
            except Exception as e:
                # Exception is also acceptable for invalid speeds
                rejected_count += 1
                self.logger.ok(f"   Correctly rejected invalid speed {speed} with exception")
        
        success = rejected_count == len(invalid_speeds)
        self.log_test_result("Invalid Speed Values Test", success, f"{rejected_count}/{len(invalid_speeds)} invalid speeds rejected")
        return success
    
    @requires_backend("Speed Control Tests")
    def test_speed_in_status(self) -> bool:
        """Test that speed is included in playback status"""
        self.log_test_start("Speed in Status Test")
        
        try:
            status = self.backend.get_playback_status()
            if status and 'playback_speed' in status:
                self.log_test_result("Speed in Status Test", True, f"Speed in status: {status['playback_speed']}x")
                return True
            else:
                self.log_test_result("Speed in Status Test", False, "Speed field missing from playback status")
                return False
        except Exception as e:
            self.log_test_result("Speed in Status Test", False, f"Exception: {str(e)}")
            return False
    
    @requires_backend("Speed Control Tests")
    def test_speed_with_playback(self) -> bool:
        """Test speed control during actual playback"""
        self.log_test_start("Speed with Playback Test")
        
        try:
            # Get available checkpoints
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Speed with Playback Test", False, "No checkpoints available for playback test")
                return False
            
            checkpoint = checkpoints[0]
            checkpoint_id = checkpoint.get('id', checkpoint.get('name', 'unknown'))
            
            # Start playback
            start_result = self.backend.start_playback(checkpoint_id)
            if not start_result:
                self.log_test_result("Speed with Playback Test", False, "Failed to start playback")
                return False
            
            # Wait a moment for playback to initialize
            import time
            time.sleep(2)
            
            # Test speed changes during playback
            test_speeds = [1.0, 2.5, 4.0]
            success_count = 0
            
            for speed in test_speeds:
                try:
                    response = self.backend.set_playback_speed(speed)
                    if response and response.get('speed') == speed:
                        success_count += 1
                        self.logger.ok(f"   Speed {speed}x set during playback")
                    else:
                        self.logger.error(f"   Failed to set speed {speed}x during playback")
                except Exception as e:
                    self.logger.error(f"   Error setting speed {speed}x during playback: {e}")
                
                time.sleep(1)  # Brief pause between speed changes
            
            # Stop playback
            try:
                import requests
                requests.post(f"{self.backend.base_url}/checkpoints/playback/stop", timeout=5)
            except:
                pass  # Ignore stop errors
            
            success = success_count == len(test_speeds)
            self.log_test_result("Speed with Playback Test", success, f"{success_count}/{len(test_speeds)} speed changes successful during playback")
            return success
            
        except Exception as e:
            self.log_test_result("Speed with Playback Test", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all speed control tests"""
        self.logger.info("Starting Speed Control Test Suite")
        self.logger.info("=" * 50)
        
        # Test backend connectivity first
        if not self.backend.test_connectivity():
            self.logger.error("Backend connectivity failed - aborting tests")
            return
        
        # Run all tests
        self.test_results['speed_get'] = self.test_get_current_speed()
        self.test_results['speed_set'] = self.test_set_speed_values()
        self.test_results['invalid_speeds'] = self.test_invalid_speed_values()
        self.test_results['status_integration'] = self.test_speed_in_status()
        self.test_results['playback_integration'] = self.test_speed_with_playback()
        
        # Generate final report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        self.logger.info("Speed Control Test Results")
        self.logger.info("=" * 50)
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            self.logger.info(f"{status} {test_name}")
        
        self.logger.info("")
        if passed == total:
            self.logger.success(f"All {total} speed control tests passed!")
        else:
            self.logger.error(f"{total - passed}/{total} speed control tests failed")
        
        self.logger.info("")
        self.logger.info("Speed Control Features Tested:")
        self.logger.info("  • Get current playback speed")
        self.logger.info("  • Set different speed values (1x, 2x, 2.5x, 3x, 4x)")
        self.logger.info("  • Reject invalid speed values")
        self.logger.info("  • Speed integration with playback status")
        self.logger.info("  • Speed changes during active playback")

def main():
    """Main test runner"""
    # Check if backend is available or start mock backend
    if not check_backend_or_start_mock():
        logger.error("Failed to start backend for testing")
        return 1
    
    # Run the test suite
    tester = SpeedControlTester()
    tester.run_all_tests()
    
    # Return exit code based on test results
    passed = sum(1 for result in tester.test_results.values() if result)
    total = len(tester.test_results)
    
    if passed == total:
        logger.success("All speed control tests completed successfully!")
        return 0
    else:
        logger.warning(f"{total - passed}/{total} speed control tests failed")
        return 1

if __name__ == "__main__":
    exit(main()) 