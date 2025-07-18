#!/usr/bin/env python3
"""
Enhanced Frontend Testing Automation
===================================

This script provides automated and semi-automated testing for the frontend
checkpoint system, with detailed instructions for manual verification steps.
"""

import requests
import time
import json
from typing import Dict, Any, List
# Add project root to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from tests.utilities.test_utils import TestLogger, BackendTester

class FrontendTester:
    """Enhanced frontend testing with automation and detailed instructions"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.logger = TestLogger()
        self.backend = BackendTester(base_url, self.logger)
    
    def test_api_endpoints_for_frontend(self) -> Dict[str, bool]:
        """Test all API endpoints that the frontend depends on"""
        self.logger.banner("Testing Frontend API Dependencies", 60)
        
        results = {}
        
        # Test basic endpoints
        basic_results = self.backend.test_basic_endpoints()
        results.update(basic_results)
        
        # Test specific frontend endpoints
        frontend_endpoints = [
            ("/training/start", "POST"),
            ("/training/stop", "POST"),
            ("/training/status", "GET"),
            ("/checkpoints/playback/start", "POST"),
            ("/checkpoints/playback/stop", "POST"),
            ("/checkpoints/playback/pause", "POST"),
            ("/checkpoints/playback/resume", "POST"),
            ("/checkpoints/playback/status", "GET"),
        ]
        
        for endpoint, method in frontend_endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                else:
                    response = requests.post(f"{self.base_url}{endpoint}", timeout=10)
                
                # Accept both 200 and 400 (bad request) as valid responses
                # since we're not providing proper parameters
                success = response.status_code in [200, 400]
                results[f"{method} {endpoint}"] = success
                
                if success:
                    self.logger.ok(f"{method} {endpoint} - endpoint accessible")
                else:
                    self.logger.error(f"{method} {endpoint} - HTTP {response.status_code}")
                    
            except Exception as e:
                results[f"{method} {endpoint}"] = False
                self.logger.error(f"{method} {endpoint} - {e}")
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        
        if passed == total:
            self.logger.ok(f"All {total} frontend API endpoints accessible")
        else:
            self.logger.warning(f"Only {passed}/{total} frontend API endpoints accessible")
        
        return results
    
    def test_websocket_connectivity(self) -> bool:
        """Test WebSocket connectivity for real-time updates"""
        self.logger.banner("Testing WebSocket Connectivity", 60)
        
        try:
            # Test if WebSocket endpoint is accessible
            # This is a basic check - full WebSocket testing would require more setup
            response = requests.get(f"{self.base_url}/ws", timeout=5)
            
            # WebSocket endpoints typically return 400 or 426 for HTTP requests
            if response.status_code in [400, 426]:
                self.logger.ok("WebSocket endpoint is accessible")
                return True
            else:
                self.logger.warning(f"WebSocket endpoint returned unexpected status: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            self.logger.error("WebSocket endpoint not accessible")
            return False
        except Exception as e:
            self.logger.error(f"WebSocket test failed: {e}")
            return False
    
    def generate_manual_test_checklist(self) -> None:
        """Generate a comprehensive manual testing checklist"""
        self.logger.banner("Manual Frontend Testing Checklist", 60)
        
        checklist = [
            {
                "category": "Initial Page Load",
                "tests": [
                    "Navigate to http://localhost:3000 (or frontend URL)",
                    "Verify page loads without errors",
                    "Check that all tabs are visible (Training, Checkpoints, etc.)",
                    "Confirm no console errors in browser developer tools",
                    "Verify responsive design on different screen sizes"
                ]
            },
            {
                "category": "Checkpoints Tab - Basic Functionality",
                "tests": [
                    "Click on Checkpoints tab",
                    "Verify checkpoints load and display immediately",
                    "Check that checkpoint cards show episode numbers",
                    "Verify file sizes are displayed correctly",
                    "Confirm creation dates are shown",
                    "Test checkpoint sorting/filtering if available"
                ]
            },
            {
                "category": "Checkpoints Tab - During Training",
                "tests": [
                    "Start training from Training tab",
                    "Navigate back to Checkpoints tab",
                    "Verify checkpoints still display (not stuck in loading)",
                    "Confirm new checkpoints appear during training",
                    "Check that UI remains responsive during training",
                    "Verify no infinite loading spinners"
                ]
            },
            {
                "category": "Checkpoint Playback",
                "tests": [
                    "Click 'Play' button on a checkpoint",
                    "Verify game board appears and starts playing",
                    "Check that game moves are smooth and visible",
                    "Confirm score updates in real-time",
                    "Test pause/resume functionality",
                    "Verify stop button works correctly",
                    "Check that playback controls are responsive"
                ]
            },
            {
                "category": "Training Tab Integration",
                "tests": [
                    "Switch between Training and Checkpoints tabs",
                    "Verify training status updates don't affect checkpoint display",
                    "Check that stopping training doesn't break checkpoint loading",
                    "Confirm training metrics display correctly",
                    "Test starting/stopping training multiple times"
                ]
            },
            {
                "category": "Error Handling",
                "tests": [
                    "Disconnect from backend (stop server)",
                    "Verify appropriate error messages appear",
                    "Check that UI handles connection loss gracefully",
                    "Restart backend and verify recovery",
                    "Test behavior with invalid checkpoint data"
                ]
            },
            {
                "category": "Performance & UX",
                "tests": [
                    "Verify page loads within 3 seconds",
                    "Check that checkpoint loading is under 5 seconds",
                    "Confirm smooth animations and transitions",
                    "Test with multiple checkpoints (10+ if available)",
                    "Verify memory usage doesn't grow excessively",
                    "Check for any memory leaks during extended use"
                ]
            },
            {
                "category": "Cross-Browser Compatibility",
                "tests": [
                    "Test in Chrome/Chromium",
                    "Test in Firefox",
                    "Test in Safari (if on macOS)",
                    "Test in Edge",
                    "Verify consistent behavior across browsers",
                    "Check for browser-specific issues"
                ]
            }
        ]
        
        for category_info in checklist:
            self.logger.log(f"\n{category_info['category']}:")
            self.logger.log("-" * len(category_info['category']))
            
            for i, test in enumerate(category_info['tests'], 1):
                self.logger.log(f"{i:2d}. {test}")
        
        self.logger.log("\n" + "=" * 60)
        self.logger.log("TESTING INSTRUCTIONS:")
        self.logger.log("=" * 60)
        self.logger.log("1. Start the backend server: cd backend && python main.py")
        self.logger.log("2. Start the frontend dev server: cd frontend && npm run dev")
        self.logger.log("3. Work through each category systematically")
        self.logger.log("4. Document any issues found with:")
        self.logger.log("   - Browser and version")
        self.logger.log("   - Steps to reproduce")
        self.logger.log("   - Expected vs actual behavior")
        self.logger.log("   - Console errors (if any)")
        self.logger.log("5. Test both normal and edge cases")
        self.logger.log("6. Verify fixes by retesting affected areas")
    
    def test_frontend_data_consistency(self) -> bool:
        """Test that frontend receives consistent data from backend"""
        self.logger.banner("Testing Frontend Data Consistency", 60)
        
        try:
            # Get checkpoint data that frontend would receive
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.logger.warning("No checkpoints available for consistency testing")
                return False
            
            # Test data structure consistency
            required_fields = ['id', 'episode', 'created_at', 'file_size']
            missing_fields = []
            
            for checkpoint in checkpoints:
                for field in required_fields:
                    if field not in checkpoint:
                        missing_fields.append(f"{checkpoint.get('id', 'unknown')}.{field}")
            
            if missing_fields:
                self.logger.error(f"Missing required fields: {missing_fields}")
                return False
            
            # Test stats consistency
            stats = self.backend.get_checkpoint_stats()
            if stats:
                expected_count = len(checkpoints)
                actual_count = stats.get('total_checkpoints', 0)
                
                if expected_count != actual_count:
                    self.logger.warning(f"Stats count mismatch: expected {expected_count}, got {actual_count}")
                else:
                    self.logger.ok("Checkpoint count consistency verified")
            
            self.logger.ok("Frontend data consistency checks passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data consistency test failed: {e}")
            return False
    
    def run_automated_frontend_tests(self) -> Dict[str, bool]:
        """Run all automated frontend tests"""
        self.logger.starting("Automated Frontend Testing Suite")
        self.logger.separator(60)
        
        results = {}
        
        # Test 1: API endpoints
        self.logger.log("\n1. Testing API endpoints...")
        api_results = self.test_api_endpoints_for_frontend()
        results['api_endpoints'] = all(api_results.values())
        
        # Test 2: WebSocket connectivity
        self.logger.log("\n2. Testing WebSocket connectivity...")
        results['websocket'] = self.test_websocket_connectivity()
        
        # Test 3: Data consistency
        self.logger.log("\n3. Testing data consistency...")
        results['data_consistency'] = self.test_frontend_data_consistency()
        
        # Summary
        self.logger.separator(60)
        passed = sum(results.values())
        total = len(results)
        
        if passed == total:
            self.logger.ok(f"All {total} automated frontend tests passed")
        else:
            self.logger.warning(f"Only {passed}/{total} automated frontend tests passed")
        
        return results
    
    def run_complete_frontend_test_suite(self) -> None:
        """Run the complete frontend test suite"""
        self.logger.banner("Complete Frontend Test Suite", 60)
        
        # Run automated tests
        automated_results = self.run_automated_frontend_tests()
        
        # Generate manual test checklist
        self.generate_manual_test_checklist()
        
        # Final summary
        self.logger.separator(60)
        self.logger.ok("Frontend test suite complete")
        self.logger.separator(60)
        
        self.logger.log("AUTOMATED TESTS:")
        for test_name, passed in automated_results.items():
            status = "PASS" if passed else "FAIL"
            self.logger.log(f"  {test_name}: {status}")
        
        self.logger.log("\nMANUAL TESTS:")
        self.logger.log("  See detailed checklist above")
        self.logger.log("  Complete manual testing before marking frontend as verified")
        
        self.logger.log("\nNEXT STEPS:")
        self.logger.log("1. Review automated test results")
        self.logger.log("2. Complete manual testing checklist")
        self.logger.log("3. Fix any issues found")
        self.logger.log("4. Retest affected areas")
        self.logger.log("5. Document test results")

def main():
    """Main entry point"""
    logger = TestLogger()
    
    logger.banner("Enhanced Frontend Testing", 60)
    
    # Check backend connectivity first
    logger.testing("Checking backend connectivity...")
    backend = BackendTester()
    if not backend.test_connectivity():
        logger.error("Backend server is not running!")
        logger.log("Please start the backend server first:")
        logger.log("   cd backend")
        logger.log("   python main.py")
        return
    
    # Run complete test suite
    tester = FrontendTester()
    tester.run_complete_frontend_test_suite()

if __name__ == "__main__":
    main() 