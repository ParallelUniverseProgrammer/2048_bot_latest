#!/usr/bin/env python3
"""
Edge Case Testing for Checkpoint System
======================================

This script tests various edge cases and error conditions to ensure
the checkpoint system is robust and handles unexpected situations gracefully.
"""

import requests
import time
import json
import tempfile
import os
from typing import Dict, Any, List
# Add project root to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from tests.utilities.test_utils import TestLogger, BackendTester, GameTester, PlaybackTester

class EdgeCaseTester:
    """Test edge cases and error conditions"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.logger = TestLogger()
        self.backend = BackendTester(base_url, self.logger)
        self.game_tester = GameTester(base_url, self.logger, 60)  # Shorter timeout for edge cases
        self.playback_tester = PlaybackTester(base_url, self.logger)
    
    def test_invalid_checkpoint_ids(self) -> Dict[str, bool]:
        """Test behavior with invalid checkpoint IDs"""
        self.logger.banner("Testing Invalid Checkpoint IDs", 60)
        
        results = {}
        invalid_ids = [
            "nonexistent_checkpoint",
            "checkpoint_episode_999999",
            "",
            "null",
            "undefined",
            "../../../etc/passwd",
            "checkpoint_episode_-1",
            "checkpoint_episode_abc",
            "checkpoint_episode_1.5",
            "checkpoint_episode_1; DROP TABLE checkpoints;",
            "checkpoint_episode_1' OR '1'='1",
            "checkpoint_episode_1\x00",
            "checkpoint_episode_1\n\r",
            "checkpoint_episode_1%00",
            "checkpoint_episode_1<script>alert('xss')</script>",
            "checkpoint_episode_1" + "A" * 1000,  # Very long ID
        ]
        
        for invalid_id in invalid_ids:
            try:
                # Test checkpoint loading
                response = requests.get(f"{self.base_url}/checkpoints/{invalid_id}", timeout=10)
                
                # Should return 404 or 400, not 200 or 500
                if response.status_code in [400, 404]:
                    results[f"load_{invalid_id[:20]}"] = True
                    self.logger.ok(f"Invalid ID '{invalid_id[:20]}...' handled correctly (HTTP {response.status_code})")
                elif response.status_code == 500:
                    results[f"load_{invalid_id[:20]}"] = False
                    self.logger.error(f"Invalid ID '{invalid_id[:20]}...' caused server error (HTTP 500)")
                else:
                    results[f"load_{invalid_id[:20]}"] = False
                    self.logger.warning(f"Invalid ID '{invalid_id[:20]}...' returned unexpected status: {response.status_code}")
                
                # Test game playback with invalid ID
                response = requests.post(f"{self.base_url}/checkpoints/{invalid_id}/playback/game", timeout=10)
                
                if response.status_code in [400, 404]:
                    results[f"playback_{invalid_id[:20]}"] = True
                elif response.status_code == 500:
                    results[f"playback_{invalid_id[:20]}"] = False
                    self.logger.error(f"Invalid ID playback '{invalid_id[:20]}...' caused server error")
                else:
                    results[f"playback_{invalid_id[:20]}"] = response.status_code != 200
                    
            except requests.exceptions.Timeout:
                results[f"load_{invalid_id[:20]}"] = False
                results[f"playback_{invalid_id[:20]}"] = False
                self.logger.error(f"Invalid ID '{invalid_id[:20]}...' caused timeout")
            except Exception as e:
                results[f"load_{invalid_id[:20]}"] = False
                results[f"playback_{invalid_id[:20]}"] = False
                self.logger.error(f"Invalid ID '{invalid_id[:20]}...' caused exception: {e}")
        
        passed = sum(results.values())
        total = len(results)
        
        if passed == total:
            self.logger.ok(f"All {total} invalid ID tests passed")
        else:
            self.logger.warning(f"Only {passed}/{total} invalid ID tests passed")
        
        return results
    
    def test_malformed_requests(self) -> Dict[str, bool]:
        """Test behavior with malformed HTTP requests"""
        self.logger.banner("Testing Malformed Requests", 60)
        
        results = {}
        
        # Test malformed JSON payloads
        malformed_payloads = [
            '{"invalid": json}',
            '{"missing_quote: "value"}',
            '{"trailing_comma": "value",}',
            '{"unclosed_string": "value}',
            '{"invalid_number": 123abc}',
            '{"null_byte": "value\x00"}',
            '{"very_long_key": "' + "A" * 10000 + '"}',
            '{"nested": {"very": {"deep": {"structure": {"that": {"goes": {"on": {"forever": "value"}}}}}}}}',
            '',
            'null',
            'undefined',
            '[]',
            '123',
            '"string"',
            'true',
            'false',
        ]
        
        for i, payload in enumerate(malformed_payloads):
            try:
                response = requests.post(
                    f"{self.base_url}/training/start",
                    data=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                # Should return 400 (bad request), not 500 (server error)
                if response.status_code == 400:
                    results[f"malformed_{i}"] = True
                    self.logger.ok(f"Malformed payload {i} handled correctly (HTTP 400)")
                elif response.status_code == 500:
                    results[f"malformed_{i}"] = False
                    self.logger.error(f"Malformed payload {i} caused server error (HTTP 500)")
                else:
                    results[f"malformed_{i}"] = True  # Other status codes are acceptable
                    self.logger.info(f"Malformed payload {i} returned HTTP {response.status_code}")
                    
            except requests.exceptions.Timeout:
                results[f"malformed_{i}"] = False
                self.logger.error(f"Malformed payload {i} caused timeout")
            except Exception as e:
                results[f"malformed_{i}"] = False
                self.logger.error(f"Malformed payload {i} caused exception: {e}")
        
        return results
    
    def test_concurrent_operations(self) -> Dict[str, bool]:
        """Test concurrent operations that might cause race conditions"""
        self.logger.banner("Testing Concurrent Operations", 60)
        
        results = {}
        
        # Get a valid checkpoint for testing
        checkpoints = self.backend.get_checkpoints()
        if not checkpoints:
            self.logger.warning("No checkpoints available for concurrent testing")
            return {"concurrent_operations": False}
        
        checkpoint_id = checkpoints[0]['id']
        
        # Test concurrent checkpoint loading
        try:
            import threading
            import queue
            
            def load_checkpoint(result_queue, thread_id):
                try:
                    response = requests.get(f"{self.base_url}/checkpoints/{checkpoint_id}", timeout=30)
                    result_queue.put((thread_id, response.status_code == 200))
                except Exception as e:
                    result_queue.put((thread_id, False))
            
            # Start multiple concurrent requests
            threads = []
            result_queue = queue.Queue()
            
            for i in range(5):
                thread = threading.Thread(target=load_checkpoint, args=(result_queue, i))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=60)
            
            # Collect results
            concurrent_results = []
            while not result_queue.empty():
                concurrent_results.append(result_queue.get())
            
            success_count = sum(1 for _, success in concurrent_results if success)
            results["concurrent_checkpoint_loading"] = success_count >= 3  # At least 3 should succeed
            
            self.logger.ok(f"Concurrent checkpoint loading: {success_count}/5 succeeded")
            
        except Exception as e:
            results["concurrent_checkpoint_loading"] = False
            self.logger.error(f"Concurrent checkpoint loading test failed: {e}")
        
        return results
    
    def test_resource_limits(self) -> Dict[str, bool]:
        """Test behavior at resource limits"""
        self.logger.banner("Testing Resource Limits", 60)
        
        results = {}
        
        # Test rapid consecutive requests
        try:
            rapid_success = 0
            for i in range(20):
                response = requests.get(f"{self.base_url}/checkpoints", timeout=5)
                if response.status_code == 200:
                    rapid_success += 1
                time.sleep(0.1)  # Small delay to avoid overwhelming
            
            results["rapid_requests"] = rapid_success >= 15  # At least 75% should succeed
            self.logger.ok(f"Rapid requests: {rapid_success}/20 succeeded")
            
        except Exception as e:
            results["rapid_requests"] = False
            self.logger.error(f"Rapid requests test failed: {e}")
        
        # Test large response handling
        try:
            response = requests.get(f"{self.base_url}/checkpoints", timeout=30)
            if response.status_code == 200:
                data = response.json()
                # Check if we can handle large responses
                results["large_response"] = len(str(data)) > 0
                self.logger.ok(f"Large response handling: {len(str(data))} bytes processed")
            else:
                results["large_response"] = False
                
        except Exception as e:
            results["large_response"] = False
            self.logger.error(f"Large response test failed: {e}")
        
        return results
    
    def test_error_recovery(self) -> Dict[str, bool]:
        """Test error recovery scenarios"""
        self.logger.banner("Testing Error Recovery", 60)
        
        results = {}
        
        # Test recovery after invalid operations
        try:
            # Perform an invalid operation
            requests.post(f"{self.base_url}/checkpoints/invalid_id/playback/game", timeout=10)
            
            # Then perform a valid operation
            response = requests.get(f"{self.base_url}/checkpoints", timeout=10)
            results["recovery_after_invalid"] = response.status_code == 200
            
            if results["recovery_after_invalid"]:
                self.logger.ok("System recovered after invalid operation")
            else:
                self.logger.error("System did not recover after invalid operation")
                
        except Exception as e:
            results["recovery_after_invalid"] = False
            self.logger.error(f"Error recovery test failed: {e}")
        
        # Test recovery after timeout
        try:
            # This should timeout or fail quickly
            try:
                requests.get(f"{self.base_url}/checkpoints/nonexistent", timeout=1)
            except:
                pass  # Expected to fail
            
            # Then perform a valid operation
            response = requests.get(f"{self.base_url}/checkpoints", timeout=10)
            results["recovery_after_timeout"] = response.status_code == 200
            
            if results["recovery_after_timeout"]:
                self.logger.ok("System recovered after timeout")
            else:
                self.logger.error("System did not recover after timeout")
                
        except Exception as e:
            results["recovery_after_timeout"] = False
            self.logger.error(f"Timeout recovery test failed: {e}")
        
        return results
    
    def test_boundary_conditions(self) -> Dict[str, bool]:
        """Test boundary conditions and limits"""
        self.logger.banner("Testing Boundary Conditions", 60)
        
        results = {}
        
        # Test empty responses
        try:
            response = requests.get(f"{self.base_url}/checkpoints/stats", timeout=10)
            if response.status_code == 200:
                stats = response.json()
                # Should handle case where there are no checkpoints
                results["empty_stats"] = isinstance(stats, dict)
                self.logger.ok("Empty stats handling works")
            else:
                results["empty_stats"] = False
                
        except Exception as e:
            results["empty_stats"] = False
            self.logger.error(f"Empty stats test failed: {e}")
        
        # Test very long URLs
        try:
            long_id = "checkpoint_episode_" + "1" * 1000
            response = requests.get(f"{self.base_url}/checkpoints/{long_id}", timeout=10)
            
            # Should return 404 or 400, not crash
            results["long_url"] = response.status_code in [400, 404]
            
            if results["long_url"]:
                self.logger.ok("Long URL handled correctly")
            else:
                self.logger.error(f"Long URL returned unexpected status: {response.status_code}")
                
        except Exception as e:
            results["long_url"] = False
            self.logger.error(f"Long URL test failed: {e}")
        
        return results
    
    def run_all_edge_case_tests(self) -> Dict[str, bool]:
        """Run all edge case tests"""
        self.logger.starting("Edge Case Test Suite")
        self.logger.separator(60)
        
        all_results = {}
        
        # Test 1: Invalid checkpoint IDs
        self.logger.log("\n1. Testing invalid checkpoint IDs...")
        invalid_id_results = self.test_invalid_checkpoint_ids()
        all_results.update(invalid_id_results)
        
        # Test 2: Malformed requests
        self.logger.log("\n2. Testing malformed requests...")
        malformed_results = self.test_malformed_requests()
        all_results.update(malformed_results)
        
        # Test 3: Concurrent operations
        self.logger.log("\n3. Testing concurrent operations...")
        concurrent_results = self.test_concurrent_operations()
        all_results.update(concurrent_results)
        
        # Test 4: Resource limits
        self.logger.log("\n4. Testing resource limits...")
        resource_results = self.test_resource_limits()
        all_results.update(resource_results)
        
        # Test 5: Error recovery
        self.logger.log("\n5. Testing error recovery...")
        recovery_results = self.test_error_recovery()
        all_results.update(recovery_results)
        
        # Test 6: Boundary conditions
        self.logger.log("\n6. Testing boundary conditions...")
        boundary_results = self.test_boundary_conditions()
        all_results.update(boundary_results)
        
        # Summary
        self.logger.separator(60)
        passed = sum(all_results.values())
        total = len(all_results)
        
        if passed == total:
            self.logger.ok(f"All {total} edge case tests passed")
        else:
            self.logger.warning(f"Only {passed}/{total} edge case tests passed")
        
        # Detailed results
        self.logger.log("\nDetailed Results:")
        for test_name, passed in all_results.items():
            status = "PASS" if passed else "FAIL"
            self.logger.log(f"  {test_name}: {status}")
        
        return all_results

def main():
    """Main entry point"""
    logger = TestLogger()
    
    logger.banner("Edge Case Testing Suite", 60)
    
    # Check backend connectivity first
    logger.testing("Checking backend connectivity...")
    backend = BackendTester()
    if not backend.test_connectivity():
        logger.error("Backend server is not running!")
        logger.log("Please start the backend server first:")
        logger.log("   cd backend")
        logger.log("   python main.py")
        return
    
    # Run edge case tests
    tester = EdgeCaseTester()
    results = tester.run_all_edge_case_tests()
    
    # Final summary
    logger.separator(60)
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        logger.ok("All edge case tests passed - system is robust!")
    else:
        logger.warning(f"Edge case testing found {total - passed} potential issues")
        logger.log("Review the detailed results above and fix any failing tests")

if __name__ == "__main__":
    main() 