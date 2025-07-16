#!/usr/bin/env python3
"""
Test Mock Backend Server
========================

This test verifies that the mock backend server works correctly and can be used
for offline testing when the real backend is not available.
"""

import sys
import os
import time
import threading
import requests
import json
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tests.mock_backend import MockBackend
from tests.test_utils import TestLogger

class MockBackendTester:
    """Test suite for mock backend server"""
    
    def __init__(self, port: int = 8001):  # Use different port to avoid conflicts
        self.logger = TestLogger()
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.mock_server = None
        self.test_results = []
    
    def start_mock_server(self):
        """Start the mock backend server"""
        try:
            self.mock_server = MockBackend("localhost", self.port)
            self.mock_server.start()
            
            # Wait for server to be ready
            time.sleep(1)
            
            # Verify server is running
            if self.mock_server.is_alive():
                self.logger.ok(f"Mock server started on port {self.port}")
                return True
            else:
                self.logger.error("Mock server failed to start")
                return False
        except Exception as e:
            self.logger.error(f"Failed to start mock server: {e}")
            return False
    
    def stop_mock_server(self):
        """Stop the mock backend server"""
        if self.mock_server:
            self.mock_server.stop()
            self.logger.ok("Mock server stopped")
    
    def test_basic_connectivity(self) -> bool:
        """Test basic connectivity to mock server"""
        self.logger.info("Testing basic connectivity...")
        
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    self.logger.ok("Basic connectivity test passed")
                    return True
                else:
                    self.logger.error(f"Unexpected response: {data}")
                    return False
            else:
                self.logger.error(f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Connectivity test failed: {e}")
            return False
    
    def test_checkpoints_endpoint(self) -> bool:
        """Test checkpoints endpoint"""
        self.logger.info("Testing checkpoints endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/checkpoints", timeout=5)
            if response.status_code == 200:
                checkpoints = response.json()
                if isinstance(checkpoints, list) and len(checkpoints) > 0:
                    self.logger.ok(f"Found {len(checkpoints)} mock checkpoints")
                    
                    # Verify checkpoint structure
                    first_checkpoint = checkpoints[0]
                    required_fields = ["id", "name", "timestamp", "episode_count", "score"]
                    missing_fields = [field for field in required_fields if field not in first_checkpoint]
                    
                    if not missing_fields:
                        self.logger.ok("Checkpoint structure is valid")
                        return True
                    else:
                        self.logger.error(f"Missing checkpoint fields: {missing_fields}")
                        return False
                else:
                    self.logger.error("No checkpoints returned")
                    return False
            else:
                self.logger.error(f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Checkpoints test failed: {e}")
            return False
    
    def test_checkpoint_stats(self) -> bool:
        """Test checkpoint stats endpoint"""
        self.logger.info("Testing checkpoint stats endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/checkpoints/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                required_fields = ["total_checkpoints", "total_size", "best_score", "latest_episode"]
                missing_fields = [field for field in required_fields if field not in stats]
                
                if not missing_fields:
                    self.logger.ok("Checkpoint stats structure is valid")
                    self.logger.info(f"  Total checkpoints: {stats['total_checkpoints']}")
                    self.logger.info(f"  Best score: {stats['best_score']}")
                    self.logger.info(f"  Latest episode: {stats['latest_episode']}")
                    return True
                else:
                    self.logger.error(f"Missing stats fields: {missing_fields}")
                    return False
            else:
                self.logger.error(f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Stats test failed: {e}")
            return False
    
    def test_individual_checkpoint(self) -> bool:
        """Test loading individual checkpoint"""
        self.logger.info("Testing individual checkpoint loading...")
        
        try:
            # First get list of checkpoints
            response = requests.get(f"{self.base_url}/checkpoints", timeout=5)
            if response.status_code != 200:
                self.logger.error("Failed to get checkpoints list")
                return False
            
            checkpoints = response.json()
            if not checkpoints:
                self.logger.error("No checkpoints available")
                return False
            
            # Test loading first checkpoint
            checkpoint_id = checkpoints[0]["id"]
            response = requests.get(f"{self.base_url}/checkpoints/{checkpoint_id}", timeout=5)
            
            if response.status_code == 200:
                checkpoint = response.json()
                if checkpoint["id"] == checkpoint_id:
                    self.logger.ok(f"Successfully loaded checkpoint {checkpoint_id}")
                    return True
                else:
                    self.logger.error("Checkpoint ID mismatch")
                    return False
            else:
                self.logger.error(f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Individual checkpoint test failed: {e}")
            return False
    
    def test_training_status(self) -> bool:
        """Test training status endpoint"""
        self.logger.info("Testing training status endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/training/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                required_fields = ["is_training", "current_episode", "total_episodes"]
                missing_fields = [field for field in required_fields if field not in status]
                
                if not missing_fields:
                    self.logger.ok("Training status structure is valid")
                    self.logger.info(f"  Is training: {status['is_training']}")
                    self.logger.info(f"  Episode: {status['current_episode']}/{status['total_episodes']}")
                    return True
                else:
                    self.logger.error(f"Missing training fields: {missing_fields}")
                    return False
            else:
                self.logger.error(f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Training status test failed: {e}")
            return False
    
    def test_playback_endpoints(self) -> bool:
        """Test playback endpoints"""
        self.logger.info("Testing playback endpoints...")
        
        try:
            # Test playback status
            response = requests.get(f"{self.base_url}/checkpoints/playback/status", timeout=5)
            if response.status_code != 200:
                self.logger.error("Failed to get playback status")
                return False
            
            status = response.json()
            self.logger.ok(f"Playback status: is_playing={status.get('is_playing')}")
            
            # Test game simulation
            # First get a checkpoint ID
            response = requests.get(f"{self.base_url}/checkpoints", timeout=5)
            if response.status_code != 200:
                self.logger.error("Failed to get checkpoints for playback test")
                return False
            
            checkpoints = response.json()
            if not checkpoints:
                self.logger.error("No checkpoints available for playback test")
                return False
            
            checkpoint_id = checkpoints[0]["id"]
            
            # Test game simulation
            response = requests.post(f"{self.base_url}/checkpoints/{checkpoint_id}/playback/game", timeout=10)
            if response.status_code == 200:
                game_result = response.json()
                required_fields = ["game_history", "final_score", "max_tile", "steps"]
                missing_fields = [field for field in required_fields if field not in game_result]
                
                if not missing_fields:
                    self.logger.ok("Game simulation test passed")
                    self.logger.info(f"  Score: {game_result['final_score']}")
                    self.logger.info(f"  Max tile: {game_result['max_tile']}")
                    self.logger.info(f"  Steps: {game_result['steps']}")
                    return True
                else:
                    self.logger.error(f"Missing game result fields: {missing_fields}")
                    return False
            else:
                self.logger.error(f"Game simulation failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Playback test failed: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling for invalid requests"""
        self.logger.info("Testing error handling...")
        
        try:
            # Test 404 for non-existent checkpoint
            response = requests.get(f"{self.base_url}/checkpoints/non_existent", timeout=5)
            if response.status_code == 404:
                self.logger.ok("404 error handling works correctly")
            else:
                self.logger.error(f"Expected 404, got {response.status_code}")
                return False
            
            # Test 404 for non-existent endpoint
            response = requests.get(f"{self.base_url}/invalid/endpoint", timeout=5)
            if response.status_code == 404:
                self.logger.ok("Invalid endpoint error handling works correctly")
                return True
            else:
                self.logger.error(f"Expected 404 for invalid endpoint, got {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Error handling test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all mock backend tests"""
        self.logger.info("🚀 Starting Mock Backend Test Suite")
        self.logger.info("=" * 50)
        
        # Start mock server
        if not self.start_mock_server():
            self.logger.error("Failed to start mock server")
            return False
        
        try:
            # Run all tests
            tests = [
                ("basic_connectivity", self.test_basic_connectivity),
                ("checkpoints_endpoint", self.test_checkpoints_endpoint),
                ("checkpoint_stats", self.test_checkpoint_stats),
                ("individual_checkpoint", self.test_individual_checkpoint),
                ("training_status", self.test_training_status),
                ("playback_endpoints", self.test_playback_endpoints),
                ("error_handling", self.test_error_handling)
            ]
            
            results = {}
            for test_name, test_func in tests:
                self.logger.info(f"\n🧪 Running {test_name}...")
                try:
                    result = test_func()
                    results[test_name] = result
                    if result:
                        self.logger.ok(f"✅ {test_name} passed")
                    else:
                        self.logger.error(f"❌ {test_name} failed")
                except Exception as e:
                    self.logger.error(f"❌ {test_name} failed with exception: {e}")
                    results[test_name] = False
            
            # Generate report
            self.generate_report(results)
            
            return all(results.values())
            
        finally:
            # Stop mock server
            self.stop_mock_server()
    
    def generate_report(self, results: Dict[str, bool]):
        """Generate test report"""
        self.logger.info("\n📊 Mock Backend Test Results")
        self.logger.info("=" * 50)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.logger.info(f"{status} {test_name}")
        
        self.logger.info("-" * 50)
        self.logger.info(f"📈 Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            self.logger.info("🎉 All mock backend tests passed!")
        else:
            self.logger.warning(f"⚠️  {total - passed} tests failed")


def main():
    """Main test execution function"""
    try:
        tester = MockBackendTester()
        success = tester.run_all_tests()
        
        if success:
            print("\n✅ Mock backend is working correctly!")
            print("You can now use the mock backend for offline testing.")
        else:
            print("\n❌ Mock backend has issues that need to be addressed.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 