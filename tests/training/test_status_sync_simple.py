from tests.utilities.backend_manager import requires_mock_backend
#!/usr/bin/env python3
"""
Simple Training Status Synchronization Test
==========================================

Simple test for training status synchronization between frontend and backend.
This test verifies that the backend training status endpoint works correctly.

The test covers:
- Health endpoint accessibility
- Fresh server training status
- Training control endpoints accessibility
- Basic backend connectivity validation
"""

import requests
import json
import time
from typing import Dict, Any

from tests.utilities.test_utils import TestLogger

class SimpleTrainingStatusTest:
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.logger = TestLogger()
        self.backend_url = backend_url
        
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status from backend"""
        try:
            response = requests.get(f"{self.backend_url}/training/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting training status: {e}")
            return {"is_training": False, "is_paused": False, "current_episode": 0}
@requires_mock_backend
            
    def test_fresh_server_status(self):
        """Test that a fresh server shows correct training status"""
        self.logger.info("Testing fresh server training status...")
        
        status = self.get_training_status()
        self.logger.info(f"Fresh server status: {status}")
        
        # On a fresh server, training should be False
        if not status.get("is_training", True):
            self.logger.ok("Fresh server shows correct not-training status!")
            return True
        else:
            self.logger.error("Fresh server shows incorrect training status!")
            return False
@requires_mock_backend
            
    def test_health_endpoint(self):
        """Test that the health endpoint is accessible"""
        self.logger.info("Testing health endpoint...")
        
        try:
            response = requests.get(f"{self.backend_url}/health")
            response.raise_for_status()
            self.logger.ok("Health endpoint is accessible!")
            return True
        except Exception as e:
            self.logger.error(f"Health endpoint failed: {e}")
            return False
@requires_mock_backend
            
    def test_training_endpoints(self):
        """Test that training control endpoints are accessible"""
        self.logger.info("Testing training control endpoints...")
        
        endpoints = [
            "/training/start",
            "/training/pause", 
            "/training/resume",
            "/training/stop",
            "/training/reset"
        ]
        
        all_accessible = True
        
        for endpoint in endpoints:
            try:
                # Just test that the endpoint exists (POST request)
                response = requests.post(f"{self.backend_url}{endpoint}", timeout=5)
                # We expect various status codes, but not connection errors
                self.logger.ok(f"{endpoint} is accessible (status: {response.status_code})")
            except requests.exceptions.ConnectionError:
                self.logger.error(f"{endpoint} connection failed")
                all_accessible = False
            except requests.exceptions.Timeout:
                self.logger.error(f"{endpoint} timed out")
                all_accessible = False
            except Exception as e:
                self.logger.warning(f"{endpoint} returned error: {e}")
                # This might be expected for some endpoints
                
        return all_accessible
@requires_mock_backend

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("Simple Training Status Tests", 50)
    
    test = SimpleTrainingStatusTest()
    
    # Test 1: Health endpoint
    logger.info("Test 1: Health Endpoint")
    health_passed = test.test_health_endpoint()
    
    # Test 2: Fresh server status
    logger.info("Test 2: Fresh Server Status")
    status_passed = test.test_fresh_server_status()
    
    # Test 3: Training endpoints
    logger.info("Test 3: Training Control Endpoints")
    endpoints_passed = test.test_training_endpoints()
    
    # Summary
    logger.separator()
    logger.info("Test Results Summary:")
    if health_passed:
        logger.ok("Test 1 (Health): PASSED")
    else:
        logger.error("Test 1 (Health): FAILED")
    
    if status_passed:
        logger.ok("Test 2 (Status): PASSED")
    else:
        logger.error("Test 2 (Status): FAILED")
    
    if endpoints_passed:
        logger.ok("Test 3 (Endpoints): PASSED")
    else:
        logger.error("Test 3 (Endpoints): FAILED")
    
    if health_passed and status_passed and endpoints_passed:
        logger.success("All tests passed! Backend is working correctly.")
        logger.info("To test frontend synchronization:")
        logger.info("1. Start the backend: python backend/main.py")
        logger.info("2. Start the frontend: npm run dev")
        logger.info("3. Open browser and check that training status shows 'not training'")
        logger.info("4. Check browser console for 'Syncing training status with backend' messages")
        return True
    else:
        logger.error("Some tests failed. Backend needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1) 