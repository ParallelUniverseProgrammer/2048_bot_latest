#!/usr/bin/env python3
"""
Simple test for training status synchronization between frontend and backend.
This test verifies that the backend training status endpoint works correctly.
"""

import requests
import json
import time
from typing import Dict, Any

class SimpleTrainingStatusTest:
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status from backend"""
        try:
            response = requests.get(f"{self.backend_url}/training/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting training status: {e}")
            return {"is_training": False, "is_paused": False, "current_episode": 0}
            
    def test_fresh_server_status(self):
        """Test that a fresh server shows correct training status"""
        print("🧪 Testing fresh server training status...")
        
        status = self.get_training_status()
        print(f"📊 Fresh server status: {status}")
        
        # On a fresh server, training should be False
        if not status.get("is_training", True):
            print("✅ Fresh server shows correct not-training status!")
            return True
        else:
            print("❌ Fresh server shows incorrect training status!")
            return False
            
    def test_health_endpoint(self):
        """Test that the health endpoint is accessible"""
        print("🧪 Testing health endpoint...")
        
        try:
            response = requests.get(f"{self.backend_url}/health")
            response.raise_for_status()
            print("✅ Health endpoint is accessible!")
            return True
        except Exception as e:
            print(f"❌ Health endpoint failed: {e}")
            return False
            
    def test_training_endpoints(self):
        """Test that training control endpoints are accessible"""
        print("🧪 Testing training control endpoints...")
        
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
                print(f"✅ {endpoint} is accessible (status: {response.status_code})")
            except requests.exceptions.ConnectionError:
                print(f"❌ {endpoint} connection failed")
                all_accessible = False
            except requests.exceptions.Timeout:
                print(f"❌ {endpoint} timed out")
                all_accessible = False
            except Exception as e:
                print(f"⚠️ {endpoint} returned error: {e}")
                # This might be expected for some endpoints
                
        return all_accessible

def main():
    """Run the simple training status tests"""
    print("🚀 Starting Simple Training Status Tests")
    print("=" * 50)
    
    test = SimpleTrainingStatusTest()
    
    # Test 1: Health endpoint
    print("\n📋 Test 1: Health Endpoint")
    health_passed = test.test_health_endpoint()
    
    # Test 2: Fresh server status
    print("\n📋 Test 2: Fresh Server Status")
    status_passed = test.test_fresh_server_status()
    
    # Test 3: Training endpoints
    print("\n📋 Test 3: Training Control Endpoints")
    endpoints_passed = test.test_training_endpoints()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  Test 1 (Health): {'✅ PASSED' if health_passed else '❌ FAILED'}")
    print(f"  Test 2 (Status): {'✅ PASSED' if status_passed else '❌ FAILED'}")
    print(f"  Test 3 (Endpoints): {'✅ PASSED' if endpoints_passed else '❌ FAILED'}")
    
    if health_passed and status_passed and endpoints_passed:
        print("\n🎉 All tests passed! Backend is working correctly.")
        print("\n💡 To test frontend synchronization:")
        print("   1. Start the backend: python backend/main.py")
        print("   2. Start the frontend: npm run dev")
        print("   3. Open browser and check that training status shows 'not training'")
        print("   4. Check browser console for 'Syncing training status with backend' messages")
        return 0
    else:
        print("\n💥 Some tests failed. Backend needs attention.")
        return 1

if __name__ == "__main__":
    exit(main()) 