#!/usr/bin/env python3
"""
Test training status synchronization between frontend and backend.
This test verifies that the frontend doesn't show stale training state from localStorage.
"""

import asyncio
import json
import time
from typing import Dict, Any
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

class TrainingStatusSyncTest:
    def __init__(self, backend_url: str = "http://localhost:8000", frontend_url: str = "http://localhost:5173"):
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.driver = None
        
    def setup_driver(self):
        """Setup Chrome driver with headless mode"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(10)
        
    def teardown_driver(self):
        """Clean up driver"""
        if self.driver:
            self.driver.quit()
            
    def get_backend_training_status(self) -> Dict[str, Any]:
        """Get current training status from backend"""
        try:
            response = requests.get(f"{self.backend_url}/training/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting backend training status: {e}")
            return {"is_training": False, "is_paused": False, "current_episode": 0}
            
    def get_frontend_training_status(self) -> Dict[str, Any]:
        """Get current training status from frontend via JavaScript"""
        try:
            # Execute JavaScript to get training store state
            js_code = """
            return {
                isTraining: window.trainingStore?.getState()?.isTraining || false,
                isPaused: window.trainingStore?.getState()?.isPaused || false,
                currentEpisode: window.trainingStore?.getState()?.currentEpisode || 0,
                isConnected: window.trainingStore?.getState()?.isConnected || false
            };
            """
            result = self.driver.execute_script(js_code)
            return result
        except Exception as e:
            print(f"Error getting frontend training status: {e}")
            return {"isTraining": False, "isPaused": False, "currentEpisode": 0, "isConnected": False}
            
    def wait_for_connection(self, timeout: int = 30) -> bool:
        """Wait for frontend to connect to backend"""
        try:
            # Wait for connection status to show "Connected"
            WebDriverWait(self.driver, timeout).until(
                EC.text_to_be_present_in_element((By.CLASS_NAME, "text-green-600"), "Connected")
            )
            return True
        except Exception as e:
            print(f"Timeout waiting for connection: {e}")
            return False
            
    def test_training_status_sync(self):
        """Test that frontend training status syncs with backend"""
        print("🧪 Testing training status synchronization...")
        
        try:
            # Setup driver
            self.setup_driver()
            
            # Get initial backend status
            backend_status = self.get_backend_training_status()
            print(f"📊 Backend training status: {backend_status}")
            
            # Navigate to frontend
            print(f"🌐 Navigating to frontend: {self.frontend_url}")
            self.driver.get(self.frontend_url)
            
            # Wait for page to load and connection to establish
            print("⏳ Waiting for connection to establish...")
            if not self.wait_for_connection():
                print("❌ Failed to establish connection")
                return False
                
            # Wait a bit more for any initial sync to complete
            time.sleep(2)
            
            # Get frontend status
            frontend_status = self.get_frontend_training_status()
            print(f"📱 Frontend training status: {frontend_status}")
            
            # Verify synchronization
            backend_training = backend_status.get("is_training", False)
            frontend_training = frontend_status.get("isTraining", False)
            backend_paused = backend_status.get("is_paused", False)
            frontend_paused = frontend_status.get("isPaused", False)
            backend_episode = backend_status.get("current_episode", 0)
            frontend_episode = frontend_status.get("currentEpisode", 0)
            
            print(f"🔍 Comparing status:")
            print(f"  Training: Backend={backend_training}, Frontend={frontend_training}")
            print(f"  Paused: Backend={backend_paused}, Frontend={frontend_paused}")
            print(f"  Episode: Backend={backend_episode}, Frontend={frontend_episode}")
            
            # Check if statuses match
            training_match = backend_training == frontend_training
            paused_match = backend_paused == frontend_paused
            episode_match = backend_episode == frontend_episode
            connected = frontend_status.get("isConnected", False)
            
            if training_match and paused_match and episode_match and connected:
                print("✅ Training status synchronization successful!")
                return True
            else:
                print("❌ Training status synchronization failed!")
                print(f"  Training match: {training_match}")
                print(f"  Paused match: {paused_match}")
                print(f"  Episode match: {episode_match}")
                print(f"  Connected: {connected}")
                return False
                
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            return False
        finally:
            self.teardown_driver()
            
    def test_fresh_server_sync(self):
        """Test that frontend shows correct status when connecting to a fresh server"""
        print("🧪 Testing fresh server synchronization...")
        
        try:
            # Setup driver
            self.setup_driver()
            
            # Get backend status (should be not training on fresh server)
            backend_status = self.get_backend_training_status()
            print(f"📊 Fresh server backend status: {backend_status}")
            
            # Navigate to frontend
            print(f"🌐 Navigating to frontend: {self.frontend_url}")
            self.driver.get(self.frontend_url)
            
            # Wait for connection
            print("⏳ Waiting for connection to establish...")
            if not self.wait_for_connection():
                print("❌ Failed to establish connection")
                return False
                
            # Wait for sync to complete
            time.sleep(3)
            
            # Get frontend status
            frontend_status = self.get_frontend_training_status()
            print(f"📱 Frontend status after sync: {frontend_status}")
            
            # On a fresh server, training should be False
            if not frontend_status.get("isTraining", True) and frontend_status.get("isConnected", False):
                print("✅ Fresh server shows correct not-training status!")
                return True
            else:
                print("❌ Fresh server shows incorrect training status!")
                return False
                
        except Exception as e:
            print(f"❌ Fresh server test failed: {e}")
            return False
        finally:
            self.teardown_driver()

def main():
    """Run the training status sync tests"""
    print("🚀 Starting Training Status Sync Tests")
    print("=" * 50)
    
    test = TrainingStatusSyncTest()
    
    # Test 1: Basic synchronization
    print("\n📋 Test 1: Basic Training Status Synchronization")
    test1_passed = test.test_training_status_sync()
    
    # Test 2: Fresh server synchronization
    print("\n📋 Test 2: Fresh Server Synchronization")
    test2_passed = test.test_fresh_server_sync()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  Test 1 (Basic Sync): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Test 2 (Fresh Server): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Training status synchronization is working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. Training status synchronization needs attention.")
        return 1

if __name__ == "__main__":
    exit(main()) 