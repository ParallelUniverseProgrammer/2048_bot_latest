#!/usr/bin/env python3
"""
Test script to investigate checkpoint loading issues
"""

import sys
import os
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import requests
import time
import json
from typing import Dict, Any
from test_utils import BackendTester, TestLogger

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_api_endpoint(endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Test an API endpoint and return results"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=TIMEOUT)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=TIMEOUT)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        return {
            "status_code": response.status_code,
            "ok": response.ok,
            "data": response.json() if response.ok else None,
            "error": response.text if not response.ok else None,
            "response_time": response.elapsed.total_seconds()
        }
    except requests.exceptions.Timeout:
        return {"error": "Request timed out", "timeout": TIMEOUT}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection failed - server may not be running"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def test_checkpoint_endpoints():
    """Test all checkpoint-related endpoints"""
    print("FIND: Testing Checkpoint Endpoints")
    print("=" * 50)
    
    # Test 1: List checkpoints
    print("\n1. Testing GET /checkpoints")
    result = test_api_endpoint("/checkpoints")
    print(f"   Status: {result.get('status_code', 'N/A')}")
    print(f"   OK: {result.get('ok', False)}")
    print(f"   Response Time: {result.get('response_time', 'N/A')}s")
    if result.get('error'):
        print(f"   Error: {result['error']}")
    elif result.get('data'):
        print(f"   Checkpoints Found: {len(result['data'])}")
        if result['data']:
            print(f"   First Checkpoint: {result['data'][0].get('id', 'N/A')}")
    
    # Test 2: Get checkpoint stats
    print("\n2. Testing GET /checkpoints/stats")
    result = test_api_endpoint("/checkpoints/stats")
    print(f"   Status: {result.get('status_code', 'N/A')}")
    print(f"   OK: {result.get('ok', False)}")
    print(f"   Response Time: {result.get('response_time', 'N/A')}s")
    if result.get('error'):
        print(f"   Error: {result['error']}")
    elif result.get('data'):
        print(f"   Stats: {result['data']}")
    
    # Test 3: Get playback status
    print("\n3. Testing GET /checkpoints/playback/status")
    result = test_api_endpoint("/checkpoints/playback/status")
    print(f"   Status: {result.get('status_code', 'N/A')}")
    print(f"   OK: {result.get('ok', False)}")
    print(f"   Response Time: {result.get('response_time', 'N/A')}s")
    if result.get('error'):
        print(f"   Error: {result['error']}")
    elif result.get('data'):
        print(f"   Playback Status: {result['data']}")
    
    # Test 4: Get current playback data
    print("\n4. Testing GET /checkpoints/playback/current")
    result = test_api_endpoint("/checkpoints/playback/current")
    print(f"   Status: {result.get('status_code', 'N/A')}")
    print(f"   OK: {result.get('ok', False)}")
    print(f"   Response Time: {result.get('response_time', 'N/A')}s")
    if result.get('error'):
        print(f"   Error: {result['error']}")
    elif result.get('data'):
        print(f"   Has Data: {result['data'].get('has_data', False)}")
        if result['data'].get('error'):
            print(f"   Error: {result['data']['error']}")

def test_training_status():
    """Test training status endpoints"""
    print("\nFIND: Testing Training Status")
    print("=" * 50)
    
    # Test training status
    print("\n1. Testing GET /training/status")
    result = test_api_endpoint("/training/status")
    print(f"   Status: {result.get('status_code', 'N/A')}")
    print(f"   OK: {result.get('ok', False)}")
    print(f"   Response Time: {result.get('response_time', 'N/A')}s")
    if result.get('error'):
        print(f"   Error: {result['error']}")
    elif result.get('data'):
        print(f"   Training Status: {result['data']}")

def test_websocket_connection():
    """Test WebSocket connection"""
    print("\nFIND: Testing WebSocket Connection")
    print("=" * 50)
    
    try:
        import websocket
        ws = websocket.create_connection(f"{BASE_URL.replace('http', 'ws')}/ws", timeout=10)
        print("   OK: WebSocket connection successful")
        ws.close()
    except ImportError:
        print("   WARNING:  websocket-client not installed, skipping WebSocket test")
    except Exception as e:
        print(f"   ERROR: WebSocket connection failed: {str(e)}")

def main():
    """Run all tests with enhanced backend availability checking"""
    logger = TestLogger()
    backend_tester = BackendTester(BASE_URL, logger)
    
    logger.banner("Checkpoint Loading Investigation")
    
    # Test backend availability with enhanced checking
    logger.section("Testing Backend Availability")
    
    if backend_tester.is_backend_available():
        logger.ok("Backend is accessible - proceeding with tests")
        
        # Run all tests with real backend
        test_checkpoint_endpoints()
        test_training_status()
        test_websocket_connection()
        
    else:
        logger.warning("Backend is not available - attempting to start mock backend")
        
        # Try to start mock backend
        try:
            from mock_backend import MockBackend
            
            logger.info("Starting mock backend server...")
            mock_backend = MockBackend()
            mock_backend.start()
            
            # Wait a moment for startup
            time.sleep(2)
            
            # Test mock backend connectivity
            if backend_tester.is_backend_available(use_cache=False):
                logger.ok("Mock backend is running - proceeding with tests")
                
                # Run tests with mock backend
                test_checkpoint_endpoints()
                test_training_status()
                test_websocket_connection()
                
                # Clean up mock backend
                logger.info("Stopping mock backend...")
                mock_backend.stop()
                
            else:
                logger.error("Failed to start mock backend")
                return
                
        except ImportError:
            logger.error("Mock backend not available - cannot proceed with tests")
            return
        except Exception as e:
            logger.error(f"Error starting mock backend: {e}")
            return
    
    logger.separator()
    logger.success("Investigation Complete")
    
    # Print next steps
    logger.info("Next steps:")
    print("   1. Check if any endpoints are timing out")
    print("   2. Verify checkpoint files exist in backend/checkpoints/")
    print("   3. Check backend logs for errors")
    print("   4. Test frontend checkpoint loading with browser dev tools")

if __name__ == "__main__":
    main() 