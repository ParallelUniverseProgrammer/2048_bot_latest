#!/usr/bin/env python3
"""
Simple mobile connectivity test script
"""

import requests
import time
import json
from urllib.parse import urljoin

def test_mobile_connectivity(base_ip="172.23.160.1", frontend_port=5173, backend_port=8000):
    """Test mobile connectivity to both frontend and backend"""
    
    frontend_url = f"http://{base_ip}:{frontend_port}"
    backend_url = f"http://{base_ip}:{backend_port}"
    
    print(f"🔍 Testing Mobile Connectivity")
    print(f"Frontend: {frontend_url}")
    print(f"Backend: {backend_url}")
    print("="*50)
    
    # Test 1: Frontend HTML
    try:
        print("📱 Testing frontend access...")
        response = requests.get(frontend_url, timeout=10)
        if response.status_code == 200:
            print(f"✅ Frontend accessible (Status: {response.status_code})")
            print(f"   Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
            print(f"   Content-Length: {len(response.content)} bytes")
        else:
            print(f"❌ Frontend returned status: {response.status_code}")
    except Exception as e:
        print(f"❌ Frontend error: {e}")
    
    # Test 2: Backend API
    try:
        print("\n🔌 Testing backend API...")
        response = requests.get(f"{backend_url}/mobile-test", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend API accessible (Status: {response.status_code})")
            print(f"   Response: {data.get('message', 'Unknown')}")
        else:
            print(f"❌ Backend API returned status: {response.status_code}")
    except Exception as e:
        print(f"❌ Backend API error: {e}")
    
    # Test 3: WebSocket URL (just the HTTP equivalent)
    try:
        print("\n🌐 Testing WebSocket URL (HTTP)...")
        response = requests.get(f"{backend_url}/ws", timeout=5)
        print(f"WebSocket endpoint status: {response.status_code}")
    except Exception as e:
        print(f"WebSocket endpoint error: {e}")
    
    # Test 4: CORS Headers
    try:
        print("\n🔒 Testing CORS headers...")
        response = requests.options(frontend_url, timeout=10)
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin', 'Not set'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods', 'Not set'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers', 'Not set'),
        }
        print(f"CORS Headers: {json.dumps(cors_headers, indent=2)}")
    except Exception as e:
        print(f"CORS test error: {e}")
    
    # Test 5: Multiple requests (mobile timeout test)
    print("\n⏱️ Testing connection stability (5 requests)...")
    for i in range(1, 6):
        try:
            start_time = time.time()
            response = requests.get(frontend_url, timeout=10)
            end_time = time.time()
            print(f"   Request {i}: {response.status_code} in {end_time - start_time:.2f}s")
        except Exception as e:
            print(f"   Request {i}: Failed - {e}")
        time.sleep(1)

def main():
    """Main function"""
    try:
        test_mobile_connectivity()
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 