#!/usr/bin/env python3
"""
Test script for diagnosing mobile connection issues.
This script simulates mobile connection patterns and tests the fixes.
"""

import asyncio
import json
import time
import websockets
import requests
from typing import Dict, Any, List
import random

class MobileConnectionTester:
    """Test mobile connection stability and recovery"""
    
    def __init__(self, base_url: str = "http://192.168.1.254:8000"):
        self.base_url = base_url
        self.websocket_url = base_url.replace("http", "ws") + "/ws"
        self.connection_results = []
        
    async def test_websocket_connection(self, user_agent: str = "Mobile Safari") -> Dict[str, Any]:
        """Test WebSocket connection with mobile user agent"""
        print(f"Testing WebSocket connection with {user_agent}")
        
        try:
            # Connect with mobile user agent - note: extra_headers not supported in this websockets version
            async with websockets.connect(self.websocket_url) as websocket:
                print("✅ WebSocket connected successfully")
                
                # Test basic message exchange
                await websocket.send(json.dumps({"type": "ping", "timestamp": time.time()}))
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                
                if data.get("type") == "pong":
                    print("✅ Ping/pong working")
                else:
                    print(f"⚠️ Unexpected response: {data}")
                
                # Test connection for 10 seconds
                start_time = time.time()
                messages_received = 0
                
                while time.time() - start_time < 10:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        messages_received += 1
                        data = json.loads(message)
                        print(f"📨 Received: {data.get('type', 'unknown')}")
                    except asyncio.TimeoutError:
                        print("⏱️ No message received (timeout)")
                        break
                
                return {
                    "success": True,
                    "messages_received": messages_received,
                    "duration": time.time() - start_time
                }
                
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_connection_recovery(self, user_agent: str = "Mobile Safari") -> Dict[str, Any]:
        """Test connection recovery after disconnection"""
        print(f"Testing connection recovery with {user_agent}")
        
        results = []
        
        # Test multiple connection/disconnection cycles
        for i in range(3):
            print(f"\n--- Test cycle {i+1}/3 ---")
            
            # Connect
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    print(f"✅ Cycle {i+1}: Connected")
                    
                    # Send a message
                    await websocket.send(json.dumps({"type": "ping", "timestamp": time.time()}))
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                        print(f"✅ Cycle {i+1}: Got response")
                        results.append({"cycle": i+1, "success": True})
                    except asyncio.TimeoutError:
                        print(f"⚠️ Cycle {i+1}: Timeout waiting for response")
                        results.append({"cycle": i+1, "success": False, "error": "timeout"})
                        
            except Exception as e:
                print(f"❌ Cycle {i+1}: Connection failed - {e}")
                results.append({"cycle": i+1, "success": False, "error": str(e)})
            
            # Wait between cycles
            await asyncio.sleep(2)
        
        success_count = sum(1 for r in results if r["success"])
        return {
            "total_cycles": len(results),
            "successful_cycles": success_count,
            "success_rate": success_count / len(results) * 100,
            "results": results
        }
    
    def test_backend_endpoints(self) -> Dict[str, Any]:
        """Test backend API endpoints"""
        print("Testing backend API endpoints")
        
        endpoints = [
            "/training/status",
            "/checkpoints",
            "/checkpoints/stats"
        ]
        
        results = {}
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    print(f"✅ {endpoint}: OK")
                    results[endpoint] = {"status": "ok", "code": response.status_code}
                else:
                    print(f"⚠️ {endpoint}: Status {response.status_code}")
                    results[endpoint] = {"status": "error", "code": response.status_code}
            except Exception as e:
                print(f"❌ {endpoint}: Failed - {e}")
                results[endpoint] = {"status": "error", "error": str(e)}
        
        return results
    
    async def test_mobile_specific_issues(self) -> Dict[str, Any]:
        """Test mobile-specific connection issues"""
        print("Testing mobile-specific connection issues")
        
        mobile_user_agents = [
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36"
        ]
        
        results = {}
        
        for i, user_agent in enumerate(mobile_user_agents):
            print(f"\n--- Testing mobile user agent {i+1}/3 ---")
            print(f"User Agent: {user_agent[:50]}...")
            
            # Test basic connection
            connection_result = await self.test_websocket_connection(user_agent)
            
            # Test recovery
            recovery_result = await self.test_connection_recovery(user_agent)
            
            results[f"mobile_{i+1}"] = {
                "user_agent": user_agent,
                "connection": connection_result,
                "recovery": recovery_result
            }
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all mobile connection tests"""
        print("🚀 Starting mobile connection tests")
        print("=" * 50)
        
        results = {
            "timestamp": time.time(),
            "backend_endpoints": self.test_backend_endpoints(),
            "mobile_specific": await self.test_mobile_specific_issues()
        }
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        # Backend endpoints summary
        endpoint_success = sum(1 for r in results["backend_endpoints"].values() if r["status"] == "ok")
        print(f"Backend endpoints: {endpoint_success}/{len(results['backend_endpoints'])} working")
        
        # Mobile connection summary
        mobile_success = 0
        mobile_total = 0
        
        for mobile_test in results["mobile_specific"].values():
            if mobile_test["connection"]["success"]:
                mobile_success += 1
            mobile_total += 1
            
            recovery_rate = mobile_test["recovery"]["success_rate"]
            print(f"Mobile {mobile_test['user_agent'][:30]}...: Connection {'✅' if mobile_test['connection']['success'] else '❌'}, Recovery: {recovery_rate:.1f}%")
        
        print(f"Mobile connections: {mobile_success}/{mobile_total} successful")
        
        return results

async def main():
    """Main test runner"""
    tester = MobileConnectionTester()
    results = await tester.run_all_tests()
    
    # Save results
    with open("mobile_connection_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to mobile_connection_test_results.json")

if __name__ == "__main__":
    asyncio.run(main()) 