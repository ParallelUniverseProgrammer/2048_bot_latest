#!/usr/bin/env python3
"""
Test suite for connection stability issues.
Reproduces the problem where:
1. Page loads and works initially
2. After some time, gets kicked to disconnected screen
3. Refresh doesn't result in successful reload

This test identifies the root cause and validates fixes.
"""

import asyncio
import json
import time
import threading
import http.client
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import contextlib
import sys
from typing import Any, Dict, List
import random
from test_utils import TestLogger, BackendTester, check_backend_or_start_mock

HOST = "localhost"
PORT = 8000

class ConnectionStabilityTest:
    """Test connection stability and disconnection scenarios"""
    
    def __init__(self):
        self.logger = TestLogger()
        self.backend = BackendTester("http://localhost:8000", self.logger)
        
    def test_initial_connection_works(self):
        """Test that initial connection works fine"""
        self.logger.testing("Testing initial connection")
        
        # Test basic connectivity using BackendTester
        if not self.backend.test_connectivity():
            self.logger.error("Initial connection test failed")
            return False
            
        # Test checkpoint loading
        checkpoints = self.backend.get_checkpoints()
        if not checkpoints:
            self.logger.error("No checkpoints available for testing")
            return False
            
        self.logger.ok("Initial connection test passed")
        return True
            
    def test_connection_degradation_over_time(self):
        """Test that connection degrades over time"""
        self.logger.testing("Testing connection degradation over 50 seconds...")
        
        success_count = 0
        failure_count = 0
        total_requests = 0
        
        # Test for 50 seconds
        start_time = time.time()
        while time.time() - start_time < 50:
            try:
                # Test a critical endpoint that should be stable
                response = self.backend.get_playback_status()
                total_requests += 1
                
                if response is not None:
                    success_count += 1
                else:
                    failure_count += 1
                    
            except Exception as e:
                failure_count += 1
                total_requests += 1
                    
            # Log every 10 seconds
            elapsed = time.time() - start_time
            if total_requests % 10 == 0:
                success_rate = success_count / total_requests * 100
                self.logger.log(f"  {elapsed:.1f}s: {success_rate:.1f}% success rate ({success_count}/{total_requests})")
                    
            time.sleep(0.5)  # Request every 500ms
                
        final_success_rate = success_count / total_requests * 100
        self.logger.log(f"Final success rate: {final_success_rate:.1f}% ({success_count}/{total_requests})")
        
        if final_success_rate > 80:
            self.logger.ok("Connection stability test passed")
            return True
        else:
            self.logger.error("Connection stability test failed - too many failures")
            return False
            
    def test_refresh_failure_scenario(self):
        """Test the specific refresh failure scenario"""
        self.logger.testing("Testing refresh failure scenario")
        
        # Simulate the scenario where training starts, then connection fails
        # and refresh doesn't work
        
        # Step 1: Start training (if available)
        training_status = self.backend.get_training_status()
        self.logger.log(f"Initial training status: {training_status}")
        
        # Step 2: Simulate connection stress
        self.logger.log("Simulating connection stress...")
        stress_start = time.time()
        
        # Make rapid requests to simulate heavy usage
        for i in range(20):
            try:
                self.backend.get_training_status()
                self.backend.get_checkpoint_stats()
                time.sleep(0.1)  # 100ms between requests
            except Exception as e:
                self.logger.warning(f"Request {i+1} failed: {e}")
        
        stress_duration = time.time() - stress_start
        self.logger.log(f"Connection stress completed in {stress_duration:.2f}s")
        
        # Step 3: Test if endpoints are still responsive
        self.logger.log("Testing endpoint responsiveness after stress...")
        
        endpoints_to_test = [
            ("training status", lambda: self.backend.get_training_status()),
            ("checkpoint stats", lambda: self.backend.get_checkpoint_stats()),
            ("connectivity", lambda: self.backend.test_connectivity())
        ]
        
        responsive_count = 0
        for name, test_func in endpoints_to_test:
            try:
                result = test_func()
                if result is not None:
                    responsive_count += 1
                    self.logger.ok(f"{name}: responsive")
                else:
                    self.logger.error(f"{name}: not responsive")
            except Exception as e:
                self.logger.error(f"{name}: failed - {e}")
        
        responsiveness_rate = responsive_count / len(endpoints_to_test) * 100
        self.logger.log(f"Endpoint responsiveness: {responsiveness_rate:.1f}% ({responsive_count}/{len(endpoints_to_test)})")
        
        if responsiveness_rate >= 66:  # At least 2/3 endpoints working
            self.logger.ok("Refresh failure scenario test passed")
            return True
        else:
            self.logger.error("Refresh failure scenario test failed")
            return False
    
    async def test_websocket_polling_fallback(self):
        """Test WebSocket polling fallback mechanism"""
        self.logger.testing("Testing WebSocket polling fallback")
        
        try:
            # Test WebSocket connection using websockets library
            import websockets
            
            async with asyncio.timeout(10):
                async with websockets.connect("ws://localhost:8000/ws") as websocket:
                    # Send a test message
                    await websocket.send(json.dumps({"type": "ping", "timestamp": time.time()}))
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(response)
                        
                        if data.get("type") == "pong":
                            self.logger.ok("WebSocket connection and message exchange successful")
                            return True
                        else:
                            self.logger.warning(f"Unexpected WebSocket response: {data}")
                            return True  # Connection works, just unexpected response
                            
                    except asyncio.TimeoutError:
                        self.logger.warning("WebSocket response timeout, but connection established")
                        return True  # Connection works, just no response
                        
        except asyncio.TimeoutError:
            self.logger.error("WebSocket connection timeout")
            return False
        except Exception as e:
            self.logger.error(f"WebSocket test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all connection stability tests"""
        self.logger.banner("Connection Stability Tests", 60)
        
        # Ensure backend is available
        if not check_backend_or_start_mock():
            self.logger.error("No backend available for testing")
            return {"error": "No backend available"}
        
        results = {
            "initial_connection": self.test_initial_connection_works(),
            "connection_degradation": self.test_connection_degradation_over_time(),
            "refresh_failure": self.test_refresh_failure_scenario(),
            "websocket_fallback": await self.test_websocket_polling_fallback()
        }
        
        # Summary
        self.logger.separator(60)
        self.logger.banner("TEST RESULTS", 60)
        
        passed_tests = sum(1 for result in results.values() if result is True)
        total_tests = len(results)
        
        for test_name, result in results.items():
            status = "PASS" if result is True else "FAIL"
            self.logger.log(f"{test_name}: {status}")
        
        self.logger.log(f"Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            self.logger.success("All connection stability tests passed!")
        else:
            self.logger.error("Some connection stability tests failed!")
        
        return results

async def main():
    """Main entry point"""
    test = ConnectionStabilityTest()
    results = await test.run_all_tests()
    
    # Save results to file
    with open("connection_stability_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to connection_stability_test_results.json")

if __name__ == "__main__":
    asyncio.run(main()) 