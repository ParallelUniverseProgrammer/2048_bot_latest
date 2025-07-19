#!/usr/bin/env python3
"""
Training Reconnection Failure Test
==================================

This test simulates training scenarios with connection failures to ensure
the system handles disconnections gracefully during training sessions.
"""

import asyncio
import time
import json
import sys
import os
import websockets
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from tests.utilities.test_utils import TestLogger, BackendTester
from tests.utilities.backend_manager import requires_mock_backend

class TrainingReconnectionFailureTest:
    """Test reconnection failure during training scenarios"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.websocket_url = base_url.replace("http", "ws") + "/ws"
        self.logger = TestLogger()
        self.backend = BackendTester(base_url, self.logger)
        
    async def test_reconnection_during_training_simulation(self) -> Dict[str, Any]:
        """Test reconnection attempts during simulated training activity"""
        self.logger.testing("Testing reconnection during training simulation")
        
        # Step 1: Establish initial connection
        self.logger.log("Step 1: Establishing initial connection...")
        try:
            async with websockets.connect(self.websocket_url) as initial_websocket:
                self.logger.ok("Initial WebSocket connected")
                
                # Send initial message
                await initial_websocket.send(json.dumps({
                    "type": "ping",
                    "timestamp": time.time(),
                    "client_type": "mobile_safari"
                }))
                
                # Wait for connection confirmation
                try:
                    response = await asyncio.wait_for(initial_websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    self.logger.log(f"Initial connection response: {data.get('type', 'unknown')}")
                except asyncio.TimeoutError:
                    self.logger.warning("No initial response received")
                
                # Step 2: Simulate some activity (like training updates)
                self.logger.log("Step 2: Simulating training activity...")
                start_time = time.time()
                messages_received = 0
                
                # Monitor for 10 seconds to simulate active training
                while time.time() - start_time < 10:
                    try:
                        message = await asyncio.wait_for(initial_websocket.recv(), timeout=1.0)
                        messages_received += 1
                        data = json.loads(message)
                        self.logger.debug(f"Training message: {data.get('type')}")
                    except asyncio.TimeoutError:
                        # Send keepalive
                        await initial_websocket.send(json.dumps({
                            "type": "ping",
                            "timestamp": time.time()
                        }))
                
                self.logger.log(f"Initial connection active for {time.time() - start_time:.1f}s, received {messages_received} messages")
                
        except Exception as e:
            self.logger.error(f"Initial connection failed: {e}")
            return {"success": False, "error": f"Initial connection failed: {e}"}
        
        # Step 3: Close connection (simulating disconnection)
        self.logger.log("Step 3: Closing connection (simulating disconnection)...")
        
        # Step 4: Attempt to reconnect multiple times
        self.logger.log("Step 4: Attempting reconnections...")
        reconnection_attempts = 5
        successful_reconnections = 0
        reconnection_results = []
        
        for attempt in range(reconnection_attempts):
            self.logger.log(f"Reconnection attempt {attempt + 1}/{reconnection_attempts}")
            
            try:
                # Wait a bit before reconnection attempt
                await asyncio.sleep(2)
                
                # Try to reconnect
                async with websockets.connect(self.websocket_url) as new_websocket:
                    self.logger.ok(f"Reconnection attempt {attempt + 1}: Connected")
                    
                    # Send reconnection message
                    await new_websocket.send(json.dumps({
                        "type": "ping",
                        "timestamp": time.time(),
                        "client_type": "mobile_safari",
                        "reconnection_attempt": attempt + 1
                    }))
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(new_websocket.recv(), timeout=5.0)
                        data = json.loads(response)
                        self.logger.log(f"Reconnection {attempt + 1} response: {data.get('type', 'unknown')}")
                        
                        # Test if we can receive messages for a few seconds
                        test_start = time.time()
                        test_messages = 0
                        connection_stable = True
                        
                        while time.time() - test_start < 5:  # Test for 5 seconds
                            try:
                                message = await asyncio.wait_for(new_websocket.recv(), timeout=1.0)
                                test_messages += 1
                                data = json.loads(message)
                                self.logger.debug(f"Reconnection {attempt + 1} message: {data.get('type')}")
                            except asyncio.TimeoutError:
                                # Send keepalive
                                await new_websocket.send(json.dumps({
                                    "type": "ping",
                                    "timestamp": time.time()
                                }))
                            except websockets.exceptions.ConnectionClosed:
                                connection_stable = False
                                self.logger.error(f"Reconnection {attempt + 1}: Connection closed during test")
                                break
                        
                        if connection_stable:
                            successful_reconnections += 1
                            self.logger.ok(f"Reconnection {attempt + 1}: Stable for 5s, {test_messages} messages")
                        else:
                            self.logger.error(f"Reconnection {attempt + 1}: Unstable")
                        
                        reconnection_results.append({
                            "attempt": attempt + 1,
                            "connected": True,
                            "stable": connection_stable,
                            "test_messages": test_messages,
                            "test_duration": time.time() - test_start
                        })
                        
                    except asyncio.TimeoutError:
                        self.logger.error(f"Reconnection {attempt + 1}: No response received")
                        reconnection_results.append({
                            "attempt": attempt + 1,
                            "connected": True,
                            "stable": False,
                            "error": "No response"
                        })
                        
            except Exception as e:
                self.logger.error(f"Reconnection {attempt + 1}: Failed to connect - {e}")
                reconnection_results.append({
                    "attempt": attempt + 1,
                    "connected": False,
                    "error": str(e)
                })
        
        reconnection_success_rate = successful_reconnections / reconnection_attempts * 100
        
        return {
            "success": reconnection_success_rate >= 80,  # At least 4/5 reconnections should work
            "reconnection_attempts": reconnection_attempts,
            "successful_reconnections": successful_reconnections,
            "reconnection_success_rate": reconnection_success_rate,
            "reconnection_results": reconnection_results
        }
    
@requires_mock_backend
    def test_backend_endpoints_after_disconnection(self) -> Dict[str, Any]:
        """Test if backend endpoints become unresponsive after disconnection"""
        self.logger.testing("Testing backend endpoints after disconnection")
        
        # Step 1: Check initial endpoint health
        initial_health = self._check_endpoint_health()
        self.logger.log(f"Initial endpoint health: {initial_health['responsive_endpoints']}/{initial_health['total_endpoints']}")
        
        # Step 2: Simulate disconnection by making rapid requests then stopping
        self.logger.log("Simulating disconnection scenario...")
        
        # Make some requests to simulate activity
        for i in range(10):
            try:
                self.backend.get_training_status()
                time.sleep(0.1)
            except Exception as e:
                self.logger.warning(f"Pre-disconnection request {i+1} failed: {e}")
        
        # Step 3: Wait a bit (simulating the disconnection period)
        self.logger.log("Waiting to simulate disconnection period...")
        time.sleep(5)
        
        # Step 4: Test endpoints after "disconnection"
        post_disconnection_health = self._check_endpoint_health()
        self.logger.log(f"Post-disconnection endpoint health: {post_disconnection_health['responsive_endpoints']}/{post_disconnection_health['total_endpoints']}")
        
        # Step 5: Test multiple reconnection attempts to endpoints
        self.logger.log("Testing multiple endpoint reconnection attempts...")
        endpoint_reconnection_successes = 0
        endpoint_reconnection_attempts = 5
        
        for i in range(endpoint_reconnection_attempts):
            try:
                # Test all critical endpoints
                health = self._check_endpoint_health()
                if health['responsive_endpoints'] == health['total_endpoints']:
                    endpoint_reconnection_successes += 1
                    self.logger.ok(f"Endpoint reconnection attempt {i+1}: Successful")
                else:
                    self.logger.error(f"Endpoint reconnection attempt {i+1}: Failed")
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Endpoint reconnection attempt {i+1}: Exception - {e}")
        
        endpoint_success_rate = endpoint_reconnection_successes / endpoint_reconnection_attempts * 100
        
        return {
            "success": endpoint_success_rate >= 80,
            "initial_health": initial_health,
            "post_disconnection_health": post_disconnection_health,
            "endpoint_reconnection_attempts": endpoint_reconnection_attempts,
            "endpoint_reconnection_successes": endpoint_reconnection_successes,
            "endpoint_success_rate": endpoint_success_rate
        }
    
    async def test_concurrent_disconnection_scenario(self) -> Dict[str, Any]:
        """Test multiple clients disconnecting and reconnecting simultaneously"""
        self.logger.testing("Testing concurrent disconnection scenario")
        
        async def client_lifecycle(client_id: int) -> Dict[str, Any]:
            """Simulate a single client's lifecycle: connect -> disconnect -> reconnect"""
            try:
                # Phase 1: Initial connection
                async with websockets.connect(self.websocket_url) as websocket:
                    self.logger.log(f"Client {client_id}: Initial connection established")
                    
                    # Send initial message
                    await websocket.send(json.dumps({
                        "type": "ping",
                        "timestamp": time.time(),
                        "client_id": client_id
                    }))
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(response)
                        self.logger.debug(f"Client {client_id}: Initial response: {data.get('type')}")
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Client {client_id}: No initial response")
                
                # Phase 2: Wait (simulating disconnection)
                await asyncio.sleep(3)
                
                # Phase 3: Attempt reconnection
                try:
                    async with websockets.connect(self.websocket_url) as new_websocket:
                        self.logger.log(f"Client {client_id}: Reconnection successful")
                        
                        # Send reconnection message
                        await new_websocket.send(json.dumps({
                            "type": "ping",
                            "timestamp": time.time(),
                            "client_id": client_id,
                            "reconnected": True
                        }))
                        
                        # Test stability for a few seconds
                        start_time = time.time()
                        messages_received = 0
                        stable = True
                        
                        while time.time() - start_time < 5:
                            try:
                                message = await asyncio.wait_for(new_websocket.recv(), timeout=1.0)
                                messages_received += 1
                            except asyncio.TimeoutError:
                                # Send keepalive
                                await new_websocket.send(json.dumps({
                                    "type": "ping",
                                    "timestamp": time.time()
                                }))
                            except websockets.exceptions.ConnectionClosed:
                                stable = False
                                break
                        
                        return {
                            "client_id": client_id,
                            "initial_connection": True,
                            "reconnection": True,
                            "stable": stable,
                            "messages_received": messages_received,
                            "duration": time.time() - start_time
                        }
                        
                except Exception as e:
                    self.logger.error(f"Client {client_id}: Reconnection failed - {e}")
                    return {
                        "client_id": client_id,
                        "initial_connection": True,
                        "reconnection": False,
                        "error": str(e)
                    }
                    
            except Exception as e:
                self.logger.error(f"Client {client_id}: Initial connection failed - {e}")
                return {
                    "client_id": client_id,
                    "initial_connection": False,
                    "error": str(e)
                }
        
        # Start multiple clients simultaneously
        client_count = 3
        tasks = [asyncio.create_task(client_lifecycle(i + 1)) for i in range(client_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_lifecycles = sum(1 for r in results if isinstance(r, dict) and r.get("reconnection") is True)
        total_clients = len(results)
        
        return {
            "success": successful_lifecycles == total_clients,
            "total_clients": total_clients,
            "successful_lifecycles": successful_lifecycles,
            "success_rate": successful_lifecycles / total_clients * 100,
            "results": results
        }
    
    def _check_endpoint_health(self) -> Dict[str, Any]:
        """Check health of all critical endpoints"""
        endpoints = [
            ("training_status", lambda: self.backend.get_training_status()),
            ("checkpoint_stats", lambda: self.backend.get_checkpoint_stats()),
            ("connectivity", lambda: self.backend.test_connectivity()),
            ("playback_status", lambda: self.backend.get_playback_status())
        ]
        
        responsive_count = 0
        for name, test_func in endpoints:
            try:
                result = test_func()
                if result is not None:
                    responsive_count += 1
            except Exception:
                pass
        
        return {
            "total_endpoints": len(endpoints),
            "responsive_endpoints": responsive_count,
            "health_rate": responsive_count / len(endpoints) * 100
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all training reconnection failure tests"""
        self.logger.banner("Training Reconnection Failure Tests", 60)
        
        # Run tests in sequence
        results = {
            "websocket_reconnection": await self.test_reconnection_during_training_simulation(),
            "endpoint_reconnection": self.test_backend_endpoints_after_disconnection(),
            "concurrent_disconnection": await self.test_concurrent_disconnection_scenario()
        }
        
        # Summary
        self.logger.separator(60)
        self.logger.banner("TEST RESULTS", 60)
        
        passed_tests = sum(1 for result in results.values() if isinstance(result, dict) and result.get("success") is True)
        total_tests = len(results)
        
        for test_name, result in results.items():
            if isinstance(result, dict):
                status = "PASS" if result.get("success") is True else "FAIL"
                self.logger.log(f"{test_name}: {status}")
                
                # Log specific details for failed tests
                if not result.get("success"):
                    if test_name == "websocket_reconnection":
                        self.logger.log(f"  Reconnection success rate: {result.get('reconnection_success_rate', 0):.1f}%")
                        self.logger.log(f"  Successful reconnections: {result.get('successful_reconnections', 0)}/{result.get('reconnection_attempts', 0)}")
                    elif test_name == "endpoint_reconnection":
                        self.logger.log(f"  Endpoint success rate: {result.get('endpoint_success_rate', 0):.1f}%")
                    elif test_name == "concurrent_disconnection":
                        self.logger.log(f"  Client success rate: {result.get('success_rate', 0):.1f}%")
            else:
                self.logger.log(f"{test_name}: ERROR")
        
        self.logger.log(f"Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            self.logger.success("All training reconnection failure tests passed!")
            self.logger.log("Note: This means reconnection works correctly in the current environment.")
            self.logger.log("The issue may be specific to mobile Safari or require actual training to be running.")
        else:
            self.logger.error("Some training reconnection failure tests failed!")
            self.logger.log("This indicates the root cause of the reconnection issue.")
        
        return results

@requires_mock_backend("Training Reconnection Failure Tests")
async def main():
    """Main entry point"""
    test = TrainingReconnectionFailureTest()
    results = await test.run_all_tests()
    
    # Save results to file
    with open("training_reconnection_failure_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    test.logger.info(f"Results saved to results file")

if __name__ == "__main__":
    asyncio.run(main()) 