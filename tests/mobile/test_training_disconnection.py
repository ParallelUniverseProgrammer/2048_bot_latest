#!/usr/bin/env python3
"""
Focused test for mobile Safari disconnection during training.
This test reproduces the specific issue where:
1. Training starts and works initially
2. After some time, mobile Safari gets kicked to disconnected fallback screen
3. Refresh doesn't fix the problem
4. Other clients fail to connect

This test combines elements from mobile connection, connection stability, and training status sync tests.
"""

import asyncio
import json
import time
import websockets
from typing import Dict, Any, List
from tests.utilities.test_utils import TestLogger, BackendTester
from tests.utilities.backend_manager import requires_mock_backend

class MobileTrainingDisconnectionTest:
    """Test mobile Safari disconnection during training scenarios"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.websocket_url = base_url.replace("http", "ws") + "/ws"
        self.logger = TestLogger()
        self.backend = BackendTester(base_url, self.logger)
        self.test_results = []
        
    async def test_websocket_connection_during_training_simulation(self) -> Dict[str, Any]:
        """Test WebSocket connection stability during simulated training activity"""
        self.logger.testing("Testing WebSocket connection during training simulation")
        
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                self.logger.ok("WebSocket connected for training simulation")
                
                # Send initial connection message
                await websocket.send(json.dumps({
                    "type": "ping", 
                    "timestamp": time.time(),
                    "client_type": "mobile_safari"
                }))
                
                # Wait for connection confirmation
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    self.logger.log(f"Connection response: {data.get('type', 'unknown')}")
                except asyncio.TimeoutError:
                    self.logger.warning("No initial response received")
                
                # Simulate training activity by monitoring for training messages
                start_time = time.time()
                training_messages = []
                connection_errors = 0
                total_messages = 0
                
                # Monitor for 60 seconds (simulating a training session)
                while time.time() - start_time < 60:
                    try:
                        # Send periodic ping to keep connection alive
                        if total_messages % 10 == 0:  # Every 10 seconds
                            await websocket.send(json.dumps({
                                "type": "ping",
                                "timestamp": time.time(),
                                "client_type": "mobile_safari"
                            }))
                        
                        # Try to receive messages
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        total_messages += 1
                        data = json.loads(message)
                        
                        # Track training-related messages
                        if data.get("type") in ["training_update", "checkpoint_playback", "training_status"]:
                            training_messages.append({
                                "type": data.get("type"),
                                "timestamp": time.time(),
                                "elapsed": time.time() - start_time
                            })
                            self.logger.debug(f"Training message: {data.get('type')}")
                        
                    except asyncio.TimeoutError:
                        # No message received, this is normal
                        pass
                    except websockets.exceptions.ConnectionClosed:
                        connection_errors += 1
                        self.logger.error(f"WebSocket connection closed at {time.time() - start_time:.1f}s")
                        break
                    except Exception as e:
                        connection_errors += 1
                        self.logger.error(f"WebSocket error at {time.time() - start_time:.1f}s: {e}")
                        break
                
                duration = time.time() - start_time
                return {
                    "success": connection_errors == 0,
                    "duration": duration,
                    "total_messages": total_messages,
                    "training_messages": len(training_messages),
                    "connection_errors": connection_errors,
                    "connection_stable": duration >= 55  # At least 55 seconds stable
                }
                
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": 0,
                "total_messages": 0,
                "training_messages": 0,
                "connection_errors": 1
            }
    
@requires_mock_backend
    def test_backend_stability_during_training_simulation(self) -> Dict[str, Any]:
        """Test backend stability during simulated training activity"""
        self.logger.testing("Testing backend stability during training simulation")
        
        # Simulate the backend load that would occur during training
        start_time = time.time()
        success_count = 0
        failure_count = 0
        total_requests = 0
        
        # Make rapid requests to simulate training activity
        for i in range(120):  # 2 minutes of requests every second
            try:
                # Alternate between different endpoints to simulate real training usage
                if i % 4 == 0:
                    result = self.backend.get_training_status()
                elif i % 4 == 1:
                    result = self.backend.get_checkpoint_stats()
                elif i % 4 == 2:
                    result = self.backend.test_connectivity()
                else:
                    result = self.backend.get_playback_status()
                
                total_requests += 1
                if result is not None:
                    success_count += 1
                else:
                    failure_count += 1
                    
            except Exception as e:
                failure_count += 1
                total_requests += 1
                self.logger.warning(f"Request {i+1} failed: {e}")
            
            # Log progress every 30 seconds
            elapsed = time.time() - start_time
            if i % 30 == 0 and i > 0:
                success_rate = success_count / total_requests * 100
                self.logger.log(f"  {elapsed:.1f}s: {success_rate:.1f}% success rate ({success_count}/{total_requests})")
            
            time.sleep(1)  # 1 second between requests
        
        duration = time.time() - start_time
        final_success_rate = success_count / total_requests * 100
        
        return {
            "success": final_success_rate >= 90,
            "duration": duration,
            "total_requests": total_requests,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": final_success_rate
        }
    
    async def test_concurrent_mobile_connections(self) -> Dict[str, Any]:
        """Test multiple concurrent mobile connections (simulating the 'other clients fail' scenario)"""
        self.logger.testing("Testing concurrent mobile connections")
        
        mobile_user_agents = [
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36"
        ]
        
        async def test_single_connection(user_agent: str, connection_id: int) -> Dict[str, Any]:
            """Test a single mobile connection"""
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    self.logger.log(f"Connection {connection_id} ({user_agent[:30]}...): Connected")
                    
                    # Send connection message
                    await websocket.send(json.dumps({
                        "type": "ping",
                        "timestamp": time.time(),
                        "client_type": f"mobile_{connection_id}",
                        "user_agent": user_agent
                    }))
                    
                    # Monitor connection for 30 seconds
                    start_time = time.time()
                    messages_received = 0
                    connection_stable = True
                    
                    while time.time() - start_time < 30:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                            messages_received += 1
                            data = json.loads(message)
                            
                            # Check for training-related messages
                            if data.get("type") in ["training_update", "checkpoint_playback"]:
                                self.logger.debug(f"Connection {connection_id}: Training message received")
                                
                        except asyncio.TimeoutError:
                            # Send keepalive ping
                            await websocket.send(json.dumps({
                                "type": "ping",
                                "timestamp": time.time(),
                                "client_type": f"mobile_{connection_id}"
                            }))
                        except websockets.exceptions.ConnectionClosed:
                            connection_stable = False
                            self.logger.error(f"Connection {connection_id}: Connection closed")
                            break
                        except Exception as e:
                            connection_stable = False
                            self.logger.error(f"Connection {connection_id}: Error - {e}")
                            break
                    
                    return {
                        "connection_id": connection_id,
                        "user_agent": user_agent,
                        "success": connection_stable,
                        "duration": time.time() - start_time,
                        "messages_received": messages_received
                    }
                    
            except Exception as e:
                self.logger.error(f"Connection {connection_id}: Failed to connect - {e}")
                return {
                    "connection_id": connection_id,
                    "user_agent": user_agent,
                    "success": False,
                    "error": str(e),
                    "duration": 0,
                    "messages_received": 0
                }
        
        # Start all connections concurrently
        tasks = []
        for i, user_agent in enumerate(mobile_user_agents):
            task = asyncio.create_task(test_single_connection(user_agent, i + 1))
            tasks.append(task)
        
        # Wait for all connections to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_connections = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        total_connections = len(results)
        
        return {
            "success": successful_connections == total_connections,
            "total_connections": total_connections,
            "successful_connections": successful_connections,
            "success_rate": successful_connections / total_connections * 100,
            "results": results
        }
    
@requires_mock_backend
    def test_refresh_recovery_scenario(self) -> Dict[str, Any]:
        """Test the refresh recovery scenario (refresh doesn't fix the problem)"""
        self.logger.testing("Testing refresh recovery scenario")
        
        # Simulate the scenario where the user would refresh the page
        # by testing if endpoints become unresponsive and stay that way
        
        # Step 1: Initial health check
        initial_health = self._check_endpoint_health()
        self.logger.log(f"Initial endpoint health: {initial_health['responsive_endpoints']}/{initial_health['total_endpoints']}")
        
        # Step 2: Simulate heavy load (like during training)
        self.logger.log("Simulating heavy load...")
        for i in range(50):  # 50 rapid requests
            try:
                self.backend.get_training_status()
                self.backend.get_checkpoint_stats()
                time.sleep(0.05)  # 50ms between requests
            except Exception as e:
                self.logger.warning(f"Load request {i+1} failed: {e}")
        
        # Step 3: Check health after load
        post_load_health = self._check_endpoint_health()
        self.logger.log(f"Post-load endpoint health: {post_load_health['responsive_endpoints']}/{post_load_health['total_endpoints']}")
        
        # Step 4: Simulate refresh attempts
        self.logger.log("Simulating refresh attempts...")
        refresh_successes = 0
        refresh_attempts = 5
        
        for i in range(refresh_attempts):
            try:
                # Simulate page refresh by checking all critical endpoints
                refresh_health = self._check_endpoint_health()
                if refresh_health['responsive_endpoints'] == refresh_health['total_endpoints']:
                    refresh_successes += 1
                    self.logger.ok(f"Refresh attempt {i+1}: Successful")
                else:
                    self.logger.error(f"Refresh attempt {i+1}: Failed")
                time.sleep(2)  # Wait between refresh attempts
            except Exception as e:
                self.logger.error(f"Refresh attempt {i+1}: Exception - {e}")
        
        refresh_success_rate = refresh_successes / refresh_attempts * 100
        
        return {
            "success": refresh_success_rate >= 80,  # At least 4/5 refresh attempts should work
            "initial_health": initial_health,
            "post_load_health": post_load_health,
            "refresh_attempts": refresh_attempts,
            "refresh_successes": refresh_successes,
            "refresh_success_rate": refresh_success_rate
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
        """Run all mobile training disconnection tests"""
        self.logger.banner("Mobile Training Disconnection Tests", 60)
        
        # Run tests in sequence to simulate the real scenario
        results = {
            "websocket_training_simulation": await self.test_websocket_connection_during_training_simulation(),
            "backend_stability": self.test_backend_stability_during_training_simulation(),
            "concurrent_connections": await self.test_concurrent_mobile_connections(),
            "refresh_recovery": self.test_refresh_recovery_scenario()
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
                    if test_name == "websocket_training_simulation":
                        self.logger.log(f"  Connection errors: {result.get('connection_errors', 0)}")
                        self.logger.log(f"  Duration: {result.get('duration', 0):.1f}s")
                    elif test_name == "backend_stability":
                        self.logger.log(f"  Success rate: {result.get('success_rate', 0):.1f}%")
                    elif test_name == "concurrent_connections":
                        self.logger.log(f"  Connection success rate: {result.get('success_rate', 0):.1f}%")
                    elif test_name == "refresh_recovery":
                        self.logger.log(f"  Refresh success rate: {result.get('refresh_success_rate', 0):.1f}%")
            else:
                self.logger.log(f"{test_name}: ERROR")
        
        self.logger.log(f"Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            self.logger.success("All mobile training disconnection tests passed!")
            self.logger.log("Note: This means the current system is stable.")
            self.logger.log("The mobile Safari issue may be intermittent or require specific conditions.")
        else:
            self.logger.error("Some mobile training disconnection tests failed!")
            self.logger.log("This may indicate the root cause of the mobile Safari disconnection issue.")
        
        return results

@requires_mock_backend("Mobile Training Disconnection Tests")
async def main():
    """Main entry point"""
    test = MobileTrainingDisconnectionTest()
    results = await test.run_all_tests()
    
    # Save results to file
    with open("mobile_training_disconnection_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    test.logger.info(f"Results saved to results file")

if __name__ == "__main__":
    asyncio.run(main()) 