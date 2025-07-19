#!/usr/bin/env python3
"""
Test for reconnection failure during REAL training.
This test starts actual training with a real checkpoint and then tests reconnection
during the active training process to reproduce the mobile Safari issue.
"""

import asyncio
import json
import time
import websockets
import requests
from typing import Dict, Any, List
from test_utils import TestLogger, BackendTester, check_backend_or_start_mock

class RealTrainingReconnectionTest:
    """Test reconnection during actual training scenarios"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.websocket_url = base_url.replace("http", "ws") + "/ws"
        self.logger = TestLogger()
        self.backend = BackendTester(base_url, self.logger)
        self.training_started = False
        
    def start_real_training(self) -> Dict[str, Any]:
        """Start actual training with a real checkpoint"""
        self.logger.testing("Starting real training with checkpoint")
        
        # First, get available checkpoints
        checkpoints = self.backend.get_checkpoints()
        if not checkpoints:
            self.logger.error("No checkpoints available for training")
            return {"success": False, "error": "No checkpoints available"}
        
        # Use the first available checkpoint
        checkpoint_id = checkpoints[0]["id"]
        self.logger.log(f"Using checkpoint: {checkpoint_id}")
        
        # Start training
        try:
            response = requests.post(f"{self.base_url}/training/start", json={
                "checkpoint_id": checkpoint_id,
                "episodes": 10  # Just a few episodes for testing
            }, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                self.logger.ok("Training started successfully")
                self.logger.log(f"Training response: {result}")
                self.training_started = True
                return {"success": True, "checkpoint_id": checkpoint_id, "response": result}
            else:
                self.logger.error(f"Failed to start training: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            self.logger.error(f"Exception starting training: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_training(self) -> Dict[str, Any]:
        """Stop the training process"""
        if not self.training_started:
            return {"success": True, "message": "No training to stop"}
        
        try:
            response = requests.post(f"{self.base_url}/training/stop", timeout=10)
            if response.status_code == 200:
                self.logger.ok("Training stopped successfully")
                self.training_started = False
                return {"success": True}
            else:
                self.logger.error(f"Failed to stop training: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            self.logger.error(f"Exception stopping training: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_reconnection_during_real_training(self) -> Dict[str, Any]:
        """Test reconnection during actual training"""
        self.logger.testing("Testing reconnection during real training")
        
        # Step 1: Start real training
        training_result = self.start_real_training()
        if not training_result["success"]:
            return {"success": False, "error": "Failed to start training"}
        
        # Wait a moment for training to initialize
        await asyncio.sleep(3)
        
        # Step 2: Establish initial connection during training
        self.logger.log("Step 2: Establishing initial connection during training...")
        try:
            async with websockets.connect(self.websocket_url) as initial_websocket:
                self.logger.ok("Initial WebSocket connected during training")
                
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
                
                # Step 3: Monitor for training messages
                self.logger.log("Step 3: Monitoring for training messages...")
                start_time = time.time()
                training_messages = []
                total_messages = 0
                
                # Monitor for 15 seconds to capture training activity
                while time.time() - start_time < 15:
                    try:
                        message = await asyncio.wait_for(initial_websocket.recv(), timeout=1.0)
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
                        else:
                            self.logger.debug(f"Other message: {data.get('type')}")
                            
                    except asyncio.TimeoutError:
                        # Send keepalive
                        await initial_websocket.send(json.dumps({
                            "type": "ping",
                            "timestamp": time.time()
                        }))
                
                self.logger.log(f"Initial connection active for {time.time() - start_time:.1f}s")
                self.logger.log(f"Total messages: {total_messages}, Training messages: {len(training_messages)}")
                
        except Exception as e:
            self.logger.error(f"Initial connection failed: {e}")
            return {"success": False, "error": f"Initial connection failed: {e}"}
        
        # Step 4: Close connection (simulating disconnection)
        self.logger.log("Step 4: Closing connection (simulating disconnection)...")
        
        # Step 5: Attempt to reconnect during active training
        self.logger.log("Step 5: Attempting reconnections during active training...")
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
                        training_messages_received = 0
                        connection_stable = True
                        
                        while time.time() - test_start < 8:  # Test for 8 seconds
                            try:
                                message = await asyncio.wait_for(new_websocket.recv(), timeout=1.0)
                                test_messages += 1
                                data = json.loads(message)
                                
                                # Check for training messages
                                if data.get("type") in ["training_update", "checkpoint_playback", "training_status"]:
                                    training_messages_received += 1
                                    self.logger.debug(f"Reconnection {attempt + 1} training message: {data.get('type')}")
                                else:
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
                            self.logger.ok(f"Reconnection {attempt + 1}: Stable for 8s, {test_messages} messages, {training_messages_received} training messages")
                        else:
                            self.logger.error(f"Reconnection {attempt + 1}: Unstable")
                        
                        reconnection_results.append({
                            "attempt": attempt + 1,
                            "connected": True,
                            "stable": connection_stable,
                            "test_messages": test_messages,
                            "training_messages": training_messages_received,
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
        
        # Step 6: Stop training
        self.logger.log("Step 6: Stopping training...")
        stop_result = self.stop_training()
        
        return {
            "success": reconnection_success_rate >= 80,  # At least 4/5 reconnections should work
            "training_started": training_result["success"],
            "training_stopped": stop_result["success"],
            "reconnection_attempts": reconnection_attempts,
            "successful_reconnections": successful_reconnections,
            "reconnection_success_rate": reconnection_success_rate,
            "reconnection_results": reconnection_results
        }
    
    def test_backend_endpoints_during_training(self) -> Dict[str, Any]:
        """Test backend endpoints during active training"""
        self.logger.testing("Testing backend endpoints during active training")
        
        # Start training first
        training_result = self.start_real_training()
        if not training_result["success"]:
            return {"success": False, "error": "Failed to start training"}
        
        # Wait for training to initialize
        time.sleep(3)
        
        # Test endpoints during training
        self.logger.log("Testing endpoints during active training...")
        endpoint_successes = 0
        endpoint_attempts = 10
        
        for i in range(endpoint_attempts):
            try:
                # Test different endpoints
                if i % 3 == 0:
                    result = self.backend.get_training_status()
                elif i % 3 == 1:
                    result = self.backend.get_checkpoint_stats()
                else:
                    result = self.backend.test_connectivity()
                
                if result is not None:
                    endpoint_successes += 1
                    self.logger.debug(f"Endpoint test {i+1}: Success")
                else:
                    self.logger.warning(f"Endpoint test {i+1}: Failed")
                    
                time.sleep(1)  # 1 second between tests
                
            except Exception as e:
                self.logger.warning(f"Endpoint test {i+1}: Exception - {e}")
        
        endpoint_success_rate = endpoint_successes / endpoint_attempts * 100
        
        # Stop training
        stop_result = self.stop_training()
        
        return {
            "success": endpoint_success_rate >= 90,
            "training_started": training_result["success"],
            "training_stopped": stop_result["success"],
            "endpoint_attempts": endpoint_attempts,
            "endpoint_successes": endpoint_successes,
            "endpoint_success_rate": endpoint_success_rate
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all real training reconnection tests"""
        self.logger.banner("Real Training Reconnection Tests", 60)
        
        # Ensure backend is available
        if not check_backend_or_start_mock():
            self.logger.error("No backend available for testing")
            return {"error": "No backend available"}
        
        # Run tests in sequence
        results = {
            "websocket_reconnection_during_training": await self.test_reconnection_during_real_training(),
            "endpoint_stability_during_training": self.test_backend_endpoints_during_training()
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
                    if test_name == "websocket_reconnection_during_training":
                        self.logger.log(f"  Reconnection success rate: {result.get('reconnection_success_rate', 0):.1f}%")
                        self.logger.log(f"  Successful reconnections: {result.get('successful_reconnections', 0)}/{result.get('reconnection_attempts', 0)}")
                    elif test_name == "endpoint_stability_during_training":
                        self.logger.log(f"  Endpoint success rate: {result.get('endpoint_success_rate', 0):.1f}%")
            else:
                self.logger.log(f"{test_name}: ERROR")
        
        self.logger.log(f"Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            self.logger.success("All real training reconnection tests passed!")
            self.logger.log("Note: This means reconnection works correctly during real training.")
            self.logger.log("The mobile Safari issue may be specific to mobile browsers or network conditions.")
        else:
            self.logger.error("Some real training reconnection tests failed!")
            self.logger.log("This indicates the root cause of the reconnection issue during training.")
        
        return results

async def main():
    """Main entry point"""
    test = RealTrainingReconnectionTest()
    results = await test.run_all_tests()
    
    # Save results to file
    with open("real_training_reconnection_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to results file")

if __name__ == "__main__":
    asyncio.run(main()) 