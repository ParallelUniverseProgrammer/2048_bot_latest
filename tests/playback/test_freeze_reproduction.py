from tests.utilities.backend_manager import requires_mock_backend
#!/usr/bin/env python3
"""
Freeze Reproduction Test Suite
==============================

Focused test to reproduce the server freezing issue during checkpoint playback.
This test targets the most likely culprits identified in our analysis.

The test covers:
- WebSocket broadcast deadlock scenarios
- Action selection infinite loops
- Checkpoint loading issues
- Concurrent operation conflicts
- Resource exhaustion conditions
"""

import sys
import os
import asyncio
import time
import threading
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from tests.utilities.test_utils import TestLogger
from backend.app.models.checkpoint_playback import CheckpointPlayback
from backend.app.models.checkpoint_metadata import CheckpointManager
from backend.app.api.websocket_manager import WebSocketManager
from backend.app.environment.gym_2048_env import Gym2048Env
from backend.app.utils.action_selection import select_action_with_fallback_for_playback

class MockWebSocket:
    """Mock WebSocket for testing"""
    def __init__(self, slow=False, fail_after=None):
        self.slow = slow
        self.fail_after = fail_after
        self.send_count = 0
        self.closed = False
        
    async def send_text(self, message):
        self.send_count += 1
        
        if self.fail_after and self.send_count > self.fail_after:
            raise Exception("Mock WebSocket failure")
            
        if self.slow:
            # Simulate slow client
            await asyncio.sleep(2.0)
            
        # Simulate some processing time
        await asyncio.sleep(0.01)

class MockWebSocketManager:
    """Mock WebSocket manager that can simulate various failure scenarios"""
    
    def __init__(self, failure_mode="none"):
        self.connections = []
        self.failure_mode = failure_mode
        self.broadcast_count = 0
        self.last_broadcast_time = time.time()
        
    async def connect(self, websocket, user_agent=""):
        """Mock connection"""
        conn_info = type('ConnectionInfo', (), {
            'websocket': websocket,
            'is_mobile': False,
            'last_heartbeat': time.time()
        })()
        self.connections.append(conn_info)
        
    def get_connection_count(self):
        return len(self.connections)
        
    async def broadcast(self, message):
        """Mock broadcast with various failure modes"""
        self.broadcast_count += 1
        self.last_broadcast_time = time.time()
        
        if self.failure_mode == "slow_clients":
            # Simulate slow clients that cause broadcast to hang
            for conn in self.connections:
                try:
                    await conn.websocket.send_text("test")
                except Exception as e:
                    logger.error(f"Client send failed: {e}")
                    
        elif self.failure_mode == "deadlock":
            # Simulate broadcast deadlock
            await asyncio.sleep(30.0)  # This will cause the test to hang
            
        elif self.failure_mode == "timeout":
            # Simulate timeout after some broadcasts
            if self.broadcast_count > 5:
                await asyncio.sleep(10.0)
                
        else:
            # Normal broadcast
            for conn in self.connections:
                try:
                    await conn.websocket.send_text("test")
                except Exception as e:
                    logger.error(f"Client send failed: {e}")

class FreezeReproductionTest:
    """Test class to reproduce freezing issues"""
    
    def __init__(self):
        self.logger = TestLogger()
        checkpoint_dir = os.getenv('CHECKPOINTS_DIR', os.path.join(os.path.dirname(__file__), '..', 'backend', 'checkpoints'))
        self.logger.info(f"Using checkpoint_dir: {checkpoint_dir}")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.test_results = []
        
    def log_test(self, test_name, success, duration, error=None):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'duration': duration,
            'error': str(error) if error else None,
            'timestamp': time.time()
        }
        self.test_results.append(result)
        if success:
            self.logger.ok(f"{test_name}: {duration:.2f}s")
        else:
            self.logger.error(f"{test_name}: {duration:.2f}s")
            if error:
                self.logger.error(f"   Error: {error}")
            
    async def test_websocket_broadcast_deadlock(self):
        """Test WebSocket broadcast deadlock scenario"""
        self.logger.banner("Testing WebSocket Broadcast Deadlock", 60)
        
        # Test 1: Normal broadcast
        start_time = time.time()
        try:
            ws_manager = MockWebSocketManager("none")
            await ws_manager.connect(MockWebSocket(), "test")
            
            # Send multiple broadcasts quickly
            for i in range(10):
                await ws_manager.broadcast({"type": "test", "data": i})
                
            duration = time.time() - start_time
            self.log_test("Normal WebSocket Broadcast", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Normal WebSocket Broadcast", False, duration, e)
            
        # Test 2: Slow clients
        start_time = time.time()
        try:
            ws_manager = MockWebSocketManager("slow_clients")
            await ws_manager.connect(MockWebSocket(slow=True), "test")
            await ws_manager.connect(MockWebSocket(slow=True), "test")
            
            # This should be slow but not hang
            await asyncio.wait_for(
                ws_manager.broadcast({"type": "test", "data": "slow"}),
                timeout=10.0
            )
            
            duration = time.time() - start_time
            self.log_test("Slow Client Broadcast", True, duration)
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.log_test("Slow Client Broadcast", False, duration, "Timeout - potential deadlock")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Slow Client Broadcast", False, duration, e)
            
        # Test 3: Deadlock simulation (with timeout)
        start_time = time.time()
        try:
            ws_manager = MockWebSocketManager("deadlock")
            await ws_manager.connect(MockWebSocket(), "test")
            
            # This should timeout
            await asyncio.wait_for(
                ws_manager.broadcast({"type": "test", "data": "deadlock"}),
                timeout=5.0
            )
            
            duration = time.time() - start_time
            self.log_test("Deadlock Detection", False, duration, "Should have timed out")
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.log_test("Deadlock Detection", True, duration, "Correctly detected deadlock")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Deadlock Detection", False, duration, e)
    
    async def test_action_selection_infinite_loop(self):
        """Test for infinite loops in action selection"""
        logger.info("\n=== Testing Action Selection Infinite Loop ===")
        
        # Create a mock model that could cause issues
        class MockModel:
            def __init__(self, device):
                self.device = device
                self.attention_weights = None
                
            def __call__(self, state_tensor):
                # Simulate model forward pass
                import torch
                batch_size = state_tensor.shape[0]
                policy_logits = torch.randn(batch_size, 4).to(self.device)
                value = torch.randn(batch_size, 1).to(self.device)
                return policy_logits, value
                
        start_time = time.time()
        try:
            # Test action selection with various game states
            env = Gym2048Env()
            device = torch.device('cpu')
            model = MockModel(device)
            
            # Test multiple game states
            for game_num in range(5):
                state, _ = env.reset()
                
                # Play a few moves
                for step in range(10):
                    if env.is_done():
                        break
                        
                    legal_actions = env.get_legal_actions()
                    if not legal_actions:
                        break
                        
                    # This is where infinite loops could occur
                    action, probs, attention = select_action_with_fallback_for_playback(
                        model, state, legal_actions, env.game, device
                    )
                    
                    # Take the action
                    state, reward, done, _, _ = env.step(action)
                    
                    if done:
                        break
                        
            duration = time.time() - start_time
            self.log_test("Action Selection Loop", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Action Selection Loop", False, duration, e)
    
    async def test_checkpoint_loading_issues(self):
        """Test checkpoint loading for potential issues"""
        self.logger.banner("Testing Checkpoint Loading Issues", 60)
        
        # Get available checkpoints
        checkpoints = self.checkpoint_manager.list_checkpoints()
        if not checkpoints:
            self.logger.warning("No checkpoints available for testing")
            return
            
        # Test loading the first checkpoint
        checkpoint_id = checkpoints[0].id
        start_time = time.time()
        
        try:
            playback = CheckpointPlayback(self.checkpoint_manager)
            
            # Test loading with timeout
            success = await asyncio.wait_for(
                asyncio.to_thread(playback.load_checkpoint, checkpoint_id),
                timeout=30.0
            )
            
            duration = time.time() - start_time
            self.log_test(f"Checkpoint Loading ({checkpoint_id})", success, duration)
            
            if success:
                # Test single game with timeout
                start_time = time.time()
                try:
                    game_result = await asyncio.wait_for(
                        asyncio.to_thread(playback.play_single_game),
                        timeout=60.0
                    )
                    
                    duration = time.time() - start_time
                    has_error = 'error' in game_result
                    self.log_test(f"Single Game Playback", not has_error, duration, 
                                game_result.get('error') if has_error else None)
                    
                except asyncio.TimeoutError:
                    duration = time.time() - start_time
                    self.log_test("Single Game Playback", False, duration, "Timeout - potential infinite loop")
                except Exception as e:
                    duration = time.time() - start_time
                    self.log_test("Single Game Playback", False, duration, e)
                    
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.log_test(f"Checkpoint Loading ({checkpoint_id})", False, duration, "Timeout")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(f"Checkpoint Loading ({checkpoint_id})", False, duration, e)
    
    async def test_concurrent_operations(self):
        """Test concurrent operations that could cause deadlocks"""
        logger.info("\n=== Testing Concurrent Operations ===")
        
        start_time = time.time()
        try:
            # Simulate multiple concurrent operations
            tasks = []
            
            # Task 1: WebSocket broadcasts
            async def broadcast_task():
                ws_manager = MockWebSocketManager()
                await ws_manager.connect(MockWebSocket(), "test")
                for i in range(5):
                    await ws_manager.broadcast({"type": "concurrent", "id": i})
                    
            # Task 2: Game simulation
            async def game_task():
                env = Gym2048Env()
                state, _ = env.reset()
                for _ in range(10):
                    if env.is_done():
                        break
                    legal_actions = env.get_legal_actions()
                    if legal_actions:
                        action = legal_actions[0]  # Simple action selection
                        state, _, done, _, _ = env.step(action)
                        if done:
                            break
                    await asyncio.sleep(0.01)
                    
            # Task 3: Checkpoint operations
            async def checkpoint_task():
                checkpoints = self.checkpoint_manager.list_checkpoints()
                if checkpoints:
                    # Just test metadata access
                    metadata = self.checkpoint_manager.get_checkpoint_metadata(checkpoints[0].id)
                    await asyncio.sleep(0.1)
                    
            # Run all tasks concurrently
            tasks = [
                asyncio.create_task(broadcast_task()),
                asyncio.create_task(game_task()),
                asyncio.create_task(checkpoint_task())
            ]
            
            # Wait for all with timeout
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=30.0)
            
            duration = time.time() - start_time
            self.log_test("Concurrent Operations", True, duration)
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.log_test("Concurrent Operations", False, duration, "Timeout - potential deadlock")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Concurrent Operations", False, duration, e)
    
    async def run_all_tests(self):
        """Run all freeze reproduction tests"""
        self.logger.banner("Starting Freeze Reproduction Tests", 60)
        
        await self.test_websocket_broadcast_deadlock()
        await self.test_action_selection_infinite_loop()
        await self.test_checkpoint_loading_issues()
        await self.test_concurrent_operations()
        
        # Summary
        self.logger.banner("Test Summary", 60)
        
        passed = sum(1 for r in self.test_results if r['success'])
        total = len(self.test_results)
        
        for result in self.test_results:
            if result['success']:
                self.logger.ok(f"{result['test']} ({result['duration']:.2f}s)")
            else:
                self.logger.error(f"{result['test']} ({result['duration']:.2f}s)")
                if result['error']:
                    self.logger.error(f"    Error: {result['error']}")
                
        self.logger.info(f"Overall: {passed}/{total} tests passed")
        
        if passed < total:
            self.logger.warning("Potential freezing issues detected!")
            self.logger.info("Check the failed tests above for specific problems.")
        else:
            self.logger.ok("All tests passed - no obvious freezing issues detected")
            self.logger.info("The freezing might be related to specific conditions not covered by these tests.")
@requires_mock_backend

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("Freeze Reproduction Test Suite", 60)
    
    async def run_tests():
        test = FreezeReproductionTest()
        await test.run_all_tests()
    
    try:
        import torch
        asyncio.run(run_tests())
        logger.success("Freeze reproduction tests completed")
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 