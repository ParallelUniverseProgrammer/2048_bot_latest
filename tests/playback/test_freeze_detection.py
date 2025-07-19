#!/usr/bin/env python3
"""
Real Playback Freeze Detection Test
==================================

This test loads actual checkpoints and tries to reproduce the freezing issue
that can occur during live playback. It uses real checkpoint data and simulates
WebSocket clients to identify potential freeze conditions.

The test performs:
- Real checkpoint loading and validation
- Single game playback testing
- Live playback with timeout detection
- WebSocket stress testing
- Multiple checkpoint testing to identify problematic ones
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from tests.utilities.test_utils import TestLogger
from backend.app.models.checkpoint_playback import CheckpointPlayback
from backend.app.models.checkpoint_metadata import CheckpointManager
from backend.app.api.websocket_manager import WebSocketManager

class MockWebSocket:
    """Mock WebSocket for testing"""
    def __init__(self, slow=False):
        self.slow = slow
        self.messages = []
        self.closed = False
        
    async def accept(self):
        """Mock accept method"""
        pass
        
    async def send_text(self, message):
        """Mock send_text method"""
        self.messages.append(message)
        if self.slow:
            await asyncio.sleep(0.5)  # Simulate slow client

class RealPlaybackFreezeTest:
    """Test class to reproduce freezing with real checkpoints"""
    
    def __init__(self):
        self.logger = TestLogger()
        checkpoint_dir = os.getenv('CHECKPOINTS_DIR', os.path.join(os.path.dirname(__file__), '..', 'backend', 'checkpoints'))
        self.logger.info(f"Using checkpoint_dir: {checkpoint_dir}")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.websocket_manager = WebSocketManager()
        self.playback = None
        self.test_timeout = 60  # seconds
        self.freeze_detected = False
        self.mock_client = None
        
    async def connect_mock_client(self, slow=False):
        """Connect a mock WebSocket client to the WebSocketManager."""
        self.mock_client = MockWebSocket(slow=slow)
        await self.websocket_manager.connect(self.mock_client, user_agent="mock-test-client")
        self.logger.ok("Mock client connected")
    
    async def test_real_checkpoint_playback(self, checkpoint_id: str):
        """Test real checkpoint playback with timeout (asyncio only) and a simulated client."""
        self.logger.banner(f"Testing Real Checkpoint Playback: {checkpoint_id}", 60)
        try:
            # Create playback instance
            self.playback = CheckpointPlayback(self.checkpoint_manager)
            
            # Load checkpoint
            self.logger.info(f"Loading checkpoint {checkpoint_id}...")
            start_time = time.time()
            success = self.playback.load_checkpoint(checkpoint_id)
            load_time = time.time() - start_time
            
            if not success:
                self.logger.error(f"Failed to load checkpoint {checkpoint_id}")
                return False, False
                
            self.logger.ok(f"Checkpoint loaded in {load_time:.2f}s")
            
            # Test single game first
            self.logger.info("Testing single game playback...")
            start_time = time.time()
            game_result = self.playback.play_single_game()
            game_time = time.time() - start_time
            
            if 'error' in game_result:
                self.logger.error(f"Single game failed: {game_result['error']}")
                return False, False
                
            self.logger.ok(f"Single game completed in {game_time:.2f}s")
            self.logger.info(f"   Score: {game_result.get('final_score', 0)}")
            self.logger.info(f"   Steps: {game_result.get('steps', 0)}")
            self.logger.info(f"   Max tile: {game_result.get('max_tile', 0)}")
            
            # Simulate a connected client before starting live playback
            await self.connect_mock_client(slow=False)
            self.logger.info("Testing live playback (will timeout after 30 seconds)...")
            start_time = time.time()
            playback_task = asyncio.create_task(
                self.playback.start_live_playback(self.websocket_manager)
            )
            try:
                await asyncio.wait_for(playback_task, timeout=30.0)
                playback_time = time.time() - start_time
                self.logger.ok(f"Live playback completed in {playback_time:.2f}s")
                return True, False
            except asyncio.TimeoutError:
                playback_time = time.time() - start_time
                self.logger.warning(f"Live playback timed out after {playback_time:.2f}s - potential freeze!")
                self.playback.stop_playback()
                try:
                    await asyncio.wait_for(playback_task, timeout=5.0)
                    self.logger.ok("Playback stopped gracefully")
                except asyncio.TimeoutError:
                    self.logger.error("Playback failed to stop gracefully - confirmed freeze!")
                return False, True
        except Exception as e:
            self.logger.error(f"Exception during playback: {e}")
            import traceback
            traceback.print_exc()
            return False, False
    
    async def test_multiple_checkpoints(self):
        """Test multiple checkpoints to find problematic ones"""
        self.logger.banner("Testing Multiple Checkpoints for Freezing Issues", 60)
        checkpoints = self.checkpoint_manager.list_checkpoints()
        if not checkpoints:
            self.logger.warning("No checkpoints available for testing")
            return
        results = []
        for checkpoint in checkpoints[:3]:  # Test first 3 checkpoints
            checkpoint_id = checkpoint.id
            self.logger.info(f"--- Testing Checkpoint: {checkpoint_id} ---")
            start_time = time.time()
            try:
                # Wrap the whole test in a timeout
                success, froze = await asyncio.wait_for(
                    self.test_real_checkpoint_playback(checkpoint_id),
                    timeout=self.test_timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Entire test timed out after {self.test_timeout} seconds - potential freeze!")
                success, froze = False, True
            total_time = time.time() - start_time
            result = {
                'checkpoint_id': checkpoint_id,
                'success': success,
                'time': total_time,
                'froze': froze
            }
            results.append(result)
            # Brief pause between tests
            await asyncio.sleep(2.0)
        # Summary
        self.logger.banner("Checkpoint Test Summary", 60)
        for result in results:
            status = "PASS" if result['success'] else "FAIL"
            freeze_indicator = " (FROZE)" if result['froze'] else ""
            self.logger.info(f"{status} {result['checkpoint_id']}: {result['time']:.2f}s{freeze_indicator}")
        frozen_count = sum(1 for r in results if r['froze'])
        if frozen_count > 0:
            self.logger.warning(f"Found {frozen_count} checkpoint(s) that caused freezing!")
        else:
            self.logger.ok("No freezing detected in checkpoint tests")
    
    async def test_websocket_stress(self):
        """Test WebSocket stress conditions"""
        self.logger.banner("Testing WebSocket Stress Conditions", 60)
        for i in range(3):
            mock_ws = MockWebSocket(slow=(i == 0))  # First connection is slow
            await self.websocket_manager.connect(mock_ws, f"test_client_{i}")
        self.logger.info(f"Connected {self.websocket_manager.get_connection_count()} mock clients")
        start_time = time.time()
        try:
            for i in range(10):
                await self.websocket_manager.broadcast({
                    'type': 'stress_test',
                    'message_id': i,
                    'timestamp': time.time()
                })
                await asyncio.sleep(0.1)  # Small delay between broadcasts
            duration = time.time() - start_time
            self.logger.ok(f"Stress test completed in {duration:.2f}s")
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Stress test failed after {duration:.2f}s: {e}")
    
    async def run_all_tests(self):
        """Run all real playback tests"""
        self.logger.banner("Starting Real Playback Freeze Tests", 60)
        await self.test_multiple_checkpoints()
        await self.test_websocket_stress()
        self.logger.banner("Test Complete", 60)
        self.logger.info("If freezing was detected, check the specific checkpoint and conditions.")
        self.logger.info("If no freezing was detected, the issue might be:")
        self.logger.info("1. Related to specific client conditions")
        self.logger.info("2. Timing-dependent")
        self.logger.info("3. Related to system resources")
        self.logger.info("4. Caused by frontend-backend interaction")

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("Real Playback Freeze Detection Test", 60)
    
    async def run_tests():
        test = RealPlaybackFreezeTest()
        await test.run_all_tests()
    
    try:
        asyncio.run(run_tests())
        logger.success("All freeze detection tests completed")
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 