#!/usr/bin/env python3
"""
Real playback freeze reproduction test.
This test loads actual checkpoints and tries to reproduce the freezing issue.
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

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
        checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'backend', 'checkpoints')
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
        print("Mock client connected.")
    
    async def test_real_checkpoint_playback(self, checkpoint_id: str):
        """Test real checkpoint playback with timeout (asyncio only) and a simulated client."""
        print(f"\n=== Testing Real Checkpoint Playback: {checkpoint_id} ===")
        try:
            # Create playback instance
            self.playback = CheckpointPlayback(self.checkpoint_manager)
            
            # Load checkpoint
            print(f"Loading checkpoint {checkpoint_id}...")
            start_time = time.time()
            success = self.playback.load_checkpoint(checkpoint_id)
            load_time = time.time() - start_time
            
            if not success:
                print(f"ERROR: Failed to load checkpoint {checkpoint_id}")
                return False, False
                
            print(f"OK: Checkpoint loaded in {load_time:.2f}s")
            
            # Test single game first
            print("Testing single game playback...")
            start_time = time.time()
            game_result = self.playback.play_single_game()
            game_time = time.time() - start_time
            
            if 'error' in game_result:
                print(f"ERROR: Single game failed: {game_result['error']}")
                return False, False
                
            print(f"OK: Single game completed in {game_time:.2f}s")
            print(f"   Score: {game_result.get('final_score', 0)}")
            print(f"   Steps: {game_result.get('steps', 0)}")
            print(f"   Max tile: {game_result.get('max_tile', 0)}")
            
            # Simulate a connected client before starting live playback
            await self.connect_mock_client(slow=False)
            print("Testing live playback (will timeout after 30 seconds)...")
            start_time = time.time()
            playback_task = asyncio.create_task(
                self.playback.start_live_playback(self.websocket_manager)
            )
            try:
                await asyncio.wait_for(playback_task, timeout=30.0)
                playback_time = time.time() - start_time
                print(f"OK: Live playback completed in {playback_time:.2f}s")
                return True, False
            except asyncio.TimeoutError:
                playback_time = time.time() - start_time
                print(f"ALARM: Live playback timed out after {playback_time:.2f}s - potential freeze!")
                self.playback.stop_playback()
                try:
                    await asyncio.wait_for(playback_task, timeout=5.0)
                    print("OK: Playback stopped gracefully")
                except asyncio.TimeoutError:
                    print("ERROR: Playback failed to stop gracefully - confirmed freeze!")
                return False, True
        except Exception as e:
            print(f"ERROR: Exception during playback: {e}")
            import traceback
            traceback.print_exc()
            return False, False
    
    async def test_multiple_checkpoints(self):
        """Test multiple checkpoints to find problematic ones"""
        print("FIND: Testing Multiple Checkpoints for Freezing Issues")
        print("=" * 60)
        checkpoints = self.checkpoint_manager.list_checkpoints()
        if not checkpoints:
            print("No checkpoints available for testing")
            return
        results = []
        for checkpoint in checkpoints[:3]:  # Test first 3 checkpoints
            checkpoint_id = checkpoint.id
            print(f"\n--- Testing Checkpoint: {checkpoint_id} ---")
            start_time = time.time()
            try:
                # Wrap the whole test in a timeout
                success, froze = await asyncio.wait_for(
                    self.test_real_checkpoint_playback(checkpoint_id),
                    timeout=self.test_timeout
                )
            except asyncio.TimeoutError:
                print(f"ALARM: Entire test timed out after {self.test_timeout} seconds - potential freeze!")
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
        print("\n" + "=" * 60)
        print("SUMMARY: Checkpoint Test Summary:")
        print("=" * 60)
        for result in results:
            status = "OK: PASS" if result['success'] else "ERROR: FAIL"
            freeze_indicator = " (FROZE)" if result['froze'] else ""
            print(f"{status} {result['checkpoint_id']}: {result['time']:.2f}s{freeze_indicator}")
        frozen_count = sum(1 for r in results if r['froze'])
        if frozen_count > 0:
            print(f"\nFIND: Found {frozen_count} checkpoint(s) that caused freezing!")
        else:
            print(f"\nOK: No freezing detected in checkpoint tests")
    async def test_websocket_stress(self):
        """Test WebSocket stress conditions"""
        print("\n=== Testing WebSocket Stress Conditions ===")
        for i in range(3):
            mock_ws = MockWebSocket(slow=(i == 0))  # First connection is slow
            await self.websocket_manager.connect(mock_ws, f"test_client_{i}")
        print(f"Connected {self.websocket_manager.get_connection_count()} mock clients")
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
            print(f"OK: Stress test completed in {duration:.2f}s")
        except Exception as e:
            duration = time.time() - start_time
            print(f"ERROR: Stress test failed after {duration:.2f}s: {e}")
    async def run_all_tests(self):
        """Run all real playback tests"""
        print("TESTING: Starting Real Playback Freeze Tests")
        print("=" * 60)
        await self.test_multiple_checkpoints()
        await self.test_websocket_stress()
        print("\n" + "=" * 60)
        print("TARGET: Test Complete")
        print("=" * 60)
        print("If freezing was detected, check the specific checkpoint and conditions.")
        print("If no freezing was detected, the issue might be:")
        print("1. Related to specific client conditions")
        print("2. Timing-dependent")
        print("3. Related to system resources")
        print("4. Caused by frontend-backend interaction")

async def main():
    """Main test runner"""
    test = RealPlaybackFreezeTest()
    await test.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 