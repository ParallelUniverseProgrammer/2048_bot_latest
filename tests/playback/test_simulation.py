#!/usr/bin/env python3
"""
Playback Simulation Test Suite
==============================

This test suite simulates various WebSocket failure scenarios to test the
robustness of the checkpoint playback system. It uses dummy components to
isolate and test specific failure conditions without requiring real checkpoints
or models.

The test scenarios include:
- Normal WebSocket operation (baseline)
- WebSocket timeout conditions
- WebSocket exception handling
- Graceful shutdown under failure conditions
"""

import sys
import os
import asyncio
import time
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from tests.utilities.test_utils import TestLogger
from backend.app.models.checkpoint_playback import CheckpointPlayback

class DummyCheckpointManager:
    def get_checkpoint_metadata(self, checkpoint_id):
        class Meta:
            nickname = 'dummy'
            episode = 0
            model_config = {}
            performance_metrics = {}
        return Meta()

    def _get_checkpoint_path(self, checkpoint_id):
        from pathlib import Path
        return Path('nonexistent.ckpt')

class BaseDummyWS:
    """Base class for fake websocket managers"""
    def __init__(self):
        self.counter = 0

    async def broadcast(self, message: dict):  # pragma: no cover
        raise NotImplementedError()

    def get_connection_count(self):
        return 1

class NormalWS(BaseDummyWS):
    async def broadcast(self, message: dict):
        # Always succeed with tiny latency
        await asyncio.sleep(0.001)

class TimeoutWS(BaseDummyWS):
    async def broadcast(self, message: dict):
        if message.get('type') == 'checkpoint_playback':
            self.counter += 1
        # After 20 steps simulate send timeout by sleeping forever
        if self.counter >= 20:
            await asyncio.sleep(10)  # longer than playback.broadcast_timeout
        else:
            await asyncio.sleep(0.001)

class ExceptionWS(BaseDummyWS):
    async def broadcast(self, message: dict):
        if message.get('type') == 'checkpoint_playback':
            self.counter += 1
        if self.counter >= 20:
            raise asyncio.TimeoutError("Simulated send_text timeout")
        await asyncio.sleep(0.001)

async def _run_playback(websocket_manager, runtime: float = 6.0) -> CheckpointPlayback:
    manager = DummyCheckpointManager()
    playback = CheckpointPlayback(manager)
    # Inject dummy model & deterministic action
    playback.current_model = object()
    playback.current_checkpoint_id = 'dummy_ckpt'
    playback.current_config = {}
    playback.select_action = lambda state, legal, env_game: (legal[0], [0.25]*4, None)
    # Run playback in background
    task = asyncio.create_task(playback.start_live_playback(websocket_manager))
    await asyncio.sleep(runtime)
    playback.stop_playback()
    # Allow graceful shutdown
    try:
        await asyncio.wait_for(task, timeout=5)
    except asyncio.TimeoutError:
        # Task did not finish -> hang detected
        logger.error("[TEST] Playback task hung during shutdown!")
    return playback

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("Playback Simulation Test Suite", 60)
    
    async def run_tests():
        logger.info("Scenario 1: All broadcasts succeed")
        pb1 = await _run_playback(NormalWS())
        assert not pb1.is_playing, "Playback should have stopped"
        logger.ok(f"Completed without hang, errors: {len(pb1.get_error_history())}")

        logger.info("Scenario 2: send_text sleeps forever (timeout in wait_for)")
        pb2 = await _run_playback(TimeoutWS())
        assert not pb2.is_playing, "Playback should have stopped even after timeouts"
        assert pb2.consecutive_failures >= 1, "Should record failures"
        logger.ok(f"Handled long-running send without hang, errors: {len(pb2.get_error_history())}")

        logger.info("Scenario 3: send_text raises TimeoutError")
        pb3 = await _run_playback(ExceptionWS())
        assert not pb3.is_playing, "Playback should have stopped after raised exception"
        assert pb3.consecutive_failures >= 1, "Should record failures"
        logger.ok(f"Handled raised exception, errors: {len(pb3.get_error_history())}")

        logger.success("All playback robustness tests passed")
    
    try:
        asyncio.run(run_tests())
        return True
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 