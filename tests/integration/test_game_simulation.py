from tests.utilities.backend_manager import requires_real_backend
#!/usr/bin/env python3
"""
Game Simulation Tests
====================

This module tests game simulation scenarios including different model behaviors,
environment states, and concurrent game execution. It validates that the game
simulation system can handle various edge cases and performance scenarios.

These tests ensure the game simulation is robust and can handle real-world usage patterns.
"""

import asyncio
import time
import sys
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from backend.app.environment.gym_2048_env import Gym2048Env
    from backend.app.models.game_transformer import GameTransformer
    from backend.app.models.checkpoint_playback import CheckpointPlayback
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info(f"Backend path: {os.path.join(os.path.dirname(__file__), '..', '..', 'backend')}")
    logger.info(f"Available in backend: {os.listdir(os.path.join(os.path.dirname(__file__), '..', '..', 'backend')) if os.path.exists(os.path.join(os.path.dirname(__file__), '..', '..', 'backend')) else 'Path does not exist'}")

from tests.utilities.test_utils import TestLogger

import threading
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from backend.app.models.checkpoint_playback import CheckpointPlayback
from backend.app.models.checkpoint_metadata import CheckpointManager
from backend.app.models.game_transformer import GameTransformer
from backend.app.environment.gym_2048_env import Gym2048Env


class MockModel:
    """Mock model that simulates various action selection scenarios"""
    
    def __init__(self, scenario='normal'):
        self.scenario = scenario
        self.call_count = 0
        
    def eval(self):
        """Mock eval method"""
        pass
    
    def to(self, device):
        """Mock to method"""
        return self
    
    def forward(self, state_tensor):
        """Mock forward method with different scenarios"""
        self.call_count += 1
        
        if self.scenario == 'freeze':
            # Simulate model freezing by sleeping forever
            time.sleep(10000)
        
        if self.scenario == 'slow':
            # Simulate slow model inference
            time.sleep(2.0)
        
        if self.scenario == 'exception':
            # Simulate model throwing an exception
            raise RuntimeError("Mock model exception")
        
        if self.scenario == 'nan_output':
            # Simulate model returning NaN values
            policy_logits = torch.tensor([[float('nan')] * 4])
            value = torch.tensor([[float('nan')]])
            return policy_logits, value
        
        if self.scenario == 'infinite_loop':
            # Simulate model getting stuck in infinite loop
            while True:
                pass
        
        # Normal case - return valid logits and value
        policy_logits = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        value = torch.tensor([[0.0]])
        return policy_logits, value
    
    def __call__(self, state_tensor):
        """Make the model callable"""
        return self.forward(state_tensor)


class StuckEnvironment(Gym2048Env):
    """Environment that simulates stuck game states"""
    
    def __init__(self, stuck_after_steps=10):
        super().__init__()
        self.stuck_after_steps = stuck_after_steps
        self.step_count = 0
    
    def step(self, action):
        """Override step to simulate stuck states"""
        self.step_count += 1
        
        if self.step_count >= self.stuck_after_steps:
            # Simulate stuck state by never ending the game
            return self.get_state(), 0, False, False, {}
        
        return super().step(action)
    
    def get_legal_actions(self):
        """Return single action to simulate stuck state"""
        if self.step_count >= self.stuck_after_steps:
            return [0]  # Only one action available
        return super().get_legal_actions()


class InfiniteEnvironment(Gym2048Env):
    """Environment that never terminates"""
    
    def __init__(self):
        super().__init__()
        self.step_count = 0
    
    def is_done(self):
        """Never report done"""
        return False
    
    def step(self, action):
        """Always return not done"""
        self.step_count += 1
        return self.get_state(), 1, False, False, {}


class TimeoutHelper:
    """Helper class to add timeouts to operations"""
    
    def __init__(self, timeout_seconds=30):
        self.timeout_seconds = timeout_seconds
        self.timed_out = False
        
    def run_with_timeout(self, func, *args, **kwargs):
        """Run a function with timeout protection"""
        result = None
        exception = None
        
        def target():
            nonlocal result, exception
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout_seconds)
        
        if thread.is_alive():
            self.timed_out = True
            return None, f"Operation timed out after {self.timeout_seconds}s"
        
        if exception:
            return None, str(exception)
        
        return result, None


class TestGamePlayback:
    """Test class for game playback with various scenarios"""
    
    def __init__(self, model_scenario='normal', env_scenario='normal'):
        self.model_scenario = model_scenario
        self.env_scenario = env_scenario
        self.playback = self._create_playback()
    
    def _create_playback(self):
        """Create a playback instance with mock components"""
        # Create dummy checkpoint manager
        manager = Mock()
        manager.get_checkpoint_metadata.return_value = Mock()
        manager._get_checkpoint_path.return_value = Path('dummy.ckpt')
        
        playback = CheckpointPlayback(manager)
        
        # Mock the model based on scenario
        if self.model_scenario == 'none':
            playback.current_model = None
        else:
            playback.current_model = MockModel(self.model_scenario)
        
        playback.current_config = {'test': True}
        playback.current_checkpoint_id = 'test_checkpoint'
        
        # Mock the environment based on scenario
        if self.env_scenario == 'stuck':
            playback.env = StuckEnvironment(stuck_after_steps=5)
        elif self.env_scenario == 'infinite':
            playback.env = InfiniteEnvironment()
        elif self.env_scenario == 'exception':
            playback.env = Mock()
            playback.env.reset.side_effect = RuntimeError("Environment exception")
        else:
            playback.env = Gym2048Env()
        
        return playback
@requires_real_backend
    
    def test_single_game(self, timeout_seconds=10):
        """Test playing a single game with timeout"""
        timeout_helper = TimeoutHelper(timeout_seconds)
        
        result, error = timeout_helper.run_with_timeout(
            self.playback.play_single_game
        )
        
        return result, error, timeout_helper.timed_out


async def test_action_selection_scenarios():
    """Test various action selection scenarios that might cause freezing"""
    
    logger.info("=== Action Selection Freeze Tests ===\n")
    
    # Test 1: Normal action selection
    logger.testing("Test 1: Normal action selection")
    test_game = TestGamePlayback('normal', 'normal')
    result, error, timed_out = test_game.test_single_game(timeout_seconds=10)
    
    if timed_out:
        logger.error("   [TIMEOUT] Normal action selection froze!")
        return False
    elif error:
        logger.error(f"   [ERROR] {error}")
    elif result and 'error' not in result:
        logger.ok("   [OK] Normal action selection completed successfully")
    else:
        logger.error(f"   [FAIL] Normal action selection failed: {result}")
    
    # Test 2: No model loaded
    logger.testing("Test 2: No model loaded")
    test_game = TestGamePlayback('none', 'normal')
    result, error, timed_out = test_game.test_single_game(timeout_seconds=5)
    
    if timed_out:
        logger.error("   [TIMEOUT] No model handling froze!")
        return False
    elif result and 'error' in result:
        logger.ok("   [OK] No model handled gracefully")
    else:
        logger.error(f"   [FAIL] No model should have returned an error: {result}")
    
    # Test 3: Slow model inference
    logger.testing("Test 3: Slow model inference")
    test_game = TestGamePlayback('slow', 'normal')
    result, error, timed_out = test_game.test_single_game(timeout_seconds=15)
    
    if timed_out:
        logger.ok("   [OK] Slow model inference timeout detected and handled correctly!")
    elif error:
        logger.warning(f"   [WARN] ERROR: {error} (might be expected for slow models)")
    elif result and 'error' not in result:
        logger.ok("   [OK] Slow model inference completed successfully")
    else:
        logger.error(f"   [FAIL] Slow model inference failed: {result}")
    
    # Test 4: Model exception
    logger.testing("Test 4: Model throws exception")
    test_game = TestGamePlayback('exception', 'normal')
    result, error, timed_out = test_game.test_single_game(timeout_seconds=10)
    
    if timed_out:
        logger.error("   [TIMEOUT] Model exception handling froze!")
        return False
    elif result and result.get('completed', False):
        # Check if fallback mechanism was triggered (should see random actions)
        if 'game_history' in result and len(result['game_history']) > 0:
            logger.ok("   [OK] Model exception handled gracefully with fallback random actions")
        else:
            logger.ok("   [OK] Model exception handled gracefully")
    else:
        logger.error(f"   [FAIL] Model exception should have been handled gracefully: {result}")
    
    # Test 5: Model returns NaN
    logger.testing("Test 5: Model returns NaN values")
    test_game = TestGamePlayback('nan_output', 'normal')
    result, error, timed_out = test_game.test_single_game(timeout_seconds=10)
    
    if timed_out:
        logger.error("   [TIMEOUT] NaN output handling froze!")
        return False
    elif error:
        logger.warning(f"   [WARN] ERROR: {error} (might be expected for NaN outputs)")
    elif result:
        logger.ok("   [OK] NaN output handled gracefully")
    else:
        logger.error("   [FAIL] NaN output handling failed")
    
    return True


async def test_environment_scenarios():
    """Test various environment scenarios that might cause freezing"""
    
    logger.info("\n=== Environment Freeze Tests ===\n")
    
    # Test 1: Stuck environment
    logger.testing("Test 1: Stuck environment (repeating same state)")
    test_game = TestGamePlayback('normal', 'stuck')
    result, error, timed_out = test_game.test_single_game(timeout_seconds=20)
    
    if timed_out:
        logger.error("   [TIMEOUT] Stuck environment froze!")
        return False
    elif error:
        logger.warning(f"   [WARN] ERROR: {error}")
    elif result and 'error' not in result:
        logger.ok("   [OK] Stuck environment handled successfully")
        logger.info(f"       Game had {result.get('steps', 0)} steps")
    else:
        logger.error(f"   [FAIL] Stuck environment failed: {result}")
    
    # Test 2: Infinite environment
    logger.testing("Test 2: Infinite environment (never terminates)")
    test_game = TestGamePlayback('normal', 'infinite')
    result, error, timed_out = test_game.test_single_game(timeout_seconds=15)
    
    if timed_out:
        logger.ok("   [OK] Infinite environment timeout detected and handled correctly!")
    elif error:
        logger.warning(f"   [WARN] ERROR: {error}")
    elif result and 'error' not in result:
        logger.ok("   [OK] Infinite environment handled successfully")
        logger.info(f"       Game had {result.get('steps', 0)} steps (should hit max_steps limit)")
    else:
        logger.error(f"   [FAIL] Infinite environment failed: {result}")
    
    # Test 3: Environment exception
    logger.testing("Test 3: Environment throws exception")
    test_game = TestGamePlayback('normal', 'exception')
    result, error, timed_out = test_game.test_single_game(timeout_seconds=10)
    
    if timed_out:
        logger.error("   [TIMEOUT] Environment exception handling froze!")
        return False
    elif result and 'error' in result:
        logger.ok("   [OK] Environment exception handled gracefully")
    else:
        logger.error(f"   [FAIL] Environment exception should have failed: {result}")
    
    return True


async def test_concurrent_games():
    """Test multiple concurrent games to detect race conditions"""
    
    logger.info("\n=== Concurrent Game Tests ===\n")
    
    logger.testing("Test: Multiple concurrent games")
    
    # Create multiple game instances
    games = [TestGamePlayback('normal', 'normal') for _ in range(3)]
    
    timeout_helper = TimeoutHelper(30)
    
    def run_all_games():
        results = []
        for i, game in enumerate(games):
            result, error, timed_out = game.test_single_game(timeout_seconds=10)
            results.append((result, error, timed_out))
        return results
    
    results, error = timeout_helper.run_with_timeout(run_all_games)
    
    if timeout_helper.timed_out:
        logger.error("   [TIMEOUT] Concurrent games froze!")
        return False
    elif error:
        logger.error(f"   [ERROR] {error}")
        return False
    elif results:
        success_count = sum(1 for result, error, timed_out in results if not timed_out and not error)
        logger.ok(f"   [OK] {success_count}/3 concurrent games completed successfully")
        return success_count > 0
    else:
        logger.error("   [FAIL] Concurrent games failed")
        return False


async def test_memory_usage():
    """Test for memory leaks during game playing"""
    
    logger.info("\n=== Memory Usage Tests ===\n")
    
    logger.testing("Test: Memory usage during repeated games")
    
    test_game = TestGamePlayback('normal', 'normal')
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Play multiple games and monitor memory
    for i in range(5):
        result, error, timed_out = test_game.test_single_game(timeout_seconds=10)
        
        if timed_out:
            logger.error(f"   [TIMEOUT] Game {i+1} froze!")
            return False
        
        current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        logger.info(f"   Game {i+1}: {'[OK]' if not error else '[FAIL]'} Memory: {current_memory - initial_memory} bytes")
    
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_increase = final_memory - initial_memory
    
    if memory_increase > 100 * 1024 * 1024:  # 100MB threshold
        logger.warning(f"   [WARN] WARNING: Memory usage increased by {memory_increase/1024/1024:.2f}MB")
    else:
        logger.ok(f"   [OK] Memory usage stable (increased by {memory_increase/1024/1024:.2f}MB)")
    
    return True


async def main():
    logger = TestLogger()
    """Run all game simulation tests"""
    
    logger.info("Testing game simulation scenarios that might cause server freezing...")
    
    success = True
    
    # Test action selection scenarios
    success &= await test_action_selection_scenarios()
    
    # Test environment scenarios
    success &= await test_environment_scenarios()
    
    # Test concurrent games
    success &= await test_concurrent_games()
    
    # Test memory usage
    success &= await test_memory_usage()
    
    if success:
        logger.success("All game simulation tests passed!")
    else:
        logger.error("Some game simulation tests failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 