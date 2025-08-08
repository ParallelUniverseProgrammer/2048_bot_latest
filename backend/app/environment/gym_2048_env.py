"""gym_2048_env.py
===================
Gymnasium-compatible wrapper for the 2048 game implemented in
`game_2048.Game2048`.

Why not use an existing library?
--------------------------------
Most Gym 2048 environments on PyPI are **stuck on the old `gym` API**, rely on
`pygame`, or add features we don't need.  A tiny, dependency-free wrapper keeps
our stack lightweight and easy to maintain.

Observation Space
-----------------
* **Shape**: ``(4, 4)``
* **dtype**: ``int32`` – raw tile values (0,2,4,…)

You can transform this further in the model (e.g. log2 one-hot encoding).

Reward Function
---------------
We use the *incremental score gain* (sum of merged tile values) that 2048 uses
internally.  This is the standard baseline in the RL literature.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces

from .game_2048 import Game2048, SIZE


class Gym2048Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.game = Game2048(seed)
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        # raw board values 0..2**15 fits in int32
        self.observation_space = spaces.Box(low=0, high=65536, shape=(SIZE, SIZE), dtype=np.int32)
        self._seed = seed

    # ------------------------------------------------------------------ Gym API
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        board = self.game.reset(seed or self._seed)
        return np.array(board, dtype=np.int32), {}

    def step(self, action: int):  # type: ignore[override]
        board, reward_raw, done = self.game.step(action)
        # Reward shaping: log2 scaling for PPO stability
        reward = math.log2(reward_raw + 1)
        # Avoid extra array allocations by reusing buffer when possible
        obs = np.asarray(board, dtype=np.int32)
        return obs, float(reward), done, False, {}

    # ------------------------------------------------------------------ Render
    def render(self):  # simple console render for debugging
        for row in self.game.board:
            print("|".join(f"{v:4d}" if v else "    " for v in row))
        print(f"Score: {self.game.score}\n")

    def close(self):
        pass
    
    # ------------------------------------------------------------------ Additional methods for playback
    def is_done(self) -> bool:
        """Check if the game is over"""
        return self.game.done
    
    def get_state(self) -> np.ndarray:
        """Get current board state as numpy array"""
        return np.array(self.game.board, dtype=np.int32)
    
    def get_legal_actions(self) -> list[int]:
        """Get list of legal actions from current state"""
        return self.game.legal_moves()
    
    def get_score(self) -> int:
        """Get current game score"""
        return self.game.score 