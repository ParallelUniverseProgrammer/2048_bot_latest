"""
2048 Game Logic for Training Environment
=======================================
This module implements the core 2048 board game mechanics. It is a pure-Python
implementation with **no third-party dependencies**, making it easy to test and
reuse from both RL environments and any other simulation code.

Key Features
------------
1. Deterministic, unit-test friendly logic (no hidden random state).
2. Helper methods for **move**/​**spawn**/**score** as well as board utilities.
3. Minimal public API so we can later swap this file with a more optimised C++
   backend without touching the training loop.

The class exposes:
    * ``reset(seed:int|None=None)`` – returns initial board.
    * ``step(action:int)`` – performs a move & spawns a tile. Returns the tuple
      ``next_board, reward, done``.
    * ``legal_moves()`` – bit-mask of valid moves for early termination.

The internal representation uses a ``list[list[int]]`` holding the **raw tile
values** (e.g. ``2,4,8`` …) so we never lose information when converting to
log-space etc.  All helper functions are `_static` for fast testability.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import List, Tuple, Sequence

# Type alias for clarity
Board = List[List[int]]

# Constants
SIZE = 4  # 4x4 board
NEW_TILE_CHOICES = [2] * 9 + [4]  # 90% chance of 2, 10% of 4

__all__ = [
    "Board",
    "SIZE",
    "Game2048",
]


class Game2048:
    """Standalone 2048 game engine."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self.board: Board = [[0] * SIZE for _ in range(SIZE)]
        self.score: int = 0
        self.done: bool = False
        # initial tiles
        self._spawn_tile()
        self._spawn_tile()

    # ----------------------------------------------------- Public API
    def reset(self, seed: int | None = None) -> Board:
        """Reset board to initial state and return the new board."""
        if seed is not None:
            self._rng.seed(seed)
        self.board = [[0] * SIZE for _ in range(SIZE)]
        self.score = 0
        self.done = False
        self._spawn_tile()
        self._spawn_tile()
        return deepcopy(self.board)

    def step(self, action: int) -> Tuple[Board, int, bool]:
        """Apply *action* (0=up,1=down,2=left,3=right).

        Returns ``(next_board, reward, done)``.
        """
        if self.done:
            return deepcopy(self.board), 0, True

        moved, reward = self._move(action)
        if moved:
            self._spawn_tile()
            self.score += reward
            if not self._has_moves():
                self.done = True
        # If move is invalid, reward is 0 and board unchanged.
        return deepcopy(self.board), reward, self.done

    def legal_moves(self) -> List[int]:
        """Return list of **valid** actions from the current state."""
        moves = []
        for a in range(4):
            if self._can_move(a):
                moves.append(a)
        return moves

    # ----------------------------------------------------- Internal helpers
    def _spawn_tile(self) -> None:
        """Place a new tile (2 / 4) on a random empty cell."""
        empties = [(r, c) for r in range(SIZE) for c in range(SIZE) if self.board[r][c] == 0]
        if not empties:
            return
        r, c = self._rng.choice(empties)
        self.board[r][c] = self._rng.choice(NEW_TILE_CHOICES)

    # ---- movement primitives ------------------------------------------------
    def _move(self, action: int) -> Tuple[bool, int]:
        """Perform move; return (moved?, reward_from_merges)."""
        if action == 0:  # up
            return self._move_vertical(up=True)
        if action == 1:  # down
            return self._move_vertical(up=False)
        if action == 2:  # left
            return self._move_horizontal(left=True)
        if action == 3:  # right
            return self._move_horizontal(left=False)
        raise ValueError(f"Invalid action: {action}")

    def _move_horizontal(self, *, left: bool) -> Tuple[bool, int]:
        moved = False
        reward = 0
        for r in range(SIZE):
            original = self.board[r]
            new_row, row_reward = self._compress_and_merge(original if left else original[::-1])
            if not left:
                new_row = new_row[::-1]
            if new_row != original:
                moved = True
            self.board[r] = new_row
            reward += row_reward
        return moved, reward

    def _move_vertical(self, *, up: bool) -> Tuple[bool, int]:
        moved = False
        reward = 0
        for c in range(SIZE):
            col = [self.board[r][c] for r in range(SIZE)]
            new_col, col_reward = self._compress_and_merge(col if up else col[::-1])
            if not up:
                new_col = new_col[::-1]
            if new_col != col:
                moved = True
            for r in range(SIZE):
                self.board[r][c] = new_col[r]
            reward += col_reward
        return moved, reward

    # ---- utility ------------------------------------------------------------
    @staticmethod
    def _compress_and_merge(line: Sequence[int]) -> Tuple[List[int], int]:
        """Slide non-zeros, merge equal neighbours, slide again."""
        SIZE = len(line)
        new = [x for x in line if x != 0]
        reward = 0
        i = 0
        while i < len(new) - 1:
            if new[i] == new[i + 1]:
                new[i] *= 2
                reward += new[i]
                del new[i + 1]
                i += 1
            else:
                i += 1
        new += [0] * (SIZE - len(new))
        return new, reward

    # ---- game over detection -------------------------------------------------
    def _can_move(self, action: int) -> bool:
        temp = deepcopy(self)
        moved, _ = temp._move(action)
        return moved

    def _has_moves(self) -> bool:
        for a in range(4):
            if self._can_move(a):
                return True
        return False
    
    def get_max_tile(self) -> int:
        """Get the maximum tile value on the board"""
        max_tile = 0
        for row in self.board:
            for tile in row:
                if tile > max_tile:
                    max_tile = tile
        return max_tile 