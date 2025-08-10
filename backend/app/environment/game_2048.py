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
import math

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
        """Reset board to initial state and return the new board.

        Performance: avoid deep-copy here. The Gym wrapper converts this to a
        NumPy array, which already materializes a separate copy for callers.
        """
        if seed is not None:
            self._rng.seed(seed)
        self.board = [[0] * SIZE for _ in range(SIZE)]
        self.score = 0
        self.done = False
        self._spawn_tile()
        self._spawn_tile()
        return self.board

    def step(self, action: int) -> Tuple[Board, int, bool]:
        """Apply *action* (0=up,1=down,2=left,3=right).

        Returns ``(next_board, reward, done)``.
        """
        if self.done:
            return self.board, 0, True

        moved, reward = self._move(action)
        if moved:
            self._spawn_tile()
            self.score += reward
            if not self._has_moves():
                self.done = True
        # If move is invalid, reward is 0 and board unchanged.
        return self.board, reward, self.done

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
            enc = _encode_row_values(original)
            if left:
                new_enc = _ROW_LEFT_NEW[enc]
                row_reward = _ROW_LEFT_REWARD[enc]
                did_move = _ROW_LEFT_MOVED[enc]
            else:
                rev = _reverse_row_enc(enc)
                new_enc = _reverse_row_enc(_ROW_LEFT_NEW[rev])
                row_reward = _ROW_LEFT_REWARD[rev]
                did_move = _ROW_LEFT_MOVED[rev]
            if did_move:
                moved = True
            self.board[r] = _decode_row_values(new_enc)
            reward += row_reward
        return moved, reward

    def _move_vertical(self, *, up: bool) -> Tuple[bool, int]:
        moved = False
        reward = 0
        for c in range(SIZE):
            col = [self.board[r][c] for r in range(SIZE)]
            enc = _encode_row_values(col)
            if up:
                new_enc = _ROW_LEFT_NEW[enc]
                col_reward = _ROW_LEFT_REWARD[enc]
                did_move = _ROW_LEFT_MOVED[enc]
            else:
                rev = _reverse_row_enc(enc)
                new_enc = _reverse_row_enc(_ROW_LEFT_NEW[rev])
                col_reward = _ROW_LEFT_REWARD[rev]
                did_move = _ROW_LEFT_MOVED[rev]
            if did_move:
                moved = True
            new_col = _decode_row_values(new_enc)
            for r in range(SIZE):
                self.board[r][c] = new_col[r]
            reward += col_reward
        return moved, reward

    # ---- game over detection -------------------------------------------------
    def _can_move(self, action: int) -> bool:
        """Fast check using precomputed row transitions.

        Encodes rows/columns and checks if a move in the given direction would change any row/col.
        """
        b = self.board
        if action == 2:  # left
            for r in range(SIZE):
                enc = _encode_row_values(b[r])
                if _ROW_LEFT_MOVED[enc]:
                    return True
            return False
        if action == 3:  # right
            for r in range(SIZE):
                enc = _reverse_row_enc(_encode_row_values(b[r]))
                if _ROW_LEFT_MOVED[enc]:
                    return True
            return False
        if action == 0:  # up
            for c in range(SIZE):
                col = [b[r][c] for r in range(SIZE)]
                enc = _encode_row_values(col)
                if _ROW_LEFT_MOVED[enc]:
                    return True
            return False
        if action == 1:  # down
            for c in range(SIZE):
                col = [b[r][c] for r in range(SIZE)]
                enc = _reverse_row_enc(_encode_row_values(col))
                if _ROW_LEFT_MOVED[enc]:
                    return True
            return False
        return False

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

# ------------------------------ Fast row transition tables ------------------------------

# We encode a 4-length row of tile values as a 16-bit key by mapping values to exponents
# (0 -> 0, 2 -> 1, 4 -> 2, ... up to a reasonable cap), storing 4 nibbles.

_VAL_TO_EXP = {0: 0}
for e in range(1, 16):
    _VAL_TO_EXP[1 << e] = e

def _to_exp(v: int) -> int:
    if v in _VAL_TO_EXP:
        return _VAL_TO_EXP[v]
    if v <= 0:
        return 0
    # Fallback: integer log2
    return int(math.log2(v))

def _encode_row_values(vals: Sequence[int]) -> int:
    # vals length is 4
    return ((_to_exp(vals[0]) & 0xF) << 12) | ((_to_exp(vals[1]) & 0xF) << 8) | ((_to_exp(vals[2]) & 0xF) << 4) | (_to_exp(vals[3]) & 0xF)

def _decode_row_values(code: int) -> List[int]:
    # Convert exponents back to values
    return [0 if ((code >> shift) & 0xF) == 0 else (1 << ((code >> shift) & 0xF)) for shift in (12, 8, 4, 0)]

def _reverse_row_enc(code: int) -> int:
    a = (code >> 12) & 0xF
    b = (code >> 8) & 0xF
    c = (code >> 4) & 0xF
    d = code & 0xF
    return (d << 12) | (c << 8) | (b << 4) | a

# Precompute transitions for all 16^4 = 65536 possible encoded rows.
_ROW_LEFT_NEW = [0] * 65536
_ROW_LEFT_REWARD = [0] * 65536
_ROW_LEFT_MOVED = [False] * 65536

def _decode_exp_row(code: int) -> List[int]:
    return [(code >> s) & 0xF for s in (12, 8, 4, 0)]

def _compress_merge_exp_row(exps: List[int]) -> Tuple[List[int], int]:
    # Slide non-zeros, merge equal, slide again — in exponent space (0 denotes empty)
    new = [e for e in exps if e != 0]
    reward = 0
    i = 0
    while i < len(new) - 1:
        if new[i] != 0 and new[i] == new[i + 1]:
            new[i] += 1
            reward += (1 << new[i])
            del new[i + 1]
            i += 1
        else:
            i += 1
    while len(new) < 4:
        new.append(0)
    return new, reward

for code in range(65536):
    exps = _decode_exp_row(code)
    new_exps, rew = _compress_merge_exp_row(exps)
    new_code = ((new_exps[0] & 0xF) << 12) | ((new_exps[1] & 0xF) << 8) | ((new_exps[2] & 0xF) << 4) | (new_exps[3] & 0xF)
    _ROW_LEFT_NEW[code] = new_code
    _ROW_LEFT_REWARD[code] = rew
    _ROW_LEFT_MOVED[code] = (new_code != code)