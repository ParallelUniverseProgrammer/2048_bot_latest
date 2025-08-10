"""
GPU-batched 2048 environment implemented with PyTorch tensors.

Features:
- Maintains a batch of 4x4 boards on device
- Vectorized legal move detection and step for 4 directions
- 2048 merge rules with left-to-right priority
- Random tile spawn (2 with 90%, 4 with 10%) after valid moves
- Basic reward shaping ported from gym wrapper (empty cells, smoothness, alive/terminal)

Note: This is optimized for small 4x4 boards; code favors clarity with
vectorized operations and avoids CPU transfers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class GPU2048BatchEnv:
    def __init__(self, batch_size: int, device: torch.device, *, seed: Optional[int] = None):
        self.batch_size = int(batch_size)
        self.device = device
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(int(seed))
        else:
            # Best-effort seed
            self.generator.seed()

        self.boards = torch.zeros((self.batch_size, 4, 4), dtype=torch.int32, device=device)
        self.done = torch.zeros((self.batch_size,), dtype=torch.bool, device=device)
        self.scores = torch.zeros((self.batch_size,), dtype=torch.int64, device=device)

        self.reset()

    @torch.no_grad()
    def reset(self) -> torch.Tensor:
        self.boards.zero_()
        self.done.zero_()
        self.scores.zero_()
        # spawn two tiles per board
        self._spawn_random_tiles(self.boards, count=2)
        return self.boards.clone()

    # ------------------------------ Core ops ------------------------------
    @staticmethod
    def _transpose_for_direction(boards: torch.Tensor, direction: int) -> torch.Tensor:
        """Return a view of boards such that a LEFT move in the returned
        coordinates corresponds to the requested direction on the original board.
        Directions: 0=up,1=down,2=left,3=right"""
        if direction == 2:  # left
            return boards
        if direction == 3:  # right
            return torch.flip(boards, dims=(2,))
        if direction == 0:  # up
            return boards.transpose(1, 2)
        if direction == 1:  # down
            return torch.flip(boards.transpose(1, 2), dims=(2,))
        raise ValueError("Invalid direction")

    @staticmethod
    def _inverse_transform(boards_view: torch.Tensor, direction: int) -> torch.Tensor:
        if direction == 2:  # left
            return boards_view
        if direction == 3:  # right
            return torch.flip(boards_view, dims=(2,))
        if direction == 0:  # up
            return boards_view.transpose(1, 2)
        if direction == 1:  # down
            return torch.flip(boards_view, dims=(2,)).transpose(1, 2)
        raise ValueError("Invalid direction")

    @staticmethod
    def _compress_left(arr: torch.Tensor) -> torch.Tensor:
        """Stable compress non-zero values to the left for shape (B,4,4)."""
        B = arr.shape[0]
        out = torch.zeros_like(arr)
        # process each row independently using cumsum trick
        mask = (arr != 0).to(torch.int64)
        # positions to place each non-zero: cumsum-1
        pos = (torch.cumsum(mask, dim=2) - 1).clamp(min=0)
        # Broadcast batch and row indices
        batch_idx = torch.arange(B, device=arr.device).view(B, 1, 1).expand_as(arr)
        row_idx = torch.arange(4, device=arr.device).view(1, 4, 1).expand_as(arr)
        nz = mask.bool()
        # Gather target indices
        tgt_col = pos[nz]
        tgt_b = batch_idx[nz]
        tgt_r = row_idx[nz]
        out[tgt_b, tgt_r, tgt_col] = arr[nz]
        return out

    @staticmethod
    def _merge_left(arr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge equal adjacent tiles from left; returns (merged_arr, merge_reward).
        Reward equals sum of newly created tile values per board.
        Shapes: arr (B,4,4) int32, returns arr int32 and reward int32 (B,)"""
        B = arr.shape[0]
        reward = torch.zeros((B,), dtype=torch.int32, device=arr.device)
        # Compare columns 0-2 with 1-3 on each row
        left = arr[:, :, 0:3]
        right = arr[:, :, 1:4]
        eq = (left == right) & (left != 0)
        # Double left where eq, zero right
        doubled = left * 2
        # For reward: sum of doubled values at eq positions
        reward += (doubled * eq).sum(dim=(1, 2)).to(torch.int32)
        left = torch.where(eq, doubled, left)
        right = torch.where(eq, torch.zeros_like(right), right)
        merged = torch.cat([left, right[:, :, -1:].clone()*0], dim=2)
        return merged, reward

    @torch.no_grad()
    def _move_left_and_reward(self, boards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a left move on provided boards; returns (new_boards, changed_mask, rewards)."""
        before = boards
        compressed = self._compress_left(before)
        merged, merge_reward = self._merge_left(compressed)
        compressed2 = self._compress_left(merged)
        changed = (compressed2 != before).any(dim=(1, 2))
        return compressed2, changed, merge_reward

    @torch.no_grad()
    def _spawn_random_tiles(self, boards: torch.Tensor, count: int = 1) -> None:
        B = boards.shape[0]
        for _ in range(count):
            empties = (boards == 0).view(B, -1).to(torch.float32)
            # If all full for a board, keep as-is
            has_empty = empties.sum(dim=1) > 0
            if not has_empty.any():
                return
            # Normalize probabilities across empty cells
            probs = torch.where(empties > 0, empties, torch.zeros_like(empties))
            sums = probs.sum(dim=1, keepdim=True)
            # Guard division by zero
            probs = torch.where(sums > 0, probs / sums.clamp(min=1e-8), probs)
            # Sample position for each batch
            # torch.multinomial expects sum(probs) > 0
            idx = torch.multinomial(probs, num_samples=1, replacement=True, generator=self.generator).squeeze(1)
            # tile value: 2 (90%) or 4 (10%)
            two_or_four = torch.where(
                torch.rand((B,), device=boards.device, generator=self.generator) < 0.9,
                torch.tensor(2, dtype=torch.int32, device=boards.device),
                torch.tensor(4, dtype=torch.int32, device=boards.device),
            )
            # Scatter into boards
            r = (idx // 4).to(torch.long)
            c = (idx % 4).to(torch.long)
            b = torch.arange(B, device=boards.device, dtype=torch.long)
            # Only assign for boards with empties
            boards[b[has_empty], r[has_empty], c[has_empty]] = two_or_four[has_empty]

    # ------------------------------ Public API ------------------------------
    @torch.no_grad()
    def legal_moves_mask(self, boards: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return mask (B,4) of legal actions for each board.
        An action is legal if after applying it the board changes and the game is not done."""
        if boards is None:
            boards = self.boards
        B = boards.shape[0]
        mask = torch.zeros((B, 4), dtype=torch.bool, device=boards.device)
        for d in range(4):
            v = self._transpose_for_direction(boards, d)
            next_v, changed, _ = self._move_left_and_reward(v)
            # Inverse transform to compare; but changed computed already
            mask[:, d] = changed & (~self.done)
        return mask

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step with actions (B,) int64; returns (next_boards, reward, done).
        Reward is log2 scaled + small shaping similar to gym wrapper.
        """
        actions = actions.to(torch.long)
        B = self.boards.shape[0]
        # Pre-move features for shaping
        empty_before = (self.boards == 0).sum(dim=(1, 2)).to(torch.float32) / 16.0
        # Smoothness: sum abs differences of adjacent log2 values
        log_board = torch.zeros_like(self.boards, dtype=torch.float32)
        nz = self.boards > 0
        log_board[nz] = torch.log2(self.boards[nz].to(torch.float32))
        smooth_h = (log_board[:, :, :-1] - log_board[:, :, 1:]).abs().sum(dim=(1, 2))
        smooth_v = (log_board[:, :-1, :] - log_board[:, 1:, :]).abs().sum(dim=(1, 2))
        smooth_before = smooth_h + smooth_v

        # Apply moves per unique action by masking
        new_boards = self.boards.clone()
        merge_reward_total = torch.zeros((B,), dtype=torch.int32, device=self.device)
        for d in range(4):
            idx = (actions == d)
            if not idx.any():
                continue
            view = self._transpose_for_direction(self.boards[idx], d)
            moved, changed, merge_rew = self._move_left_and_reward(view)
            merge_reward_total[idx] = merge_rew
            moved = self._inverse_transform(moved, d)
            new_boards[idx] = moved

        # Determine which boards actually changed and are not done
        changed_mask = (new_boards != self.boards).any(dim=(1, 2)) & (~self.done)

        # Spawn tiles where changes occurred
        if changed_mask.any():
            self._spawn_random_tiles(new_boards[changed_mask], count=1)

        # Compute terminal: no legal moves on new board
        self.boards = new_boards
        legal = self.legal_moves_mask(self.boards)
        no_moves = ~(legal.any(dim=1))
        self.done = self.done | no_moves

        # Score and rewards
        # Base reward: log2 scaling of merge sum
        reward_raw = merge_reward_total.to(torch.float32)
        reward = torch.log2(reward_raw + 1.0)

        # Shaping similar to gym wrapper
        empty_after = (self.boards == 0).sum(dim=(1, 2)).to(torch.float32) / 16.0
        log_board = torch.zeros_like(self.boards, dtype=torch.float32)
        nz = self.boards > 0
        log_board[nz] = torch.log2(self.boards[nz].to(torch.float32))
        smooth_h = (log_board[:, :, :-1] - log_board[:, :, 1:]).abs().sum(dim=(1, 2))
        smooth_v = (log_board[:, :-1, :] - log_board[:, 1:, :]).abs().sum(dim=(1, 2))
        smooth_after = smooth_h + smooth_v

        EMPTY_BONUS_COEF = 0.05
        SMOOTH_BONUS_COEF = 0.0035
        ALIVE_BONUS = 0.001
        TERMINAL_PENALTY = 0.05

        empty_delta = empty_after - empty_before
        smooth_delta = smooth_before - smooth_after

        shaping = (
            EMPTY_BONUS_COEF * empty_delta +
            SMOOTH_BONUS_COEF * smooth_delta +
            torch.where(self.done, torch.zeros_like(empty_after), torch.full_like(empty_after, ALIVE_BONUS)) -
            torch.where(self.done, torch.full_like(empty_after, TERMINAL_PENALTY), torch.zeros_like(empty_after))
        )
        reward = reward + shaping

        # Accumulate score as raw merge sum (classic 2048 rule)
        self.scores += merge_reward_total.to(torch.int64)

        return self.boards.clone(), reward, self.done.clone()


