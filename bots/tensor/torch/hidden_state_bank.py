"""GPU/CPU hidden-state ownership for batched tensor collection."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class HiddenStateBatch:
    row_index: torch.Tensor
    h_prev: torch.Tensor


class HiddenStateBank:
    """Own recurrent hidden state for active (game, unit) pairs."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        initial_capacity: int = 4096,
        growth_capacity: int = 4096,
    ):
        if initial_capacity <= 0:
            raise ValueError(f"initial_capacity must be > 0, got {initial_capacity}")
        if growth_capacity <= 0:
            raise ValueError(f"growth_capacity must be > 0, got {growth_capacity}")
        self.hidden_dim = int(hidden_dim)
        self.device = device
        self.dtype = dtype
        self._growth_capacity = int(growth_capacity)
        self._hidden = torch.zeros(initial_capacity, hidden_dim, device=device, dtype=dtype)
        self._rows_by_game: dict[int, dict[int, int]] = {}
        self._free_rows: list[int] = []
        self._next_row = 0

    @property
    def capacity(self) -> int:
        return int(self._hidden.shape[0])

    def gather_into(self, keys: list[tuple[int, int]], out: torch.Tensor) -> torch.Tensor:
        """Fill a preallocated device buffer with hidden rows for the given keys."""
        if out.device.type != self.device.type:
            raise ValueError(f"out must live on {self.device}, got {out.device}")
        if self.device.index is not None and out.device.index != self.device.index:
            raise ValueError(f"out must live on {self.device}, got {out.device}")
        if out.shape != (len(keys), self.hidden_dim):
            raise ValueError(
                f"out must have shape ({len(keys)}, {self.hidden_dim}), got {tuple(out.shape)}"
            )
        if out.dtype != self.dtype:
            raise ValueError(f"out must have dtype {self.dtype}, got {out.dtype}")
        row_index = self._row_index(keys)
        if row_index.numel() != 0:
            out.copy_(self._hidden[row_index])
        return row_index

    def batch(self, keys: list[tuple[int, int]]) -> HiddenStateBatch:
        row_index = self._row_index(keys)
        if row_index.numel() == 0:
            empty_hidden = torch.empty(0, self.hidden_dim, dtype=self.dtype, device=self.device)
            return HiddenStateBatch(row_index=row_index, h_prev=empty_hidden)
        return HiddenStateBatch(
            row_index=row_index,
            h_prev=self._hidden[row_index],
        )

    def update(self, row_index: torch.Tensor, h_next: torch.Tensor) -> None:
        if row_index.numel() == 0:
            return
        if row_index.device.type != "cpu":
            row_index = row_index.cpu()
        if h_next.device != self._hidden.device:
            h_next = h_next.to(self._hidden.device)
        if h_next.shape != (row_index.shape[0], self.hidden_dim):
            raise ValueError(
                f"h_next must have shape ({row_index.shape[0]}, {self.hidden_dim}), got {tuple(h_next.shape)}"
            )
        self._hidden[row_index] = h_next

    def release_game(self, game_idx: int) -> None:
        game_rows = self._rows_by_game.pop(game_idx, None)
        if not game_rows:
            return
        rows = list(game_rows.values())
        row_index = torch.tensor(rows, dtype=torch.long)
        self._hidden[row_index] = 0.0
        self._free_rows.extend(rows)

    def _ensure_row(self, game_idx: int, unit_id: int) -> int:
        game_rows = self._rows_by_game.setdefault(game_idx, {})
        row = game_rows.get(unit_id)
        if row is not None:
            return row
        row = self._allocate_row()
        game_rows[unit_id] = row
        return row

    def _row_index(self, keys: list[tuple[int, int]]) -> torch.Tensor:
        if not keys:
            return torch.empty(0, dtype=torch.long)
        row_indices = [self._ensure_row(game_idx, unit_id) for game_idx, unit_id in keys]
        return torch.tensor(row_indices, dtype=torch.long)

    def _allocate_row(self) -> int:
        if self._free_rows:
            return self._free_rows.pop()
        if self._next_row >= self.capacity:
            self._grow()
        row = self._next_row
        self._next_row += 1
        return row

    def _grow(self) -> None:
        growth = max(self._growth_capacity, self.capacity)
        new_hidden = torch.zeros(self.capacity + growth, self.hidden_dim, device=self.device, dtype=self.dtype)
        new_hidden[: self.capacity].copy_(self._hidden)
        self._hidden = new_hidden
