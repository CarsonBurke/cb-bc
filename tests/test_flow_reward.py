"""Tests for retroactive conveyor build credit assignment."""

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bots" / "tensor" / "torch"))

from flow_reward import (
    compute_retroactive_build_rewards,
    augment_trajectories_with_build_rewards,
)

try:
    from model import _parse_trajectories, REWARD_BUILD_CREDIT
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Helpers ──────────────────────────────────────────────────────────────


def _pack_triples(triples: list[tuple[int, int, int]]) -> bytes:
    """Pack list of (turn, x, y) into flat i32 buffer."""
    return b"".join(struct.pack("<iii", t, x, y) for t, x, y in triples)


def _make_step(round_num: int, n_working: float = 0.0) -> list:
    """Create a minimal long-format trajectory step for the given round."""
    obs = [0.0] * 490  # 475 grid + 15 globals
    obs[475] = round_num / 2000.0  # round_norm
    # Long format: obs, h, bt, bd, cd, md, value, n_working, conv_r, ti, harv_r, dmg, old_lp
    return [obs, [0.0] * 24, 0, 0, 0, 0, 0.0, n_working, 0, 0, 0, 0, 0]


# ── compute_retroactive_build_rewards ────────────────────────────────────


class TestComputeRetroactiveBuildRewards:
    def test_empty_buffers(self):
        assert compute_retroactive_build_rewards({}) == {}
        assert compute_retroactive_build_rewards({"n_turns": 100}) == {}

    def test_births_no_transfers(self):
        gs = {
            "conveyor_births_buf": _pack_triples([(5, 10, 10)]),
            "tile_transfers_buf": b"",
            "n_turns": 100,
        }
        assert compute_retroactive_build_rewards(gs) == {}

    def test_single_conveyor_lifetime(self):
        """One conveyor built at turn 5, transfers at turns 10, 20, 30."""
        gs = {
            "conveyor_births_buf": _pack_triples([(5, 3, 4)]),
            "tile_transfers_buf": _pack_triples([(10, 3, 4), (20, 3, 4), (30, 3, 4)]),
            "n_turns": 100,
        }
        rewards = compute_retroactive_build_rewards(gs)
        assert rewards == {5: 3.0}

    def test_rebuild_same_tile(self):
        """Conveyor built turn 5, rebuilt turn 20. Transfers at 10, 15, 25."""
        gs = {
            "conveyor_births_buf": _pack_triples([(5, 3, 4), (20, 3, 4)]),
            "tile_transfers_buf": _pack_triples([(10, 3, 4), (15, 3, 4), (25, 3, 4)]),
            "n_turns": 100,
        }
        rewards = compute_retroactive_build_rewards(gs)
        # First conveyor: turns [5, 20) → transfers at 10, 15 → 2
        # Second conveyor: turns [20, 100) → transfer at 25 → 1
        assert rewards == {5: 2.0, 20: 1.0}

    def test_multiple_tiles_same_build_round(self):
        """Two conveyors built on the same round at different positions."""
        gs = {
            "conveyor_births_buf": _pack_triples([(10, 1, 1), (10, 2, 2)]),
            "tile_transfers_buf": _pack_triples([
                (15, 1, 1), (20, 1, 1),  # tile (1,1): 2 transfers
                (15, 2, 2),               # tile (2,2): 1 transfer
            ]),
            "n_turns": 100,
        }
        rewards = compute_retroactive_build_rewards(gs)
        assert rewards == {10: 3.0}  # summed across both tiles

    def test_transfers_before_birth_ignored(self):
        """Transfers before the conveyor was built are not counted."""
        gs = {
            "conveyor_births_buf": _pack_triples([(10, 5, 5)]),
            "tile_transfers_buf": _pack_triples([(3, 5, 5), (7, 5, 5), (15, 5, 5)]),
            "n_turns": 100,
        }
        rewards = compute_retroactive_build_rewards(gs)
        assert rewards == {10: 1.0}  # only transfer at turn 15

    def test_transfer_on_birth_turn_counted(self):
        """A transfer on the exact birth turn should be counted."""
        gs = {
            "conveyor_births_buf": _pack_triples([(10, 5, 5)]),
            "tile_transfers_buf": _pack_triples([(10, 5, 5)]),
            "n_turns": 100,
        }
        rewards = compute_retroactive_build_rewards(gs)
        assert rewards == {10: 1.0}

    def test_transfers_at_different_tile_not_attributed(self):
        """Transfers at a different tile are not attributed to this conveyor."""
        gs = {
            "conveyor_births_buf": _pack_triples([(5, 3, 4)]),
            "tile_transfers_buf": _pack_triples([(10, 9, 9), (20, 9, 9)]),
            "n_turns": 100,
        }
        assert compute_retroactive_build_rewards(gs) == {}

    def test_destroyed_and_never_rebuilt(self):
        """Conveyor built turn 5, destroyed turn 15 (no rebuild).
        Transfers at 10 and 20 — only 10 should count since 20 is after
        the implicit death (but we don't track destruction, only rebirths,
        so 20 IS counted as the conveyor's lifetime extends to n_turns)."""
        gs = {
            "conveyor_births_buf": _pack_triples([(5, 3, 4)]),
            "tile_transfers_buf": _pack_triples([(10, 3, 4), (20, 3, 4)]),
            "n_turns": 100,
        }
        rewards = compute_retroactive_build_rewards(gs)
        # Without explicit death tracking, lifetime is [5, n_turns)
        assert rewards == {5: 2.0}


# ── augment_trajectories_with_build_rewards ──────────────────────────────


class TestAugmentTrajectories:
    def test_zeroes_stale_n_working(self):
        """Stale n_working values are zeroed when no rewards exist."""
        step = _make_step(round_num=50, n_working=999.0)
        traj = [[step]]
        augment_trajectories_with_build_rewards(traj, {})
        assert step[7] == 0.0

    def test_writes_reward_to_correct_round(self):
        step10 = _make_step(round_num=10)
        step20 = _make_step(round_num=20)
        traj = [[step10, step20]]
        augment_trajectories_with_build_rewards(traj, {10: 5.0, 20: 3.0})
        assert step10[7] == 5.0
        assert step20[7] == 3.0

    def test_unmatched_rounds_get_zero(self):
        step15 = _make_step(round_num=15)
        traj = [[step15]]
        augment_trajectories_with_build_rewards(traj, {10: 5.0})
        assert step15[7] == 0.0

    def test_converts_tuples_to_lists(self):
        """Tuple steps should be converted to lists for mutation."""
        step = tuple(_make_step(round_num=10))
        traj = [[step]]
        augment_trajectories_with_build_rewards(traj, {10: 7.0})
        assert isinstance(traj[0][0], list)
        assert traj[0][0][7] == 7.0

    def test_short_format_uses_slot_6(self):
        """Short-format steps (<=12 elements) use slot 6 for n_working."""
        obs = [0.0] * 490
        obs[475] = 10 / 2000.0
        step = [obs, [0.0] * 24, 0, 0, 0, 0, 999.0, 0, 0, 0, 0, 0]  # 12 elements
        traj = [[step]]
        augment_trajectories_with_build_rewards(traj, {10: 4.0})
        assert step[6] == 4.0

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    def test_shared_round_reward_is_not_multiplied_by_unit_count(self):
        step_a = _make_step(round_num=10)
        step_b = _make_step(round_num=10)
        trajs = [[step_a], [step_b]]
        augment_trajectories_with_build_rewards(trajs, {10: 3.0})
        data = _parse_trajectories(trajs, traj_game_map=[0, 0])
        assert data is not None
        assert data["rewards"].tolist() == [3.0 * REWARD_BUILD_CREDIT, 3.0 * REWARD_BUILD_CREDIT]


# ── Integration: replay parser → build rewards ──────────────────────────


@pytest.fixture
def replay_path():
    """Find a replay file generated from a real game."""
    candidates = list(Path(".").glob("*.replay26")) + list(Path("replays").glob("*.replay26"))
    if not candidates:
        pytest.skip("No replay files available")
    return str(candidates[0])


class TestIntegration:
    def test_replay_round_trip(self, replay_path):
        """Parse a real replay, compute rewards, augment trajectories."""
        from replay_parser import ReplayData

        rd = ReplayData.from_file(replay_path, team=0)
        gs = {
            "conveyor_births_buf": rd.conveyor_births_bytes,
            "tile_transfers_buf": rd.tile_transfers_bytes,
            "n_turns": rd.n_turns,
        }
        rewards = compute_retroactive_build_rewards(gs)

        # Rewards should be non-negative
        for rnd, val in rewards.items():
            assert val >= 0.0, f"Negative reward at round {rnd}"
            assert 0 <= rnd < rd.n_turns, f"Round {rnd} out of range"

        # Augmenting should not crash
        steps = [_make_step(r) for r in range(0, min(rd.n_turns, 50))]
        traj = [steps]
        augment_trajectories_with_build_rewards(traj, rewards)

        # All n_working slots should be floats
        for s in steps:
            assert isinstance(s[7], (int, float))

    def test_births_and_transfers_are_valid_triples(self, replay_path):
        """Buffer lengths must be multiples of 12 (3 × i32)."""
        from replay_parser import ReplayData

        rd = ReplayData.from_file(replay_path, team=0)
        assert len(rd.conveyor_births_bytes) % 12 == 0
        assert len(rd.tile_transfers_bytes) % 12 == 0


# ── Simulated supply chain scenarios ─────────────────────────────────────


class SupplyChainBuilder:
    """Helper to build realistic supply chain scenarios.

    Simulates a grid where entities are placed and resources flow each round.
    Produces the same (conveyor_births, tile_transfers) buffers the Rust
    parser would emit, so we can test the Python reward logic in isolation.

    Coordinate system: (x, y) with core at some position. The hop filter
    (dst closer to core than src) is simulated by only generating transfers
    for tiles whose downstream neighbor is closer to core.
    """

    def __init__(self, n_turns: int, core_xy: tuple[int, int] = (0, 0)):
        self.n_turns = n_turns
        self.core_xy = core_xy
        # (turn, x, y) for each placement
        self.births: list[tuple[int, int, int]] = []
        # entity state: (x, y) -> {alive_from, alive_to, kind, output_dir}
        self.entities: dict[tuple[int, int], list[dict]] = {}
        # generated transfers
        self.transfers: list[tuple[int, int, int]] = []

    def _dist_to_core(self, x: int, y: int) -> int:
        return abs(x - self.core_xy[0]) + abs(y - self.core_xy[1])

    def place(self, turn: int, x: int, y: int, dx: int = 0, dy: int = -1):
        """Place a conveyor-type entity at (x, y) on the given turn."""
        self.births.append((turn, x, y))
        entry = {"alive_from": turn, "alive_to": self.n_turns, "dx": dx, "dy": dy}
        self.entities.setdefault((x, y), []).append(entry)
        return self

    def destroy(self, turn: int, x: int, y: int):
        """Destroy the most recent entity at (x, y)."""
        entries = self.entities.get((x, y), [])
        for e in reversed(entries):
            if e["alive_from"] <= turn < e["alive_to"]:
                e["alive_to"] = turn
                return self
        raise ValueError(f"No alive entity at ({x}, {y}) on turn {turn}")

    def run_transfers(self, start: int | None = None, end: int | None = None, interval: int = 1):
        """Simulate resource flow: each alive entity transfers if its output
        is closer to core (hop filter). Generates transfer events."""
        start = start or 0
        end = end or self.n_turns
        for t in range(start, end, interval):
            for (x, y), entries in self.entities.items():
                for e in entries:
                    if e["alive_from"] <= t < e["alive_to"]:
                        out_x, out_y = x + e["dx"], y + e["dy"]
                        if self._dist_to_core(out_x, out_y) < self._dist_to_core(x, y):
                            self.transfers.append((t, x, y))
        return self

    def build(self) -> dict:
        return {
            "conveyor_births_buf": _pack_triples(self.births),
            "tile_transfers_buf": _pack_triples(self.transfers),
            "n_turns": self.n_turns,
        }


class TestLinearSupplyChain:
    """Linear chain: harvester → conv → conv → conv → core.

    Core at (0, 0). Chain extends east along x-axis.
    Each conveyor outputs west (dx=-1, dy=0) toward core.

        core(0,0) ← conv(1,0) ← conv(2,0) ← conv(3,0) ← harvester(4,0)
    """

    def test_full_chain_all_tiles_rewarded(self):
        """All 4 entities transfer every round → all build rounds get credit."""
        sc = SupplyChainBuilder(n_turns=100, core_xy=(0, 0))
        # Build chain left-to-right on rounds 5-8
        sc.place(5, 4, 0, dx=-1, dy=0)  # harvester at (4,0)
        sc.place(6, 3, 0, dx=-1, dy=0)  # conv at (3,0)
        sc.place(7, 2, 0, dx=-1, dy=0)  # conv at (2,0)
        sc.place(8, 1, 0, dx=-1, dy=0)  # conv at (1,0)
        # Resources flow from turn 10 onward every round
        sc.run_transfers(start=10, end=100)

        rewards = compute_retroactive_build_rewards(sc.build())

        # Each tile transfers 90 times (turns 10-99)
        assert rewards[5] == 90.0   # harvester at (4,0)
        assert rewards[6] == 90.0   # conv at (3,0)
        assert rewards[7] == 90.0   # conv at (2,0)
        assert rewards[8] == 90.0   # conv at (1,0)

    def test_chain_built_incrementally(self):
        """Chain built over time — inner conveyors start transferring earlier."""
        sc = SupplyChainBuilder(n_turns=100, core_xy=(0, 0))
        sc.place(5, 1, 0, dx=-1, dy=0)   # closest to core, built first
        sc.place(15, 2, 0, dx=-1, dy=0)  # second
        sc.place(25, 3, 0, dx=-1, dy=0)  # third
        # Each tile transfers starting from when the CHAIN is complete enough
        # In reality transfers start when resources reach each tile.
        # Simulate: tile (1,0) transfers from turn 15, (2,0) from 25, (3,0) from 25
        for t in range(15, 100):
            sc.transfers.append((t, 1, 0))
        for t in range(25, 100):
            sc.transfers.append((t, 2, 0))
        for t in range(25, 100):
            sc.transfers.append((t, 3, 0))

        rewards = compute_retroactive_build_rewards(sc.build())

        assert rewards[5] == 85.0    # tile (1,0): 85 transfers (turns 15-99)
        assert rewards[15] == 75.0   # tile (2,0): 75 transfers (turns 25-99)
        assert rewards[25] == 75.0   # tile (3,0): 75 transfers (turns 25-99)

    def test_middle_destroyed_breaks_chain(self):
        """Destroying a middle conveyor stops its transfers.
        Upstream tiles may still transfer (resources pile up but still hop-filter pass).
        """
        sc = SupplyChainBuilder(n_turns=100, core_xy=(0, 0))
        sc.place(5, 1, 0, dx=-1, dy=0)
        sc.place(5, 2, 0, dx=-1, dy=0)
        sc.place(5, 3, 0, dx=-1, dy=0)
        # All transfer from 10-49
        for t in range(10, 50):
            for x in [1, 2, 3]:
                sc.transfers.append((t, x, 0))
        # conv at (2,0) destroyed at turn 50 — its transfers stop
        sc.destroy(50, 2, 0)
        # tiles (1,0) and (3,0) keep transferring (they still pass hop filter)
        for t in range(50, 100):
            sc.transfers.append((t, 1, 0))
            sc.transfers.append((t, 3, 0))

        rewards = compute_retroactive_build_rewards(sc.build())

        # All three built on round 5
        # (1,0): 40 + 50 = 90 transfers
        # (2,0): 40 transfers (alive [5, 50))
        # (3,0): 40 + 50 = 90 transfers
        assert rewards == {5: 90.0 + 40.0 + 90.0}

    def test_middle_destroyed_and_rebuilt(self):
        """Destroy and rebuild middle conveyor — credit splits between builds."""
        sc = SupplyChainBuilder(n_turns=100, core_xy=(0, 0))
        sc.place(5, 1, 0, dx=-1, dy=0)
        sc.place(5, 2, 0, dx=-1, dy=0)
        sc.place(5, 3, 0, dx=-1, dy=0)
        # All transfer turns 10-49
        for t in range(10, 50):
            for x in [1, 2, 3]:
                sc.transfers.append((t, x, 0))
        # Destroy and rebuild (2,0) at turn 50
        sc.destroy(50, 2, 0)
        sc.place(55, 2, 0, dx=-1, dy=0)
        # All resume at turn 60
        for t in range(60, 100):
            for x in [1, 2, 3]:
                sc.transfers.append((t, x, 0))

        rewards = compute_retroactive_build_rewards(sc.build())

        # (1,0): built round 5, 40 + 40 = 80 transfers
        # (2,0) first life: built round 5, alive [5, 55) → 40 transfers (10-49)
        # (2,0) second life: built round 55, alive [55, 100) → 40 transfers (60-99)
        # (3,0): built round 5, 40 + 40 = 80 transfers
        assert rewards[5] == 80.0 + 40.0 + 80.0  # round 5: tiles (1,0), (2,0)v1, (3,0)
        assert rewards[55] == 40.0                # round 55: tile (2,0) v2


class TestBranchingSupplyChain:
    """Branching: splitter fans out to multiple chains.

        core(0,0) ← conv(1,0) ← splitter(2,0) ← conv(3,0) ← harv(4,0)
                                      ↓
                                  conv(2,1) ← conv(3,1) ← harv(4,1)
    """

    def test_splitter_with_two_branches(self):
        """Splitter feeds two branches — all tiles in both branches get credit."""
        sc = SupplyChainBuilder(n_turns=200, core_xy=(0, 0))
        # Main trunk
        sc.place(5, 4, 0, dx=-1, dy=0)   # harvester
        sc.place(5, 3, 0, dx=-1, dy=0)   # conv
        sc.place(5, 2, 0, dx=-1, dy=0)   # splitter (outputs west, treated same)
        sc.place(5, 1, 0, dx=-1, dy=0)   # conv near core
        # Branch
        sc.place(5, 4, 1, dx=-1, dy=0)   # harvester branch
        sc.place(5, 3, 1, dx=-1, dy=0)   # conv branch
        sc.place(5, 2, 1, dx=-1, dy=0)   # conv (feeds into splitter from south)

        # Main trunk transfers every round from 10-199
        sc.run_transfers(start=10, end=200)

        rewards = compute_retroactive_build_rewards(sc.build())

        # All built on round 5, 190 rounds of transfers each
        assert rewards == {5: 190.0 * 7}

    def test_branch_added_later(self):
        """Second branch added at turn 50 — only gets credit from turn 50 onward."""
        sc = SupplyChainBuilder(n_turns=200, core_xy=(0, 0))
        # Main chain built turn 5
        sc.place(5, 3, 0, dx=-1, dy=0)
        sc.place(5, 2, 0, dx=-1, dy=0)
        sc.place(5, 1, 0, dx=-1, dy=0)
        # Branch added turn 50
        sc.place(50, 3, 1, dx=-1, dy=0)
        sc.place(50, 2, 1, dx=-1, dy=0)

        # Main chain transfers from turn 10
        for t in range(10, 200):
            for x in [1, 2, 3]:
                sc.transfers.append((t, x, 0))
        # Branch transfers from turn 55
        for t in range(55, 200):
            for x in [2, 3]:
                sc.transfers.append((t, x, 1))

        rewards = compute_retroactive_build_rewards(sc.build())

        assert rewards[5] == 190.0 * 3    # 3 tiles × 190 transfers
        assert rewards[50] == 145.0 * 2   # 2 tiles × 145 transfers


class TestEnemyRaidScenario:
    """Enemy destroys part of supply chain, player rebuilds.

    Tests that destroyed conveyors stop accumulating credit and
    rebuilt ones start fresh.
    """

    def test_raid_and_rebuild(self):
        """Enemy destroys two conveyors at turn 100, rebuilt at turn 110."""
        sc = SupplyChainBuilder(n_turns=500, core_xy=(0, 0))
        # Initial chain built turn 5
        sc.place(5, 1, 0, dx=-1, dy=0)
        sc.place(5, 2, 0, dx=-1, dy=0)
        sc.place(5, 3, 0, dx=-1, dy=0)
        sc.place(5, 4, 0, dx=-1, dy=0)

        # Flowing turns 10-99 (all 4 tiles)
        for t in range(10, 100):
            for x in [1, 2, 3, 4]:
                sc.transfers.append((t, x, 0))

        # Enemy raid destroys tiles (2,0) and (3,0) at turn 100
        sc.destroy(100, 2, 0)
        sc.destroy(100, 3, 0)

        # Only (1,0) and (4,0) transfer during downtime 100-109
        for t in range(100, 110):
            sc.transfers.append((t, 1, 0))
            sc.transfers.append((t, 4, 0))

        # Rebuild at turn 110
        sc.place(110, 2, 0, dx=-1, dy=0)
        sc.place(110, 3, 0, dx=-1, dy=0)

        # Full chain resumes 115-499
        for t in range(115, 500):
            for x in [1, 2, 3, 4]:
                sc.transfers.append((t, x, 0))

        rewards = compute_retroactive_build_rewards(sc.build())

        # Round 5 (initial build):
        #   (1,0): 90 + 10 + 385 = 485 (never destroyed)
        #   (2,0): 90 (alive [5, 110))
        #   (3,0): 90 (alive [5, 110))
        #   (4,0): 90 + 10 + 385 = 485 (never destroyed)
        assert rewards[5] == 485 + 90 + 90 + 485

        # Round 110 (rebuild):
        #   (2,0): 385 (turns 115-499)
        #   (3,0): 385 (turns 115-499)
        assert rewards[110] == 385 + 385

    def test_repeated_raids(self):
        """Tile destroyed and rebuilt multiple times."""
        sc = SupplyChainBuilder(n_turns=300, core_xy=(0, 0))

        # Build → destroy → rebuild → destroy → rebuild
        sc.place(10, 1, 0, dx=-1, dy=0)
        for t in range(15, 50):
            sc.transfers.append((t, 1, 0))
        sc.destroy(50, 1, 0)

        sc.place(60, 1, 0, dx=-1, dy=0)
        for t in range(65, 100):
            sc.transfers.append((t, 1, 0))
        sc.destroy(100, 1, 0)

        sc.place(110, 1, 0, dx=-1, dy=0)
        for t in range(115, 300):
            sc.transfers.append((t, 1, 0))

        rewards = compute_retroactive_build_rewards(sc.build())

        assert rewards[10] == 35    # turns 15-49
        assert rewards[60] == 35    # turns 65-99
        assert rewards[110] == 185  # turns 115-299


class TestDualSupplyLines:
    """Two independent supply lines to core from opposite directions.

    Tests that tiles on different chains don't interfere.

        harv(4,0) → conv(3,0) → conv(2,0) → conv(1,0) → core(0,0)
                                                           ↑
        harv(0,4) → conv(0,3) → conv(0,2) → conv(0,1) ───┘
    """

    def test_independent_chains(self):
        sc = SupplyChainBuilder(n_turns=200, core_xy=(0, 0))

        # East chain (built turn 5)
        for x in range(1, 5):
            sc.place(5, x, 0, dx=-1, dy=0)
        # South chain (built turn 10)
        for y in range(1, 5):
            sc.place(10, 0, y, dx=0, dy=-1)

        # East chain flows from turn 8
        for t in range(8, 200):
            for x in range(1, 5):
                sc.transfers.append((t, x, 0))
        # South chain flows from turn 15
        for t in range(15, 200):
            for y in range(1, 5):
                sc.transfers.append((t, 0, y))

        rewards = compute_retroactive_build_rewards(sc.build())

        assert rewards[5] == 192 * 4    # east: 4 tiles × 192 transfers
        assert rewards[10] == 185 * 4   # south: 4 tiles × 185 transfers

    def test_one_chain_destroyed_other_unaffected(self):
        """Destroying one chain doesn't affect the other's credit."""
        sc = SupplyChainBuilder(n_turns=200, core_xy=(0, 0))

        # East chain
        sc.place(5, 1, 0, dx=-1, dy=0)
        sc.place(5, 2, 0, dx=-1, dy=0)
        # South chain
        sc.place(5, 0, 1, dx=0, dy=-1)
        sc.place(5, 0, 2, dx=0, dy=-1)

        # Both flow turns 10-99
        for t in range(10, 100):
            sc.transfers.extend([(t, 1, 0), (t, 2, 0), (t, 0, 1), (t, 0, 2)])

        # East chain destroyed at 100
        sc.destroy(100, 1, 0)
        sc.destroy(100, 2, 0)

        # South chain continues 100-199
        for t in range(100, 200):
            sc.transfers.extend([(t, 0, 1), (t, 0, 2)])

        rewards = compute_retroactive_build_rewards(sc.build())

        # Round 5:
        #   east (1,0): 90, east (2,0): 90
        #   south (0,1): 90+100=190, south (0,2): 90+100=190
        assert rewards == {5: 90 + 90 + 190 + 190}


class TestEdgeCases:
    """Stress tests and boundary conditions."""

    def test_massive_chain(self):
        """50-tile chain, 2000 turns — performance and correctness."""
        sc = SupplyChainBuilder(n_turns=2000, core_xy=(0, 0))
        for x in range(1, 51):
            sc.place(x, x, 0, dx=-1, dy=0)
        sc.run_transfers(start=55, end=2000)

        rewards = compute_retroactive_build_rewards(sc.build())

        total = sum(rewards.values())
        # 50 tiles × 1945 transfers each
        assert total == 50 * 1945

    def test_all_conveyors_built_same_round(self):
        """20 conveyors built simultaneously."""
        sc = SupplyChainBuilder(n_turns=500, core_xy=(0, 0))
        for x in range(1, 21):
            sc.place(10, x, 0, dx=-1, dy=0)
        sc.run_transfers(start=10, end=500)

        rewards = compute_retroactive_build_rewards(sc.build())

        assert rewards == {10: 490 * 20}

    def test_zero_turn_game(self):
        assert compute_retroactive_build_rewards({"n_turns": 0}) == {}

    def test_conveyor_built_last_turn(self):
        """Conveyor built on the very last turn — no time to transfer."""
        sc = SupplyChainBuilder(n_turns=100, core_xy=(0, 0))
        sc.place(99, 1, 0, dx=-1, dy=0)
        # No transfers possible

        rewards = compute_retroactive_build_rewards(sc.build())
        assert rewards == {}

    def test_transfer_exactly_at_n_turns_not_counted(self):
        """Transfer at turn == n_turns is out of bounds (should not exist
        in real data, but verify we don't count it)."""
        gs = {
            "conveyor_births_buf": _pack_triples([(5, 1, 0)]),
            "tile_transfers_buf": _pack_triples([(100, 1, 0)]),  # at n_turns
            "n_turns": 100,
        }
        rewards = compute_retroactive_build_rewards(gs)
        # Turn 100 is out of [5, 100), so not counted
        assert rewards == {}

    def test_dense_rebuild_same_tile_many_times(self):
        """Tile rebuilt 10 times — each life gets only its own transfers."""
        births = [(i * 10, 1, 0) for i in range(10)]  # turns 0, 10, 20, ..., 90
        # One transfer per life, 5 turns after birth
        transfers = [(i * 10 + 5, 1, 0) for i in range(10)]
        gs = {
            "conveyor_births_buf": _pack_triples(births),
            "tile_transfers_buf": _pack_triples(transfers),
            "n_turns": 100,
        }
        rewards = compute_retroactive_build_rewards(gs)

        for i in range(10):
            assert rewards[i * 10] == 1.0
