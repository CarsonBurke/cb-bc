"""Tests for dense replay-derived flow reward assignment."""

import struct
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"
TENSOR_DIR = PROJECT_ROOT / "bots" / "tensor"
REPLAY_PARSER_PYTHON = PROJECT_ROOT / "packages" / "replay_parser" / "python"

sys.path.insert(0, str(TENSOR_DIR))
sys.path.insert(0, str(TORCH_DIR))
sys.path.insert(0, str(REPLAY_PARSER_PYTHON))

from trajectory_schema import make_trajectory_step
from flow_reward import (
    compute_dense_flow_rewards, compute_dense_delivery_rewards,
    augment_trajectories_with_flow_rewards,
    _TILE_FEATURES, _OBS_GLOBAL_START, _N_SCALARS, _RESOURCES_COLLECTED_IDX,
    _RESOURCES_COLLECTED_NORM, REWARD_INCOME_SCALE, REWARD_FLOW_SCALE,
)

OBS_SIZE = 5 * 5 * _TILE_FEATURES + 17 + 8  # fallback
try:
    from model import _compute_gae, _parse_trajectories, OBS_SIZE as MODEL_OBS_SIZE, H_DIM as MODEL_H_DIM, REWARD_BUILD_CREDIT
    from train_batched import _shared_flow_reward_total
    OBS_SIZE = MODEL_OBS_SIZE
    H_DIM = MODEL_H_DIM
    HAS_TORCH = True
except ImportError:
    H_DIM = 24
    HAS_TORCH = False


def _pack_float_pairs(pairs: list[tuple[float, float]]) -> bytes:
    return b"".join(struct.pack("<ff", a, b) for a, b in pairs)


def _pack_int_pairs(pairs: list[tuple[int, int]]) -> bytes:
    return b"".join(struct.pack("<ii", a, b) for a, b in pairs)


def _make_step(round_num: int, n_working: float = 0.0):
    obs = [0.0] * OBS_SIZE
    obs[_OBS_GLOBAL_START] = round_num / 2000.0
    return make_trajectory_step(
        obs=struct.pack(f"<{OBS_SIZE}f", *obs),
        h_prev=struct.pack(f"<{H_DIM}f", *([0.0] * H_DIM)),
        obs_bytes_len=OBS_SIZE * 4,
        h_bytes_len=H_DIM * 4,
        round_idx=round_num,
        bt=0,
        bd=0,
        cd=0,
        brt=0,
        md=0,
        reward=n_working,
        old_log_prob=0.0,
        enemy_core_dmg=0.0,
        own_core_dmg=0.0,
        harv_attempt=0,
        harv_fail=0,
        harv_prox_gate=0,
        harv_prox_paid=0,
        conv_attempt=0,
        conv_fail=0,
        conv_prox_gate=0,
        conv_prox_paid=0,
    )


def _pack_scalars(resources_collected_cumulative: list[float]) -> bytes:
    """Build a scalars buffer with resources_collected at index 9."""
    buf = b""
    for rc in resources_collected_cumulative:
        sc = [0.0] * _N_SCALARS
        sc[_RESOURCES_COLLECTED_IDX] = rc / _RESOURCES_COLLECTED_NORM
        buf += struct.pack(f"<{_N_SCALARS}f", *sc)
    return buf


def _zero_scalars(n: int) -> dict:
    """Helper: zero-resource scalars buffer for n turns."""
    return {"scalars_buf": _pack_scalars([0.0] * n)}


class TestComputeDenseFlowRewards:
    def test_empty_buffer(self):
        assert compute_dense_flow_rewards({}) == {}
        # n_turns > 0 but no transfer_events → raises
        with pytest.raises(ValueError, match="transfer_events"):
            compute_dense_flow_rewards({"n_turns": 10})

    def test_decodes_team_a_rewards(self):
        gs = {
            "transfer_events_buf": _pack_float_pairs([(0.0, 1.0), (2.5, 3.0), (0.0, 0.0)]),
            "n_turns": 3,
            **_zero_scalars(3),
        }
        assert compute_dense_flow_rewards(gs) == {1: 2.5 * REWARD_FLOW_SCALE}

    def test_decodes_team_b_rewards(self):
        gs = {
            "transfer_events_buf": _pack_float_pairs([(0.0, 1.0), (2.5, 3.0), (0.0, 0.5)]),
            "n_turns": 3,
            **_zero_scalars(3),
        }
        assert compute_dense_flow_rewards(gs, team=1) == {
            0: 1.0 * REWARD_FLOW_SCALE,
            1: 3.0 * REWARD_FLOW_SCALE,
            2: 0.5 * REWARD_FLOW_SCALE,
        }

    def test_includes_core_deliveries(self):
        """Core deliveries from resources_collected deltas."""
        gs = {
            "transfer_events_buf": _pack_float_pairs([(1.0, 0.0), (0.0, 0.0)]),
            # cumulative resources_collected: 7 at turn 0, still 7 at turn 1
            "scalars_buf": _pack_scalars([7.0, 7.0]),
            "n_turns": 2,
        }
        rewards = compute_dense_flow_rewards(gs, team=0)
        # round 0: 1.0 * REWARD_FLOW_SCALE (flow) + 7 * REWARD_INCOME_SCALE
        assert rewards[0] == pytest.approx(1.0 * REWARD_FLOW_SCALE + 7.0 * REWARD_INCOME_SCALE)
        assert 1 not in rewards

    def test_delivery_only_no_flow(self):
        """Rounds with zero flow but nonzero delivery still get reward."""
        gs = {
            "transfer_events_buf": _pack_float_pairs([(0.0, 0.0)]),
            # 5 resources collected on turn 0
            "scalars_buf": _pack_scalars([5.0]),
            "n_turns": 1,
        }
        rewards = compute_dense_flow_rewards(gs, team=0)
        assert rewards[0] == pytest.approx(5.0 * REWARD_INCOME_SCALE)

    def test_no_scalars_flow_only(self):
        """Missing scalars buffer → raises (scalars required)."""
        gs = {
            "transfer_events_buf": _pack_float_pairs([(2.0, 0.0)]),
            "n_turns": 1,
        }
        with pytest.raises(ValueError, match="scalars"):
            compute_dense_flow_rewards(gs, team=0)

    def test_path_backed_buffers_load_without_writable_copy(self, tmp_path):
        transfer_path = tmp_path / "transfer.bin"
        transfer_path.write_bytes(_pack_float_pairs([(2.0, 0.0)]))
        scalars_path = tmp_path / "scalars.bin"
        scalars_path.write_bytes(_pack_scalars([0.0]))
        gs = {
            "transfer_events_path": str(transfer_path),
            "scalars_path": str(scalars_path),
            "n_turns": 1,
        }

        rewards = compute_dense_flow_rewards(gs, team=0)

        assert rewards[0] == pytest.approx(2.0 * REWARD_FLOW_SCALE)
        assert isinstance(gs["transfer_events_buf"], bytes)


class TestAugmentTrajectories:
    def test_zeroes_stale_n_working(self):
        step = _make_step(round_num=50, n_working=999.0)
        traj = [[step]]
        augment_trajectories_with_flow_rewards(traj, {})
        assert traj[0][0].reward == 0.0

    def test_writes_reward_to_correct_round(self):
        step10 = _make_step(round_num=10)
        step20 = _make_step(round_num=20)
        traj = [[step10, step20]]
        augment_trajectories_with_flow_rewards(traj, {10: 5.0, 20: 3.0})
        assert traj[0][0].reward == 5.0
        assert traj[0][1].reward == 3.0

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    def test_shared_round_reward_is_not_multiplied_by_unit_count(self):
        step_a = _make_step(round_num=10)
        step_b = _make_step(round_num=10)
        trajs = [[step_a], [step_b]]
        augment_trajectories_with_flow_rewards(trajs, {10: 3.0})
        data = _parse_trajectories(trajs, traj_game_map=[0, 0])
        assert data is not None
        expected = 3.0 * REWARD_BUILD_CREDIT
        assert abs(data["rewards"][0].item() - expected) < 1e-6
        assert abs(data["rewards"][1].item() - expected) < 1e-6

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    def test_shared_flow_reward_total_counts_each_round_once(self):
        step_a = _make_step(round_num=10)
        step_b = _make_step(round_num=10)
        step_c = _make_step(round_num=20)
        trajs = [[step_a, step_c], [step_b]]
        augment_trajectories_with_flow_rewards(trajs, {10: 3.0, 20: 2.0})
        assert _shared_flow_reward_total(trajs) == 5.0 * REWARD_BUILD_CREDIT


@pytest.fixture
def replay_path():
    candidates = list(Path(".").glob("*.replay26")) + list(Path("replays").glob("*.replay26"))
    if not candidates:
        pytest.skip("No replay files available")
    return str(candidates[0])


class TestDeliveryRewards:
    def test_empty(self):
        assert compute_dense_delivery_rewards({}) == {}

    def test_cumulative_deltas(self):
        # cumulative: 0, 3, 3, 10 → deltas: 0, 3, 0, 7
        gs = {
            "scalars_buf": _pack_scalars([0.0, 3.0, 3.0, 10.0]),
            "n_turns": 4,
        }
        rewards = compute_dense_delivery_rewards(gs)
        assert 0 not in rewards
        assert rewards[1] == pytest.approx(3.0)
        assert 2 not in rewards
        assert rewards[3] == pytest.approx(7.0)


class TestIntegration:
    def test_replay_round_trip(self, replay_path):
        replay_parser = pytest.importorskip("replay_parser")
        ReplayData = replay_parser.ReplayData

        rd = ReplayData.from_file(replay_path, team=0)
        gs = {
            "transfer_events_buf": rd.resource_transfers_bytes,
            "scalars_buf": rd.scalars_bytes,
            "n_turns": rd.n_turns,
        }
        rewards = compute_dense_flow_rewards(gs)

        for rnd, val in rewards.items():
            assert val >= 0.0
            assert 0 <= rnd < rd.n_turns

        steps = [_make_step(r) for r in range(0, min(rd.n_turns, 50))]
        traj = [steps]
        augment_trajectories_with_flow_rewards(traj, rewards)

        for s in steps:
            assert isinstance(s.reward, (int, float))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestComputeGAE:
    def test_long_episode_tail_is_terminal(self):
        import torch

        rewards = torch.zeros(1900)
        values = torch.full((1900,), 5.0)
        advantages, returns = _compute_gae(
            rewards,
            values,
            [(0, 1900)],
            gamma=0.99,
            gae_lambda=1.0,
        )

        assert advantages[-1].item() == pytest.approx(-5.0)
        assert returns[-1].item() == pytest.approx(0.0)
