"""Regression tests for tensor training utilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


training_utils = _load_module("tensor_training_utils_for_tests", "bots/tensor/torch/training_utils.py")


def _league_with_models(tmp_path: Path):
    league = training_utils.League(tmp_path, tmp_path / "league.json", max_size=20)
    league.models = [
        {"id": "gen_0", "gen": 0, "mu": 20.0, "sigma": 7.0, "elo": -1.0, "path": "g0.pkl", "wins": 0, "losses": 0, "draws": 0},
        {"id": "gen_1", "gen": 1, "mu": 25.0, "sigma": 6.0, "elo": 7.0, "path": "g1.pkl", "wins": 0, "losses": 0, "draws": 0},
    ]
    return league


def test_sample_opponents_excludes_latest_when_alternative_exists(tmp_path):
    league = _league_with_models(tmp_path)
    opponents = league.sample_opponents(1)
    assert [opp["id"] for opp in opponents] == ["gen_0"]


def test_update_elo_ignores_self_match(tmp_path):
    league = _league_with_models(tmp_path)
    before = dict(league.models[1])
    league.update_elo("gen_1", "gen_1", "a")
    assert league.models[1] == before
