"""Regression coverage for GPU-first pretrain configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gpu = _load_module("tensor_pretrain_gpu_for_tests", "bots/tensor/torch/pretrain/gpu.py")


def test_resolve_cuda_config_disables_gpu_paths_on_cpu():
    cfg = gpu.resolve_cuda_config(torch.device("cpu"), spatial_inputs=True)
    assert not cfg.enabled
    assert cfg.amp_dtype is None
    assert not cfg.channels_last
    assert not cfg.compile_model


def test_resolve_cuda_config_prefers_bf16_and_channels_last_on_cuda():
    cfg = gpu.resolve_cuda_config(
        torch.device("cuda"),
        spatial_inputs=True,
        enable_compile=True,
        bf16_supported=True,
        compile_supported=True,
    )
    assert cfg.enabled
    assert cfg.amp_dtype == torch.bfloat16
    assert cfg.channels_last
    assert cfg.compile_model


def test_resolve_cuda_config_falls_back_to_fp16_without_compile():
    cfg = gpu.resolve_cuda_config(
        torch.device("cuda"),
        spatial_inputs=False,
        enable_compile=False,
        bf16_supported=False,
        compile_supported=True,
    )
    assert cfg.enabled
    assert cfg.amp_dtype == torch.float16
    assert not cfg.channels_last
    assert not cfg.compile_model
