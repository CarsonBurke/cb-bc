"""CUDA-first helpers for replay pretraining."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PretrainCudaConfig:
    enabled: bool
    amp_dtype: torch.dtype | None
    channels_last: bool
    compile_model: bool


def resolve_cuda_config(
    device: torch.device,
    *,
    spatial_inputs: bool,
    enable_compile: bool = True,
    bf16_supported: bool | None = None,
    compile_supported: bool | None = None,
) -> PretrainCudaConfig:
    """Pick CUDA execution settings for a pretrain workload."""
    if device.type != "cuda":
        return PretrainCudaConfig(
            enabled=False,
            amp_dtype=None,
            channels_last=False,
            compile_model=False,
        )

    if bf16_supported is None:
        bf16_supported = torch.cuda.is_bf16_supported()
    if compile_supported is None:
        compile_supported = hasattr(torch, "compile")

    # Dynamic spatial workloads (variable map batches/shapes) trigger very
    # expensive torch.compile recompiles, so keep compile for non-spatial paths.
    compile_model = enable_compile and compile_supported and not spatial_inputs

    return PretrainCudaConfig(
        enabled=True,
        amp_dtype=torch.bfloat16 if bf16_supported else torch.float16,
        channels_last=spatial_inputs,
        compile_model=compile_model,
    )


def configure_cuda_runtime(device: torch.device) -> None:
    """Enable the CUDA runtime knobs that help training throughput."""
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def autocast_context(config: PretrainCudaConfig):
    if not config.enabled:
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=config.amp_dtype)


def maybe_compile(module, config: PretrainCudaConfig, *, label: str):
    if not config.compile_model:
        return module
    try:
        return torch.compile(module)
    except Exception as exc:
        print(f"Warning: torch.compile disabled for {label}: {exc}")
        return module


def prepare_spatial_batch(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    non_blocking: bool,
    channels_last: bool,
) -> torch.Tensor:
    tensor = tensor.to(device=device, non_blocking=non_blocking)
    if channels_last and tensor.ndim == 4:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor
