"""Single-device NorMuon optimizer helpers for actor/critic training."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

_AUX_PARAM_NAMES = {
    "actor_cls",
    "global_cls",
}


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the zeroth power / orthogonalization used by Muon-style updates."""
    if grad.ndim < 2:
        raise ValueError(f"NorMuon requires matrix-shaped gradients, got ndim={grad.ndim}")
    work_dtype = torch.bfloat16 if grad.device.type == "cuda" else torch.float32
    x = grad.to(dtype=work_dtype)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        gram = x @ x.mT
        poly = b * gram + c * gram @ gram
        x = a * x + poly @ x
    if transposed:
        x = x.mT
    return x


def normuon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    second_momentum: torch.Tensor,
    *,
    beta: float = 0.95,
    beta2: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """Compute a NorMuon update for one parameter tensor."""
    momentum.lerp_(grad, 1.0 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    original_shape: tuple[int, ...] | None = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps).to(dtype=grad.dtype)
    if original_shape is not None:
        update = update.reshape(original_shape)

    update_norm = update.norm(dim=(-2, -1), keepdim=True)
    mean_square = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(mean_square, 1.0 - beta2)
    step_scale = second_momentum.sqrt().add(1e-10).reciprocal()
    update = update * step_scale
    renorm = update.norm(dim=(-2, -1), keepdim=True).add(1e-10)
    update = update * (update_norm / renorm)
    update = update * math.sqrt(max(1.0, grad.size(-2) / grad.size(-1)))
    return update


def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    *,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    """Reference Adam update used for aux parameters outside the NorMuon path."""
    exp_avg.lerp_(grad, 1.0 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1.0 - betas[1])
    exp_avg_hat = exp_avg / (1.0 - betas[0] ** step)
    exp_avg_sq_hat = exp_avg_sq / (1.0 - betas[1] ** step)
    return exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps)


class SingleDeviceNorMuonWithAuxAdam(torch.optim.Optimizer):
    """NorMuon for matrix weights paired with Adam on auxiliary parameters."""

    def __init__(self, param_groups: list[dict]):
        if not param_groups:
            raise ValueError("NorMuon optimizer requires at least one parameter group")
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("each parameter group must set use_muon")
            if group["use_muon"]:
                group["lr"] = group.get("lr", 3e-4)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for param in group["params"]:
                    had_grad = param.grad is not None
                    grad = param.grad if had_grad else torch.zeros_like(param)
                    state = self.state[param]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(param)
                        state["second_momentum_buffer"] = torch.zeros_like(param[..., :1])
                    update = normuon_update(
                        grad,
                        state["momentum_buffer"],
                        state["second_momentum_buffer"],
                        beta=group["momentum"],
                        beta2=group["beta2"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    )
                    if group["weight_decay"] and had_grad:
                        param.mul_(1.0 - group["lr"] * group["weight_decay"])
                    param.add_(update.reshape(param.shape), alpha=-group["lr"])
            else:
                for param in group["params"]:
                    had_grad = param.grad is not None
                    grad = param.grad if had_grad else torch.zeros_like(param)
                    state = self.state[param]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        betas=group["betas"],
                        eps=group["eps"],
                    )
                    if group["weight_decay"] and had_grad:
                        param.mul_(1.0 - group["lr"] * group["weight_decay"])
                    param.add_(update, alpha=-group["lr"])
        return loss


def use_normuon(name: str, param: torch.nn.Parameter) -> bool:
    """Return whether a named parameter should use the NorMuon path."""
    if not param.requires_grad or param.ndim not in (2, 4):
        return False
    if name in _AUX_PARAM_NAMES:
        return False
    if name.startswith("head_"):
        return False
    if name == "value_head.mlp.4.weight":
        return False
    return True


def split_normuon_params(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    """Split named parameters into NorMuon and auxiliary Adam buckets."""
    muon_params: list[tuple[str, torch.nn.Parameter]] = []
    aux_params: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if use_normuon(name, param):
            muon_params.append((name, param))
        else:
            aux_params.append((name, param))
    if not muon_params and not aux_params:
        raise ValueError("module has no trainable parameters")
    return muon_params, aux_params


def make_normuon_optimizer(
    module: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.0,
    muon_momentum: float = 0.95,
    muon_beta2: float = 0.95,
    adam_betas: tuple[float, float] = (0.9, 0.95),
    adam_eps: float = 1e-10,
) -> torch.optim.Optimizer:
    """Build the local single-device NorMuon optimizer for a module."""
    muon_named, aux_named = split_normuon_params(module.named_parameters())
    param_groups: list[dict] = []
    if muon_named:
        param_groups.append({
            "params": [param for _, param in muon_named],
            "lr": lr,
            "momentum": muon_momentum,
            "beta2": muon_beta2,
            "weight_decay": weight_decay,
            "use_muon": True,
        })
    if aux_named:
        param_groups.append({
            "params": [param for _, param in aux_named],
            "lr": lr,
            "betas": adam_betas,
            "eps": adam_eps,
            "weight_decay": weight_decay,
            "use_muon": False,
        })
    return SingleDeviceNorMuonWithAuxAdam(param_groups)
