"""
Z-quad shard experts: heterogeneous Micro-UNet per z chunk with a global param budget.
"""
from __future__ import annotations

import copy
import sys
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def z_quad_shard_bounds(depth: int, n_shards: int = 4) -> List[Tuple[int, int]]:
    depth = int(depth)
    n_shards = int(n_shards)
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    chunk = (depth + n_shards - 1) // n_shards
    bounds = []
    for i in range(n_shards):
        z0 = i * chunk
        z1 = min((i + 1) * chunk, depth)
        if z0 < z1:
            bounds.append((z0, z1))
    return bounds


def compute_shard_residual_energy(
    x_true: np.ndarray,
    x_prime: np.ndarray,
    bounds: Sequence[Tuple[int, int]],
) -> List[float]:
    residual = np.abs(np.asarray(x_true, np.float32) - np.asarray(x_prime, np.float32))
    energies = []
    for z0, z1 in bounds:
        slab = residual[z0:z1]
        energies.append(float(np.mean(slab)) if slab.size else 0.0)
    return energies


def _estimate_bg_params(
    bg_h: int,
    shape: Tuple[int, int, int],
    n_fields: int = 6,
    bg_arch: str = "spatial",
    dtype_bytes: int = 2,
    **kwargs
) -> int:
    from experiment import estimate_bg_model_param_bytes

    estimate_kwargs = {
        "n_fields": int(n_fields),
        "shape": shape,
        "bg_arch": str(bg_arch),
        "bg_h": int(bg_h),
        "dtype_bytes": int(dtype_bytes),
        "bg_split_bands": True,
        "bg_split_mode": "three",
    }
    estimate_kwargs.update(kwargs)
    n_p, _ = estimate_bg_model_param_bytes(**estimate_kwargs)
    return int(n_p)


def allocate_bg_h_hetero(
    energies: Sequence[float],
    total_param_budget: int = 4000,
    h_candidates: Sequence[int] = tuple(range(3, 30)),
    shape: Tuple[int, int, int] = (512, 512, 512),
    n_fields: int = 6,
    bg_arch: str = "spatial",
) -> Tuple[List[int], List[int], int]:
    """
    Pick per-shard bg_h maximizing sum(energy_i * h_i) s.t. sum(params) <= budget.
    Returns (bg_h_list, param_counts, total_params).
    """
    n = len(energies)
    energies = np.asarray(energies, dtype=np.float64)
    h_candidates = tuple(int(h) for h in h_candidates)
    param_cache = {h: _estimate_bg_params(h, shape, n_fields, bg_arch) for h in h_candidates}

    best_hs = None
    best_score = -np.inf
    best_total = 0
    for hs in product(h_candidates, repeat=n):
        counts = [param_cache[h] for h in hs]
        total = int(sum(counts))
        if total > int(total_param_budget):
            continue
        score = float(np.dot(energies, np.asarray(hs, dtype=np.float64)))
        if score > best_score:
            best_score = score
            best_hs = list(hs)
            best_total = total

    if best_hs is None:
        h_min = min(h_candidates, key=lambda h: param_cache[h])
        best_hs = [h_min] * n
        best_total = param_cache[h_min] * n

    return best_hs, [param_cache[h] for h in best_hs], int(best_total)


def allocate_bg_h_uniform(
    n_shards: int,
    total_param_budget: int = 4000,
    h_candidates: Sequence[int] = tuple(range(3, 30)),
    shape: Tuple[int, int, int] = (512, 512, 512),
    n_fields: int = 6,
    bg_arch: str = "spatial",
) -> Tuple[int, int]:
    """Largest equal bg_h with n_shards * params(h) <= budget."""
    h_candidates = sorted((int(h) for h in h_candidates), reverse=True)
    for h in h_candidates:
        p = _estimate_bg_params(h, shape, n_fields, bg_arch)
        if p * int(n_shards) <= int(total_param_budget):
            return int(h), int(p * n_shards)
    h = min(h_candidates)
    p = _estimate_bg_params(h, shape, n_fields, bg_arch)
    return int(h), int(p * n_shards)


def pick_bg_h_under_budget(
    param_budget: int,
    energy: float = 1.0,
    h_candidates: Sequence[int] = tuple(range(3, 30)),
    shape: Tuple[int, int, int] = (512, 512, 512),
    n_fields: int = 6,
    bg_arch: str = "spatial",
    **kwargs
) -> Tuple[int, int]:
    """
    One model / one shard: largest bg_h with params(h) <= param_budget.
    Tie-break: maximize energy * h (hetero per-shard scoring).
    """
    h_candidates = tuple(int(h) for h in h_candidates)
    param_cache = {h: _estimate_bg_params(h, shape, n_fields, bg_arch, **kwargs) for h in h_candidates}
    budget = int(param_budget)
    energy = float(energy)

    best_h = min(h_candidates, key=lambda h: param_cache[h])
    best_score = -np.inf
    for h in h_candidates:
        p = param_cache[h]
        if p > budget:
            continue
        score = energy * float(h)
        if score > best_score:
            best_score = score
            best_h = int(h)
    return int(best_h), int(param_cache[best_h])


def allocate_bg_h_hetero_per_shard(
    energies: Sequence[float],
    per_shard_param_budget: int = 4000,
    h_candidates: Sequence[int] = tuple(range(3, 30)),
    shape: Tuple[int, int, int] = (512, 512, 512),
    n_fields: int = 6,
    bg_arch: str = "spatial",
) -> Tuple[List[int], List[int], int]:
    """Each shard independently: params(h_i) <= per_shard_param_budget."""
    bg_hs, param_each = [], []
    for e in energies:
        h, p = pick_bg_h_under_budget(
            per_shard_param_budget,
            energy=float(e),
            h_candidates=h_candidates,
            shape=shape,
            n_fields=n_fields,
            bg_arch=bg_arch,
        )
        bg_hs.append(h)
        param_each.append(p)
    return bg_hs, param_each, int(sum(param_each))


def allocate_bg_h_uniform_per_shard(
    n_shards: int,
    per_shard_param_budget: int = 4000,
    h_candidates: Sequence[int] = tuple(range(3, 30)),
    shape: Tuple[int, int, int] = (512, 512, 512),
    n_fields: int = 6,
    bg_arch: str = "spatial",
) -> Tuple[List[int], List[int], int]:
    """Same bg_h on every shard; each shard may use up to per_shard_param_budget."""
    h_uni, p_one = pick_bg_h_under_budget(
        per_shard_param_budget,
        energy=1.0,
        h_candidates=h_candidates,
        shape=shape,
        n_fields=n_fields,
        bg_arch=bg_arch,
    )
    n = int(n_shards)
    return [h_uni] * n, [p_one] * n, int(p_one * n)


def crop_multifield_zyx(
    Xs: Sequence[np.ndarray],
    Xps: Sequence[np.ndarray],
    z0: int,
    z1: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    z0, z1 = int(z0), int(z1)
    return (
        [np.asarray(x[z0:z1], np.float32) for x in Xs],
        [np.asarray(x[z0:z1], np.float32) for x in Xps],
    )


def build_shard_plan(
    x_true: np.ndarray,
    x_prime: np.ndarray,
    depth: int,
    total_param_budget: int = 4000,
    per_shard_param_budget: Optional[int] = None,
    heterogeneous: bool = True,
) -> List[Dict[str, Any]]:
    """
    Parameter budget modes (mutually exclusive):

    - per_shard_param_budget: each GPU/shard may use up to this many params
      (4 shards × 4000 ≈ 16k total for the ensemble).
    - total_param_budget: legacy global cap split across shards (4 shards share 4000).
    """
    bounds = z_quad_shard_bounds(depth, 4)
    energies = compute_shard_residual_energy(x_true, x_prime, bounds)
    shape = tuple(int(s) for s in np.asarray(x_true).shape)
    n = len(bounds)

    if per_shard_param_budget is not None:
        budget_each = int(per_shard_param_budget)
        if heterogeneous:
            bg_hs, param_each, total_p = allocate_bg_h_hetero_per_shard(
                energies, per_shard_param_budget=budget_each, shape=shape
            )
        else:
            bg_hs, param_each, total_p = allocate_bg_h_uniform_per_shard(
                n, per_shard_param_budget=budget_each, shape=shape
            )
    elif heterogeneous:
        bg_hs, param_each, total_p = allocate_bg_h_hetero(
            energies, total_param_budget=total_param_budget, shape=shape
        )
    else:
        h_uni, total_p = allocate_bg_h_uniform(
            n, total_param_budget=total_param_budget, shape=shape
        )
        bg_hs = [h_uni] * len(bounds)
        param_each = [_estimate_bg_params(h_uni, shape) for _ in bounds]

    plan = []
    for i, ((z0, z1), e, bg_h, n_p) in enumerate(zip(bounds, energies, bg_hs, param_each)):
        plan.append(
            {
                "shard_id": i,
                "z0": int(z0),
                "z1": int(z1),
                "depth": int(z1 - z0),
                "energy": float(e),
                "bg_h": int(bg_h),
                "n_params": int(n_p),
            }
        )
    plan.append({"total_params": int(sum(param_each)), "heterogeneous": bool(heterogeneous)})
    return plan


def _z_segment_profile(length: int, blend: str = "hann", floor: float = 0.0) -> np.ndarray:
    """1D weights on an extended z interval (length >= 1)."""
    length = int(length)
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if length == 1:
        seg = np.ones(1, dtype=np.float32)
    elif str(blend).lower() in ("uniform", "flat", "ones"):
        seg = np.ones(length, dtype=np.float32)
    else:
        seg = np.hanning(length).astype(np.float32)
    floor = float(max(0.0, floor))
    if floor > 0.0:
        peak = float(np.max(seg)) if seg.size else 1.0
        if peak > 0:
            seg = floor + (1.0 - floor) * (seg / peak)
    return seg


def shard_infer_z_range(
    z0: int,
    z1: int,
    depth: int,
    overlap: int,
) -> Tuple[int, int]:
    """Core [z0,z1) extended by overlap for inference."""
    z0, z1 = int(z0), int(z1)
    depth = int(depth)
    overlap = int(max(0, overlap))
    za = max(0, z0 - overlap)
    zb = min(depth, z1 + overlap)
    return za, zb


def build_shard_blend_weights(
    depth: int,
    shard_bounds: Sequence[Tuple[int, int]],
    overlap: int = 16,
    blend: str = "hann",
    weight_floor: float = 0.05,
) -> np.ndarray:
    """
    Per-shard z weights shaped [n_shards, depth], normalized so sum_i w_i(z) == 1.
    """
    depth = int(depth)
    bounds = list(shard_bounds)
    n = len(bounds)
    weights = np.zeros((n, depth), dtype=np.float64)
    for i, (z0, z1) in enumerate(bounds):
        z0, z1 = int(z0), int(z1)
        if z1 <= z0:
            continue
        overlap_i = int(max(0, overlap))
        za, zb = shard_infer_z_range(z0, z1, depth, overlap_i)
        seg_len = zb - za
        if overlap_i == 0:
            weights[i, z0:z1] = 1.0
        else:
            weights[i, za:zb] = _z_segment_profile(seg_len, blend=blend, floor=weight_floor)
    denom = np.maximum(weights.sum(axis=0), 1e-8)
    weights /= denom[np.newaxis, :]
    return weights


def infer_shard_ensemble_blend(
    shard_models: Sequence[Any],
    shard_cfgs: Sequence[Any],
    Xs: Sequence[np.ndarray],
    Xps: Sequence[np.ndarray],
    shard_bounds: Sequence[Tuple[int, int]],
    rel_err: float,
    overlap: int = 16,
    blend: str = "hann",
    weight_floor: float = 0.05,
    apply_final_post_process: bool = True,
) -> np.ndarray:
    """
    Fuse per-shard predictions with z-overlap + soft weights (Hann by default).

    Each expert infers on [z0-overlap, z1+overlap); overlapping z layers are
    blended so w_i(z) sum to 1. Fuses AI residual (part - Xp) then adds back Xp.
    """
    from bg_stage import run_bg_inference
    from config_io import _error_bounded_post_process

    lq = np.asarray(Xps[0], np.float32)
    depth = int(lq.shape[0])
    bounds = list(shard_bounds)
    overlap = int(max(0, overlap))

    if overlap == 0 and len(bounds) > 0:
        return infer_shard_ensemble(
            shard_models, shard_cfgs, Xs, Xps, bounds, rel_err
        )

    wmap = build_shard_blend_weights(
        depth, bounds, overlap=overlap, blend=blend, weight_floor=weight_floor
    )
    fused_delta = np.zeros_like(lq, dtype=np.float64)

    for i, (model, cfg, (z0, z1)) in enumerate(
        zip(shard_models, shard_cfgs, bounds)
    ):
        za, zb = shard_infer_z_range(z0, z1, depth, overlap)
        part = run_bg_inference(
            model,
            Xs,
            Xps,
            cfg,
            rel_err,
            z_start=int(za),
            z_stop=int(zb),
        )
        part = np.asarray(part, np.float32)
        delta = part - lq
        wi = wmap[i]
        for z in range(za, zb):
            if wi[z] > 0.0:
                fused_delta[z] += wi[z] * delta[z]

    x_hat = lq + fused_delta.astype(np.float32)
    if apply_final_post_process:
        x_hat = _error_bounded_post_process(
            x_enhanced=x_hat,
            x_prime=lq,
            absolute_error_bound=0.0,
            relative_error_bound=float(rel_err),
            verbose=False,
            a=1.0,
        )
    return np.asarray(x_hat, np.float32)


def infer_shard_ensemble(
    shard_models: Sequence[Any],
    shard_cfgs: Sequence[Any],
    Xs: Sequence[np.ndarray],
    Xps: Sequence[np.ndarray],
    shard_bounds: Sequence[Tuple[int, int]],
    rel_err: float,
) -> np.ndarray:
    """Hard z-splice (no overlap). Prefer infer_shard_ensemble_blend for seams."""
    from bg_stage import run_bg_inference

    lq = np.asarray(Xps[0], np.float32)
    x_hat = lq.copy()
    for model, cfg, (z0, z1) in zip(shard_models, shard_cfgs, shard_bounds):
        part = run_bg_inference(
            model,
            Xs,
            Xps,
            cfg,
            rel_err,
            z_start=int(z0),
            z_stop=int(z1),
        )
        x_hat[int(z0) : int(z1)] = np.asarray(part, np.float32)[int(z0) : int(z1)]
    return x_hat


def _build_shard_cfg(
    Xs_full,
    Xps_full,
    shard: Dict[str, Any],
    rel_err: float,
    max_train_time: float,
    epochs: int = 20,
    bg_lr: float = 1e-3,
    bg_patch: int = 512,
    bg_arch: str = "spatial",
):
    from experiment import build_bg_only_cfg

    z0, z1 = int(shard["z0"]), int(shard["z1"])
    bg_h = int(shard["bg_h"])
    sid = int(shard["shard_id"])
    cfg = build_bg_only_cfg(
        X_target=Xs_full[0],
        Xps=Xps_full,
        max_train_time=float(max_train_time),
        bg_h=bg_h,
        roi_h=4,
        epochs=int(epochs),
        steps_per_epoch=int(z1 - z0),
        bg_patch_size=int(bg_patch),
        bg_batch=1,
        lr=float(bg_lr),
    )
    cfg.bg_arch = str(bg_arch)
    cfg.bg_split_mode = "three"
    cfg.bg_split_bands = True
    cfg.bg_sample_mode = "sequential"
    cfg.bg_log_prefix = f"s{sid}_h{bg_h}"
    cfg.bg_input_norm = "global"
    cfg.bg_residual_norm = "global"
    cfg.rel_err = float(rel_err)
    return cfg


def train_shard_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Spawn worker: one z-shard on one GPU. evaluator=None for speed."""
    import os
    import time

    import torch

    gpu_id = int(payload["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    scripts_path = payload.get("scripts_path") or _HERE
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    from bg_stage import train_bg_only

    Xs_full = payload["Xs_full"]
    Xps_full = payload["Xps_full"]
    shard = payload["shard"]
    rel_err = float(payload["rel_err"])
    max_train_time = float(payload["max_train_time"])

    z0, z1 = int(shard["z0"]), int(shard["z1"])
    Xs_s, Xps_s = crop_multifield_zyx(Xs_full, Xps_full, z0, z1)
    cfg = _build_shard_cfg(
        Xs_full,
        Xps_full,
        shard,
        rel_err,
        max_train_time,
        epochs=int(payload.get("epochs", 20)),
        bg_lr=float(payload.get("bg_lr", 1e-3)),
        bg_patch=int(payload.get("bg_patch", 512)),
        bg_arch=str(payload.get("bg_arch", "spatial")),
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t0 = time.perf_counter()
    model, hist = train_bg_only(
        Xs=Xs_s,
        Xps=Xps_s,
        device=device,
        cfg=cfg,
        evaluator=None,
    )
    wall = time.perf_counter() - t0
    _ew = list(hist.get("epoch_wall") or [])
    return {
        "shard_id": int(shard["shard_id"]),
        "z0": z0,
        "z1": z1,
        "bg_h": int(shard["bg_h"]),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "wall_s": float(wall),
        "train_epochs": int(len(hist.get("loss") or [])),
        "epoch_wall_mean": float(np.mean(_ew)) if _ew else float("nan"),
        "epoch_wall_sum": float(np.sum(_ew)) if _ew else 0.0,
        "pure_train_s": float(sum(_ew)) if _ew else float("nan"),
    }


def train_shards_parallel(
    shards: Sequence[Dict[str, Any]],
    Xs_full: Sequence[np.ndarray],
    Xps_full: Sequence[np.ndarray],
    rel_err: float,
    max_train_time: float,
    epochs: int = 20,
    n_gpus: Optional[int] = None,
    scripts_path: str = "",
) -> List[Dict[str, Any]]:
    """Train all shards in parallel (spawn). Wall time ~ max(per-shard), not sum."""
    import multiprocessing as mp

    n_gpus = int(n_gpus or 4)
    ctx = mp.get_context("spawn")
    payloads = []
    for shard in shards:
        payloads.append(
            {
                "gpu_id": int(shard["shard_id"]) % max(n_gpus, 1),
                "shard": dict(shard),
                "Xs_full": [np.asarray(x, np.float32) for x in Xs_full],
                "Xps_full": [np.asarray(x, np.float32) for x in Xps_full],
                "rel_err": float(rel_err),
                "max_train_time": float(max_train_time),
                "epochs": int(epochs),
                "scripts_path": scripts_path,
            }
        )
    with ctx.Pool(processes=len(shards)) as pool:
        return list(pool.map(train_shard_worker, payloads))


def reload_shard_model(state_dict, cfg, shape=(512, 512, 512), n_fields=6, device=None):
    """Rebuild UNET_Model from cfg + weights."""
    import torch
    from siren_fft_backbone_model import UNET_Model

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    depth, height, width = (int(shape[0]), int(shape[1]), int(shape[2]))
    n_fields = int(n_fields)
    model = UNET_Model(
        n_fields=n_fields,
        K=7,
        D=int(depth),
        H=int(height),
        W=int(width),
        bg_hidden=int(cfg.bg_h),
        roi_hidden=4,
        bg_arch=str(getattr(cfg, "bg_arch", "spatial")),
        bg_split_bands=bool(getattr(cfg, "bg_split_bands", True)),
        bg_split_mode=getattr(cfg, "bg_split_mode", "three"),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
