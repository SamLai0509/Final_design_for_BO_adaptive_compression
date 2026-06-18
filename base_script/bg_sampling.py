"""BG patch sampling and Gaussian frequency-band splitting.

Depends on bg_normalize for the cfg/normalisation helpers."""

import copy
import time
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from config_io import _error_bounded_post_process, set_deterministic_seed
from frequency_losses import fft_mag_phase_loss_bg_t, masked_fft_mag_l1_t, masked_fft_phase_l1_t

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from Patch_data import (
    sample_bg_patches_multifield,
    sample_bg_center_slabs_multifield,
    sample_bg_slices_at_indices_multifield,
    sample_bg_center_slab_at_z_multifield,
    sample_bg_volume_slab_at_z_multifield,
)

from bg_normalize import _bg_arch_kind, normalize_bg_inputs, denormalize_bg_residual_tensor


# --- Cached radial Gaussian band masks -----------------------------------------
# Masks depend only on (H, W, sigma..., device); cache so the per-step split-band
# target computation reuses them instead of rebuilding every call.
_GAUSS3_CACHE = {}
_GAUSS2_CACHE = {}


def _gauss_masks_three(height, width, sigma_low, sigma_mid, device):
    sl = float(max(sigma_low, 1e-4))
    sm = float(max(sigma_mid, sl + 1e-4))
    key = (int(height), int(width), round(sl, 8), round(sm, 8), str(device))
    m = _GAUSS3_CACHE.get(key)
    if m is None:
        fy = torch.fft.fftfreq(height, d=1.0, device=device)
        fx = torch.fft.fftfreq(width, d=1.0, device=device)
        rr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        rr = rr / (rr.max() + 1e-8)
        low = torch.exp(-(rr ** 2) / (2.0 * sl ** 2)).view(1, 1, height, width)
        broad = torch.exp(-(rr ** 2) / (2.0 * sm ** 2)).view(1, 1, height, width)
        m = (low, (broad - low).clamp(min=0.0), (1.0 - broad).clamp(min=0.0))
        _GAUSS3_CACHE[key] = m
    return m


def _gauss_masks_two(height, width, sigma_ratio, device):
    sr = float(max(sigma_ratio, 1e-4))
    key = (int(height), int(width), round(sr, 8), str(device))
    m = _GAUSS2_CACHE.get(key)
    if m is None:
        fy = torch.fft.fftfreq(height, d=1.0, device=device)
        fx = torch.fft.fftfreq(width, d=1.0, device=device)
        rr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        rr = rr / (rr.max() + 1e-8)
        low = torch.exp(-(rr ** 2) / (2.0 * sr ** 2)).view(1, 1, height, width)
        m = (low, 1.0 - low)
        _GAUSS2_CACHE[key] = m
    return m


def _gaussian_low_high_split_t(x, sigma_ratio=0.12):
    _, _, height, width = x.shape
    low_mask, high_mask = _gauss_masks_two(height, width, sigma_ratio, x.device)
    xf = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
    x_low = torch.fft.ifft2(xf * low_mask, dim=(-2, -1), norm="ortho").real
    x_high = torch.fft.ifft2(xf * high_mask, dim=(-2, -1), norm="ortho").real
    return x_low.type_as(x), x_high.type_as(x)


def _gaussian_low_mid_high_split_t(x, sigma_low=0.08, sigma_mid=0.18):
    """
    x: [B, C, H, W]
    Returns x_low, x_mid, x_high with masks summing to ~1.
    """
    _, _, height, width = x.shape
    low_mask, mid_mask, high_mask = _gauss_masks_three(height, width, sigma_low, sigma_mid, x.device)
    xf = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
    x_low = torch.fft.ifft2(xf * low_mask, dim=(-2, -1), norm="ortho").real
    x_mid = torch.fft.ifft2(xf * mid_mask, dim=(-2, -1), norm="ortho").real
    x_high = torch.fft.ifft2(xf * high_mask, dim=(-2, -1), norm="ortho").real
    return x_low.type_as(x), x_mid.type_as(x), x_high.type_as(x)


def _sample_bg_training_batch(
    Xs_for_sampling,
    Xps,
    cfg,
    ep,
    step,
    *,
    seed,
    sampling_mask=None,
    sampling_min_frac=0.0,
):
    n = int(cfg.bg_batch)
    patch = int(cfg.bg_patch_size)
    mode = str(getattr(cfg, "bg_sample_mode", "random")).lower()
    kind = _bg_arch_kind(cfg)
    depth = int(Xs_for_sampling[0].shape[0])
    k_slab = int(getattr(cfg, "bg_slab_k", 7))
    half = k_slab // 2

    if kind == "slab2d":
        if mode in ("sequential", "all_slices", "deterministic"):
            global_step = int(ep) * int(cfg.steps_per_epoch) + int(step)
            zc = global_step % depth
            if zc < half or zc >= depth - half:
                zc = int(np.clip(zc, half, depth - half - 1))
            return sample_bg_center_slab_at_z_multifield(
                Xs_for_sampling, Xps, z_center=zc, K=k_slab, patch=patch, n=n
            )
        return sample_bg_center_slabs_multifield(
            Xs_for_sampling, Xps, n=n, K=k_slab, patch=patch, seed=seed
        )

    if kind == "res3d":
        if mode in ("sequential", "all_slices", "deterministic"):
            global_step = int(ep) * int(cfg.steps_per_epoch) + int(step)
            zc = global_step % depth
            if zc < half or zc >= depth - half:
                zc = int(np.clip(zc, half, depth - half - 1))
            return sample_bg_volume_slab_at_z_multifield(
                Xs_for_sampling, Xps, z_center=zc, K=k_slab, patch=patch, n=n
            )
        return sample_bg_volume_slab_at_z_multifield(
            Xs_for_sampling,
            Xps,
            z_center=int(np.random.default_rng(seed).integers(half, depth - half)),
            K=k_slab,
            patch=patch,
            n=n,
        )

    if mode in ("z_shard", "z_shards", "shard_parallel"):
        chunk = max(depth // max(n, 1), 1)
        z_off = int(step) % chunk
        z = np.array([(z_off + i * chunk) % depth for i in range(n)], dtype=np.int64)
        return sample_bg_slices_at_indices_multifield(
            Xs_for_sampling,
            Xps,
            z,
            patch=patch,
            mask=sampling_mask,
        )

    if mode in ("sequential", "all_slices", "deterministic"):
        global_step = int(ep) * int(cfg.steps_per_epoch) + int(step)
        z = np.array([(global_step + i) % depth for i in range(n)], dtype=np.int64)
        return sample_bg_slices_at_indices_multifield(
            Xs_for_sampling,
            Xps,
            z,
            patch=patch,
            mask=sampling_mask,
        )

    return sample_bg_patches_multifield(
        Xs=Xs_for_sampling,
        Xps=Xps,
        n=n,
        patch=patch,
        seed=seed,
        mask=sampling_mask,
        min_mask_frac=sampling_min_frac,
    )


def _to_gpu_volume(arr, device):
    """One (D, H, W) field -> contiguous, NaN-free float32 tensor resident on ``device``."""
    tensor = torch.as_tensor(np.asarray(arr), dtype=torch.float32, device=device)
    return torch.nan_to_num(tensor, nan=0.0)


def _sample_slice2d_gpu(Xs_gpu, Xps_gpu, cfg, ep, step, *, seed):
    """Device-resident equivalent of the slice2d path of ``_sample_bg_training_batch``.

    Generates the SAME (z, y0, x0) indices as the CPU sampler, then slices the
    resident volumes -- so the patches are identical but there is no per-step disk
    read, NumPy extraction, or host->device copy.  Supports random / sequential /
    z_shard modes with no ROI mask (the caller falls back to the CPU sampler
    otherwise).  Returns the same dict keys as the CPU sampler (tensors on device).
    """
    n = int(cfg.bg_batch)
    patch = int(cfg.bg_patch_size)
    mode = str(getattr(cfg, "bg_sample_mode", "random")).lower()
    n_fields = len(Xps_gpu)
    depth, height, width = Xps_gpu[0].shape
    y_max = max(height - patch + 1, 1)
    x_max = max(width - patch + 1, 1)

    if mode in ("z_shard", "z_shards", "shard_parallel"):
        chunk = max(depth // max(n, 1), 1)
        z_off = int(step) % chunk
        zs = [(z_off + i * chunk) % depth for i in range(n)]
        ys = [0] * n
        xs = [0] * n
    elif mode in ("sequential", "all_slices", "deterministic"):
        gstep = int(ep) * int(cfg.steps_per_epoch) + int(step)
        zs = [(gstep + i) % depth for i in range(n)]
        ys = [0] * n
        xs = [0] * n
    else:  # random -- matches sample_bg_patches_multifield's per-sample draws
        rng = np.random.default_rng(seed)
        zs, ys, xs = [], [], []
        for _ in range(n):
            zs.append(int(rng.integers(0, depth)))
            ys.append(int(rng.integers(0, y_max)))
            xs.append(int(rng.integers(0, x_max)))

    dev = Xps_gpu[0].device
    xp = torch.empty((n, n_fields, patch, patch), dtype=torch.float32, device=dev)
    x_target = torch.empty((n, 1, patch, patch), dtype=torch.float32, device=dev)
    for i in range(n):
        zi = int(zs[i]) % depth
        yi = int(ys[i]) % y_max
        xi = int(xs[i]) % x_max
        for f in range(n_fields):
            xp[i, f] = Xps_gpu[f][zi, yi:yi + patch, xi:xi + patch]
        x_target[i, 0] = Xs_gpu[0][zi, yi:yi + patch, xi:xi + patch]
        zs[i], ys[i], xs[i] = zi, yi, xi

    z_t = torch.tensor(zs, dtype=torch.float32, device=dev)
    y0_t = torch.tensor(ys, dtype=torch.float32, device=dev)
    x0_t = torch.tensor(xs, dtype=torch.float32, device=dev)
    return {"xp": xp, "x": x_target, "z": z_t, "y0": y0_t, "x0": x0_t}
