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


def _gaussian_low_high_split_t(x, sigma_ratio=0.12):
    batch, channels, height, width = x.shape
    fy = torch.fft.fftfreq(height, d=1.0, device=x.device)
    fx = torch.fft.fftfreq(width, d=1.0, device=x.device)
    rr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    rr = rr / (rr.max() + 1e-8)

    sigma_ratio = float(max(sigma_ratio, 1e-4))
    low_mask = torch.exp(-(rr ** 2) / (2.0 * sigma_ratio ** 2)).view(1, 1, height, width)
    high_mask = 1.0 - low_mask

    xf = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
    x_low = torch.fft.ifft2(xf * low_mask, dim=(-2, -1), norm="ortho").real
    x_high = torch.fft.ifft2(xf * high_mask, dim=(-2, -1), norm="ortho").real
    return x_low.type_as(x), x_high.type_as(x)


def _gaussian_low_mid_high_split_t(x, sigma_low=0.08, sigma_mid=0.18):
    """
    x: [B, C, H, W]
    Returns x_low, x_mid, x_high with masks summing to ~1.
    """
    batch, channels, height, width = x.shape

    fy = torch.fft.fftfreq(height, d=1.0, device=x.device)
    fx = torch.fft.fftfreq(width, d=1.0, device=x.device)
    rr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    rr = rr / (rr.max() + 1e-8)

    sigma_low = float(max(sigma_low, 1e-4))
    sigma_mid = float(max(sigma_mid, sigma_low + 1e-4))

    low_mask = torch.exp(-(rr ** 2) / (2.0 * sigma_low ** 2)).view(1, 1, height, width)
    broad_mask = torch.exp(-(rr ** 2) / (2.0 * sigma_mid ** 2)).view(1, 1, height, width)

    mid_mask = (broad_mask - low_mask).clamp(min=0.0)
    high_mask = (1.0 - broad_mask).clamp(min=0.0)

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


