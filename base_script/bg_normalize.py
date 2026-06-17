"""BG config/normalisation helpers (cfg readers + tensor (de)normalisation).

Leaf module: depends only on numpy/torch. Used by bg_sampling and the core
bg_stage training/inference."""

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


def _bg_arch_kind(cfg):
    arch = str(getattr(cfg, "bg_arch", "spatial")).lower()
    if arch in ("res3d_unet", "res3d"):
        return "res3d"
    if arch in ("slab2d", "slab2d_unet", "slab_2d"):
        return "slab2d"
    return "slice2d"


def _bg_norm_mode(cfg):
    return str(getattr(cfg, "bg_field_norm", "zscore")).lower()


def _bg_norm_eps(cfg):
    return float(getattr(cfg, "field_norm_eps", 1e-8))


def _bg_input_norm_mode(cfg):
    """global: volume-wide input_means/stds; revin_slice: per-patch/slice mean/std over H,W."""
    return str(getattr(cfg, "bg_input_norm", "global")).lower()


def _bg_residual_norm_mode(cfg):
    """global: res/res_std; revin_slice: per-patch/slice (res-mu)/std over H,W."""
    return str(getattr(cfg, "bg_residual_norm", "global")).lower()


def _revin_mu_sig(x, eps):
    mu = x.mean(dim=(2, 3), keepdim=True)
    sig = x.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(float(eps))
    return mu, sig


def _build_input_norm_tensors(cfg, device, n_fields):
    mean_t = torch.as_tensor(cfg.input_means, dtype=torch.float32, device=device).view(
        1, n_fields, 1, 1
    )
    std_t = torch.as_tensor(cfg.input_stds, dtype=torch.float32, device=device).view(
        1, n_fields, 1, 1
    )
    mins = getattr(cfg, "input_mins", None)
    maxs = getattr(cfg, "input_maxs", None)
    if mins is None or maxs is None:
        min_t = torch.zeros_like(mean_t)
        max_t = torch.ones_like(mean_t)
    else:
        min_t = torch.as_tensor(mins, dtype=torch.float32, device=device).view(
            1, n_fields, 1, 1
        )
        max_t = torch.as_tensor(maxs, dtype=torch.float32, device=device).view(
            1, n_fields, 1, 1
        )
    return mean_t, std_t, min_t, max_t


def normalize_bg_inputs(x, cfg, mean_t, std_t, min_t, max_t):
    """x: [B, C, H, W]; broadcasts against mean_t/std_t/min_t/max_t."""
    if _bg_input_norm_mode(cfg) == "revin_slice":
        eps = _bg_norm_eps(cfg)
        mu, sig = _revin_mu_sig(x, eps)
        return (x - mu) / sig
    mode = _bg_norm_mode(cfg)
    eps = _bg_norm_eps(cfg)
    if mode == "zscore":
        return (x - mean_t) / std_t
    if mode in ("minmax01", "minmax_01", "mm01"):
        return (x - min_t) / (max_t - min_t + eps)
    if mode in ("minmax11", "minmax_11", "mm11"):
        mm = (x - min_t) / (max_t - min_t + eps)
        return 2.0 * mm - 1.0
    raise ValueError(f"Unknown bg_field_norm: {mode}")


def normalize_bg_residual_tensor(res, cfg, return_revin_stats=False):
    if _bg_residual_norm_mode(cfg) == "revin_slice":
        eps = _bg_norm_eps(cfg)
        mu, sig = _revin_mu_sig(res, eps)
        out = (res - mu) / sig
        if return_revin_stats:
            return out, mu, sig
        return out
    mode = _bg_norm_mode(cfg)
    eps = _bg_norm_eps(cfg)
    r_lo = float(getattr(cfg, "res_min", 0.0))
    r_hi = float(getattr(cfg, "res_max", 1.0))
    span = r_hi - r_lo + eps
    if mode == "zscore":
        out = res / float(cfg.res_std)
        if return_revin_stats:
            return out, None, None
        return out
    if mode in ("minmax01", "minmax_01", "mm01"):
        return (res - r_lo) / span
    if mode in ("minmax11", "minmax_11", "mm11"):
        mm = (res - r_lo) / span
        return 2.0 * mm - 1.0
    raise ValueError(f"Unknown bg_field_norm: {mode}")


def denormalize_bg_residual_tensor(pred_norm, cfg, revin_mu=None, revin_sig=None):
    if _bg_residual_norm_mode(cfg) == "revin_slice":
        if revin_mu is None or revin_sig is None:
            raise ValueError(
                "revin_slice residual denorm requires revin_mu and revin_sig "
                "(compute from raw residual slice/patch)"
            )
        return pred_norm * revin_sig + revin_mu
    mode = _bg_norm_mode(cfg)
    eps = _bg_norm_eps(cfg)
    r_lo = float(getattr(cfg, "res_min", 0.0))
    r_hi = float(getattr(cfg, "res_max", 1.0))
    span = r_hi - r_lo + eps
    if mode == "zscore":
        return pred_norm * float(cfg.res_std)
    if mode in ("minmax01", "minmax_01", "mm01"):
        return pred_norm * span + r_lo
    if mode in ("minmax11", "minmax_11", "mm11"):
        return (pred_norm + 1.0) * 0.5 * span + r_lo
    raise ValueError(f"Unknown bg_field_norm: {mode}")


def _normalize_bg_batch(bg_xs_t, cfg, mean_t, std_t, min_t, max_t):
    kind = _bg_arch_kind(cfg)
    if kind == "res3d":
        # [B, F, K, H, W]: same per-field stats across K (do not flatten F*K)
        mean5 = mean_t.unsqueeze(2)
        std5 = std_t.unsqueeze(2)
        min5 = min_t.unsqueeze(2)
        max5 = max_t.unsqueeze(2)
        mode = _bg_norm_mode(cfg)
        eps = _bg_norm_eps(cfg)
        if mode == "zscore":
            return (bg_xs_t - mean5) / std5
        if mode in ("minmax01", "minmax_01", "mm01"):
            return (bg_xs_t - min5) / (max5 - min5 + eps)
        if mode in ("minmax11", "minmax_11", "mm11"):
            mm = (bg_xs_t - min5) / (max5 - min5 + eps)
            return 2.0 * mm - 1.0
        raise ValueError(f"Unknown bg_field_norm: {mode}")
    if kind == "slab2d":
        k = int(getattr(cfg, "bg_slab_k", 7))
        mean_s = mean_t.repeat_interleave(k, dim=1)
        std_s = std_t.repeat_interleave(k, dim=1)
        min_s = min_t.repeat_interleave(k, dim=1)
        max_s = max_t.repeat_interleave(k, dim=1)
        return normalize_bg_inputs(bg_xs_t, cfg, mean_s, std_s, min_s, max_s)
    return normalize_bg_inputs(bg_xs_t, cfg, mean_t, std_t, min_t, max_t)


