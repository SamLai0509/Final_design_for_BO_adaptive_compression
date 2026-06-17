"""Experiment helpers: BG training-config building and model-size budgeting.

Assemble ``TrainConfig`` objects for BG-only training and size the neural model
against a target compression ratio. Training / inference lives in ``bg_stage.py``.
"""

import sys
from pathlib import Path

import numpy as np

# Make sibling modules in this folder importable by bare name regardless of cwd.
_HERE = Path(__file__).resolve().parent.as_posix()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from train import TrainConfig


def build_bg_only_cfg(
    X_target,
    Xps,
    max_train_time,
    bg_h=7,
    roi_h=4,
    epochs=100,
    steps_per_epoch=512,
    bg_patch_size=512,
    bg_batch=1,
    lr=1e-3,
    bg_freq_weight=1.0,
    bg_freq_focus="low",
    bg_freq_boost=1.0,
    bg_freq_warmup_epochs=3,
    bg_fft_phase_weight=1.0,
    bg_field_norm="zscore",
):
    """Assemble a ``TrainConfig`` for BG-only training.

    Packs the BG hyper-parameters — model width ``bg_h``, patch size, learning rate,
    the frequency-loss weights / focus / warmup, and input normalisation — into the
    ``TrainConfig`` consumed by ``train_bg_only``.  ``X_target`` and ``Xps`` set the
    shape and ``n_fields``.  Returns the cfg.
    """
    cfg = TrainConfig(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        max_train_time=max_train_time,
        bg_patch_size=bg_patch_size,
        bg_batch=bg_batch,
        lr=lr,
        bg_h=bg_h,
        roi_h=roi_h,        # BG_only builds no ROI net (0 params); the arg is still required
    )
    x_prime = Xps[0]
    cfg.bg_field_norm = str(bg_field_norm)
    cfg.field_norm_eps = 1e-8
    cfg.res_mean = float(np.mean(X_target - x_prime))
    cfg.res_std = float(np.std(X_target - x_prime)) + 1e-8
    cfg.input_means = [float(np.mean(field)) for field in Xps]
    cfg.input_stds = [float(np.std(field)) + 1e-8 for field in Xps]
    cfg.input_mins = [float(np.min(field)) for field in Xps]
    cfg.input_maxs = [float(np.max(field)) for field in Xps]
    _res = X_target - x_prime
    cfg.res_min = float(np.min(_res))
    cfg.res_max = float(np.max(_res))
    cfg.bg_freq_mode = "fft"
    cfg.bg_freq_weight = bg_freq_weight
    cfg.bg_freq_focus = bg_freq_focus
    cfg.bg_freq_boost = bg_freq_boost
    cfg.bg_freq_warmup_epochs = bg_freq_warmup_epochs
    cfg.bg_fft_phase_weight = bg_fft_phase_weight
    return cfg


def compute_param_budget_bytes(original_bytes, sz3_bytes, target_total_cr):
    """Byte budget for the neural model under a target *total* compression ratio.

    Total stored bytes = ``sz3_bytes + model_bytes`` and ``CR = original_bytes / total``,
    so ``model_bytes = original_bytes / target_total_cr - sz3_bytes`` (clamped at 0).
    Combine with ``estimate_bg_model_param_bytes`` / ``pick_bg_h_under_budget`` to pick
    a model width that fits.
    """
    target_total_cr = float(target_total_cr)
    if target_total_cr <= 0:
        raise ValueError("target_total_cr must be positive.")
    budget = float(original_bytes) / target_total_cr - float(sz3_bytes)
    return max(0.0, float(budget))


def estimate_bg_model_param_bytes(
    n_fields,
    shape,
    bg_arch,
    bg_h,
    dtype_bytes=4,
    bg_use_se=False,
    bg_se_reduction=4,
    bg_feat_attn=False,
    bg_low_adapter=False,
    bg_mid_adapter=False,
    bg_high_adapter=False,
    bg_slab_k=7,
    bg_split_bands=True,
    bg_split_mode="three",
):
    """Instantiate the BG model for a given shape/width and report its size.

    Returns ``(num_trainable_params, param_bytes)`` where
    ``param_bytes = num_params * dtype_bytes`` (e.g. ``dtype_bytes=2`` for bf16 storage).
    Used to convert a parameter budget into a model width and to account for the
    model's contribution to the compression ratio.
    """
    from siren_fft_backbone_model import UNET_Model

    model = UNET_Model(
        n_fields=int(n_fields),
        K=7,
        D=int(shape[0]),
        H=int(shape[1]),
        W=int(shape[2]),
        bg_hidden=int(bg_h),
        bg_arch=str(bg_arch),
        bg_use_se=bool(bg_use_se),
        bg_se_reduction=int(bg_se_reduction),
        bg_feat_attn=bool(bg_feat_attn),
        bg_low_adapter=bool(bg_low_adapter),
        bg_mid_adapter=bool(bg_mid_adapter),
        bg_high_adapter=bool(bg_high_adapter),
        bg_slab_k=int(bg_slab_k),
        bg_split_bands=bool(bg_split_bands),
        bg_split_mode=bg_split_mode,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = int(num_params) * int(dtype_bytes)
    return int(num_params), int(param_bytes)
