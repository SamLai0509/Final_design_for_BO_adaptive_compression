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


def estimate_bg_model_param_bytes(n_fields, shape, bg_arch="spatial", bg_h=7, dtype_bytes=4, **legacy_kwargs):
    """Instantiate the BG model for a given width and report its size.

    Returns ``(num_trainable_params, param_bytes)`` with ``param_bytes = num_params *
    dtype_bytes`` (``dtype_bytes=2`` for the bf16 weights that are charged to the CR).
    ``shape`` and legacy keyword arguments are accepted for call-site compatibility.
    """
    from siren_fft_backbone_model import UNET_Model

    model = UNET_Model(n_fields=int(n_fields), bg_hidden=int(bg_h), bg_arch=str(bg_arch),
                       bg_split_bands=True, bg_split_mode="three")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(num_params), int(num_params) * int(dtype_bytes)
