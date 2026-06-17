"""Background (BG) residual-model training and inference.

The BG model is a small CNN that refines a lossy *base* reconstruction ``Xps[0]``
(e.g. the SZ3 / SPERR decompression of the target field) by predicting the residual
``Xs[0] - Xps[0]``; optional auxiliary fields are fed as extra input channels.
Training combines a spatial (residual) loss with the frequency-domain loss in
``frequency_losses.py``, plus optional split-band supervision.

Public API:
  * ``train_bg_only(Xs, Xps, device, cfg, evaluator=...)`` -> (model, history)
  * ``run_bg_inference(model, Xs, Xps, cfg, rel_err)``     -> reconstructed volume

Everything else is a private helper (input/residual normalisation, patch sampling,
Gaussian band splitting, ...).

The normalisation and sampling helpers now live in bg_normalize.py / bg_sampling.py
and are imported below (re-exported for backward compatibility)."""

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
)

from bg_normalize import (
    _bg_arch_kind, _bg_norm_mode, _bg_norm_eps, _bg_input_norm_mode, _bg_residual_norm_mode,
    _revin_mu_sig, _build_input_norm_tensors, normalize_bg_inputs, normalize_bg_residual_tensor,
    denormalize_bg_residual_tensor, _normalize_bg_batch,
)
from bg_sampling import (
    _gaussian_low_high_split_t, _gaussian_low_mid_high_split_t,
    _sample_bg_training_batch,
)


def unwrap_bg_model(model):
    """Return the underlying UNET_Model (strip DataParallel / train adapter)."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        inner = model.module
        if hasattr(inner, "core"):
            return inner.core
        return inner
    if hasattr(model, "core"):
        return model.core
    return model


class _BGTrainParallelAdapter(nn.Module):
    """Thin wrapper so nn.DataParallel can run bg_forward_split on batch dim 0."""

    def __init__(self, core, split_mode=None):
        super().__init__()
        self.core = core
        self.split_mode = split_mode

    def forward(self, xp_norm, z_idx, y0, x0, rel_err_scalar):
        rel = float(rel_err_scalar.reshape(-1)[0].item())
        if self.split_mode == "three":
            pred_low, pred_mid, pred_high, pred = self.core.bg_forward_split(
                xp_norm,
                z_idx,
                y0,
                x0,
                rel_err=rel,
            )
            return pred, pred_low, pred_mid, pred_high
        else:
            pred = self.core.bg_forward(xp_norm, z_idx, y0, x0, rel_err=rel)
            empty = xp_norm.new_empty(0)
            return pred, empty, empty, empty


def _maybe_wrap_dataparallel(model, cfg, device):
    use_dp = bool(getattr(cfg, "bg_data_parallel", False))
    use_ddp = bool(getattr(cfg, "bg_ddp", False))
    n_gpu = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    split_mode = getattr(cfg, "bg_split_mode", None)
    
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        adapter = _BGTrainParallelAdapter(model, split_mode).to(device)
        dp_model = nn.parallel.DistributedDataParallel(
            adapter, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
        )
        return dp_model, True

    if not use_dp or n_gpu <= 1:
        return model, False

    n_dp = min(int(cfg.bg_batch), n_gpu)
    if int(cfg.bg_batch) < n_dp:
        cfg.bg_batch = int(n_dp)
    adapter = _BGTrainParallelAdapter(model, split_mode).to(device)
    device_ids = list(range(n_dp))
    dp_model = nn.DataParallel(adapter, device_ids=device_ids)
    return dp_model, True


def _forward_bg_outputs(model, x_norm, z_idx, y0, x0, split_mode=None, rel_err=None):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        rel_t = torch.full(
            (x_norm.shape[0],),
            float(rel_err if rel_err is not None else 0.0),
            device=x_norm.device,
            dtype=x_norm.dtype,
        )
        pred, pred_low, pred_mid, pred_high = model(x_norm, z_idx, y0, x0, rel_t)
        return {"pred": pred, "low": pred_low, "mid": pred_mid, "high": pred_high}

    core = unwrap_bg_model(model)

    if split_mode == "three":
        pred_low, pred_mid, pred_high, pred = core.bg_forward_split(
            x_norm, z_idx, y0, x0, rel_err=rel_err
        )
        return {"pred": pred, "low": pred_low, "mid": pred_mid, "high": pred_high}

    elif split_mode == "two":
        pred_low, pred_high, pred = core.bg_forward_split(
            x_norm, z_idx, y0, x0, rel_err=rel_err
        )
        return {"pred": pred, "low": pred_low, "mid": None, "high": pred_high}

    else:
        pred = core.bg_forward(x_norm, z_idx, y0, x0, rel_err=rel_err)
        return {"pred": pred, "low": None, "mid": None, "high": None}


def run_bg_inference(
    model,
    Xs,
    Xps,
    cfg,
    rel_err,
    return_components=False,
    z_start=None,
    z_stop=None,
):
    """Reconstruct the full volume with a trained BG model.

    Runs the model over every depth slice of the base reconstruction ``Xps[0]`` (plus
    aux channels), adds the predicted residual back, and applies an error-bounded clamp
    that keeps the output within ``rel_err * range(Xps[0])`` of the base — so the
    refinement cannot exceed the base compressor's error envelope.  ``z_start`` /
    ``z_stop`` restrict the reconstructed slice range.

    Returns the reconstructed ``(D, H, W)`` volume (or per-band components if
    ``return_components`` is set).
    """
    model_was_training = model.training
    model.eval()

    model_device = next(model.parameters()).device
    gt_target = Xs[0]
    lq_target = Xps[0]
    depth, height, width = gt_target.shape
    z_lo = 0 if z_start is None else int(z_start)
    z_hi = depth if z_stop is None else int(z_stop)
    z_lo = int(np.clip(z_lo, 0, depth))
    z_hi = int(np.clip(z_hi, 0, depth))
    if z_hi < z_lo:
        z_hi = z_lo
    n_fields = len(Xps)
    patch = int(cfg.bg_patch_size)
    k_slab = int(getattr(cfg, "bg_slab_k", 7))
    half = k_slab // 2
    arch_kind = _bg_arch_kind(cfg)

    ai_contribution = np.zeros_like(lq_target, dtype=np.float32)

    mean_t, std_t, min_t, max_t = _build_input_norm_tensors(cfg, model_device, n_fields)

    with torch.no_grad():
        for z in range(z_lo, z_hi):
            y0 = 0
            x0 = 0

            if arch_kind == "slab2d":
                zc = int(np.clip(z, half, depth - half - 1))
                slab_np = sample_bg_center_slab_at_z_multifield(
                    Xs, Xps, z_center=zc, K=k_slab, patch=patch, y0=y0, x0=x0, n=1
                )["xp"]
                slab_t = torch.from_numpy(slab_np).to(model_device)
                slab_norm = _normalize_bg_batch(slab_t, cfg, mean_t, std_t, min_t, max_t)
                pred_norm = model.bg_forward(
                    slab_norm,
                    torch.tensor([zc], device=model_device).float(),
                    torch.tensor([0], device=model_device).float(),
                    torch.tensor([0], device=model_device).float(),
                    rel_err=rel_err,
                )
            else:
                slice_data = np.stack([field[z] for field in Xps], axis=0).astype(np.float32)
                slice_t = torch.from_numpy(slice_data).unsqueeze(0).to(model_device)
                slice_norm = normalize_bg_inputs(slice_t, cfg, mean_t, std_t, min_t, max_t)
                pred_norm = model.bg_forward(
                    slice_norm,
                    torch.tensor([z], device=model_device).float(),
                    torch.tensor([0], device=model_device).float(),
                    torch.tensor([0], device=model_device).float(),
                    rel_err=rel_err,
                )

            if _bg_residual_norm_mode(cfg) == "revin_slice":
                res_raw = torch.from_numpy(
                    (gt_target[z] - lq_target[z]).astype(np.float32)
                ).to(model_device).view(1, 1, height, width)
                r_mu, r_sig = _revin_mu_sig(res_raw, _bg_norm_eps(cfg))
                pred = denormalize_bg_residual_tensor(
                    pred_norm, cfg, revin_mu=r_mu, revin_sig=r_sig
                ).cpu().numpy()[0, 0]
            else:
                pred = denormalize_bg_residual_tensor(pred_norm, cfg).cpu().numpy()[0, 0]
            ai_contribution[z] = pred

    x_hat_raw = lq_target + ai_contribution
    x_hat_raw = _error_bounded_post_process(
        x_enhanced=x_hat_raw,
        x_prime=lq_target,
        absolute_error_bound=0.0,
        relative_error_bound=rel_err,
        verbose=False,
        a=1.0,
    )

    if model_was_training:
        model.train()

    if return_components:
        return {
            "x_hat": x_hat_raw,
            "xp": np.asarray(lq_target, dtype=np.float32),
            "ai": ai_contribution,
        }
    return x_hat_raw


def train_bg_only(
    Xs,
    Xps,
    device,
    cfg,
    evaluator=None,
    init_state_dict=None,
    init_optimizer_state=None,
):
    """Train the BG residual model on one (Xs, Xps) pair.

    Args:
        Xs:  ``[target_field]`` (+ optional aux fields), each ``(D, H, W)`` ground truth.
        Xps: ``[base_recon]`` (+ the same aux fields).  ``base_recon`` is the lossy
             reconstruction the model refines; ``n_fields = len(Xps)``.
        device, cfg: torch device and a ``TrainConfig`` (see ``build_bg_only_cfg``).
        evaluator: optional ``callable(model) -> (psnr, ...)`` run at each epoch end;
            the best-PSNR weights are restored before returning.
        init_state_dict / init_optimizer_state: optional warm starts.

    Trains slice-by-slice (each depth slice is one 2-D patch) under a wall-clock budget
    (``cfg.max_train_time``) and an epoch cap (``cfg.epochs``), in bf16 AMP.  The loss is
    the spatial residual loss + the frequency loss (enabled after
    ``cfg.bg_freq_warmup_epochs``) + optional split-band terms.  An opt-in
    slope/patience early-stop (``cfg.bg_early_stop``) can end training once the loss
    flattens.

    Returns:
        (model, history) — ``model`` carries the best-PSNR weights; ``history`` holds
        per-epoch loss / psnr / time lists.
    """
    from siren_fft_backbone_model import UNET_Model

    split_mode = getattr(cfg, "bg_split_mode", None)
    use_split_bands = split_mode in {"two", "three"}
    use_three_bands = split_mode == "three"

    seed = getattr(cfg, "seed", 42)
    set_deterministic_seed(seed)

    n_fields = len(Xps)
    depth, height, width = Xs[0].shape

    true_residuals = Xs[0] - Xps[0]
    Xs_for_sampling = [true_residuals] + Xs[1:]

    model = UNET_Model(
        n_fields=n_fields,
        K=7,
        D=depth,
        H=height,
        W=width,
        bg_hidden=cfg.bg_h,
        bg_arch=getattr(cfg, "bg_arch", "spatial"),
        bg_split_bands=bool(getattr(cfg, "bg_split_bands", False)),
        bg_split_mode=getattr(cfg, "bg_split_mode", None),
        bg_use_se=bool(getattr(cfg, "bg_use_se", False)),
        bg_se_reduction=int(getattr(cfg, "bg_se_reduction", 4)),
        bg_feat_attn=bool(getattr(cfg, "bg_feat_attn", False)),
        bg_low_adapter=bool(getattr(cfg, "bg_low_adapter", False)),
        bg_mid_adapter=bool(getattr(cfg, "bg_mid_adapter", False)),
        bg_high_adapter=bool(getattr(cfg, "bg_high_adapter", False)),
        bg_slab_k=int(getattr(cfg, "bg_slab_k", 7)),
    ).to(device)

    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=True)

    model, _dp_wrapped = _maybe_wrap_dataparallel(model, cfg, device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.lr)
    
    if init_optimizer_state is not None:
        optimizer.load_state_dict(init_optimizer_state)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.epochs * cfg.steps_per_epoch),
    )
    mse_loss = nn.MSELoss()

    # ---- AMP / BF16 switch (default: bf16 on) ----
    use_amp = bool(getattr(cfg, "amp", True))
    amp_dtype = str(getattr(cfg, "amp_dtype", "bf16")).lower()
    if use_amp and amp_dtype in ("bf16", "bfloat16"):
        autocast_dtype = torch.bfloat16
    elif use_amp and amp_dtype in ("fp16", "float16"):
        autocast_dtype = torch.float16
    else:
        use_amp = False
        autocast_dtype = torch.float32  # placeholder

    sampling_mask = getattr(cfg, "bg_sampling_mask", None)
    sampling_min_frac = float(getattr(cfg, "bg_sampling_min_frac", 0.0))

    history = {
        "epoch": [],
        "loss": [],
        "psnr": [],
        "time": [],
        "max_err": [],
        "epoch_wall": [],
    }

    best_model_weights = None
    best_psnr = -float("inf")

    _log_pfx = str(getattr(cfg, "bg_log_prefix", "") or "").strip()
    if _log_pfx:
        _log_pfx = _log_pfx + " "

    if evaluator is not None:
        base_eval_res = evaluator(unwrap_bg_model(model))
        if isinstance(base_eval_res, tuple):
            if len(base_eval_res) >= 2:
                base_psnr, base_max_err = base_eval_res[:2]
            else:
                base_psnr = base_eval_res[0]
                base_max_err = None
        else:
            base_psnr = base_eval_res
            base_max_err = None

        history["psnr"].append((0, base_psnr))
        history["time"].append(0.0)
        history["epoch"].append(0)
        if base_max_err is not None:
            history["max_err"].append(base_max_err)

        best_psnr = base_psnr
        best_model_weights = copy.deepcopy(unwrap_bg_model(model).state_dict())

        err_str = f"{base_max_err:.1f}" if base_max_err is not None else "N/A"
        print(f"{_log_pfx}[Init] Epoch   0 | Global PSNR: {base_psnr:.2f} dB | MaxErr: {err_str}")
    else:
        print(f"{_log_pfx}[Init] evaluator=None, train only without PSNR tracking.")

    amp_str = "off" if not use_amp else ("bf16" if autocast_dtype is torch.bfloat16 else "fp16")
    _sample_mode = str(getattr(cfg, "bg_sample_mode", "random")).lower()
    _dp_flag = bool(getattr(cfg, "bg_data_parallel", False)) and torch.cuda.device_count() > 1
    print(
        f"{_log_pfx}[plan] pure_train_budget={float(cfg.max_train_time):.2f}s | "
        f"epochs_cap={int(cfg.epochs)} | steps/epoch={int(cfg.steps_per_epoch)} | "
        f"patch={int(cfg.bg_patch_size)} | batch={int(cfg.bg_batch)} | "
        f"sample={_sample_mode} | data_parallel={_dp_flag} | amp={amp_str}"
    )
    # Diagnostic: print the early-stop config train_bg_only ACTUALLY received, so a
    # stale module / unset flag is obvious instead of silently running to budget.
    if bool(getattr(cfg, "bg_early_stop", False)):
        print(
            f"{_log_pfx}[early-stop] ENABLED v2 | metric={getattr(cfg, 'bg_es_metric', '?')} "
            f"| min_drop={getattr(cfg, 'bg_es_min_drop', '?')} patience={getattr(cfg, 'bg_es_patience', '?')} "
            f"| freq_warmup={getattr(cfg, 'bg_freq_warmup_epochs', 3)} freq_weight={getattr(cfg, 'bg_freq_weight', 0.0)}"
        )
    else:
        print(f"{_log_pfx}[early-stop] DISABLED (cfg.bg_early_stop is False/unset)")

    t_start_train = time.perf_counter()
    eval_time_total = 0.0
    stop_training = False

    mean_t, std_t, min_t, max_t = _build_input_norm_tensors(cfg, device, n_fields)

    for ep in range(cfg.epochs):
        if stop_training:
            break

        epoch_start = time.perf_counter()
        model.train()
        epoch_losses = []
        epoch_freq_losses = []
        epoch_low_losses = []
        epoch_mid_losses = []
        epoch_high_losses = []

        for step in range(cfg.steps_per_epoch):
            current_pure_time = time.perf_counter() - t_start_train - eval_time_total
            
            stop_flag = int(current_pure_time >= cfg.max_train_time)
            if getattr(cfg, "bg_ddp", False):
                import torch.distributed as dist
                if dist.is_initialized():
                    t_flag = torch.tensor([stop_flag], device=device, dtype=torch.int32)
                    dist.all_reduce(t_flag, op=dist.ReduceOp.MAX)
                    stop_flag = t_flag.item()
                    
            if stop_flag > 0:
                stop_training = True
                break

            step_seed = seed + ep * 10000 + step

            bg_dict = _sample_bg_training_batch(
                Xs_for_sampling,
                Xps,
                cfg,
                ep,
                step,
                seed=step_seed,
                sampling_mask=sampling_mask,
                sampling_min_frac=sampling_min_frac,
            )

            bg_xs_t = torch.from_numpy(bg_dict["xp"]).to(device)
            bg_ys_norm = normalize_bg_residual_tensor(
                torch.from_numpy(bg_dict["x"]).to(device), cfg
            )
            z_key = "z" if "z" in bg_dict else "zc"
            bg_z_idx = torch.from_numpy(bg_dict[z_key]).to(device)
            bg_y0 = torch.from_numpy(bg_dict["y0"]).to(device)
            bg_x0 = torch.from_numpy(bg_dict["x0"]).to(device)

            bg_xs_norm = _normalize_bg_batch(bg_xs_t, cfg, mean_t, std_t, min_t, max_t)

            # ---- AMP autocast: wraps the forward pass + losses ----
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                student_out = _forward_bg_outputs(
                    model,
                    bg_xs_norm,
                    bg_z_idx,
                    bg_y0,
                    bg_x0,
                    split_mode=split_mode if use_split_bands else None,
                    rel_err=getattr(cfg, "rel_err", None),
                )
                bg_pred = student_out["pred"]

                if use_split_bands:
                    if use_three_bands:
                        tgt_low, tgt_mid, tgt_high = _gaussian_low_mid_high_split_t(
                            bg_ys_norm,
                            sigma_low=float(getattr(cfg, "bg_sigma_low", 0.08)),
                            sigma_mid=float(getattr(cfg, "bg_sigma_mid", 0.18)),
                        )
                        loss_low = mse_loss(student_out["low"], tgt_low)
                        loss_mid = mse_loss(student_out["mid"], tgt_mid)
                        loss_high = mse_loss(student_out["high"], tgt_high)
                    else:
                        tgt_low, tgt_high = _gaussian_low_high_split_t(
                            bg_ys_norm,
                            sigma_ratio=float(getattr(cfg, "bg_split_sigma", 0.12)),
                        )
                        loss_low = mse_loss(student_out["low"], tgt_low)
                        loss_mid = bg_ys_norm.new_tensor(0.0)
                        loss_high = mse_loss(student_out["high"], tgt_high)
                else:
                    loss_low = bg_ys_norm.new_tensor(0.0)
                    loss_mid = bg_ys_norm.new_tensor(0.0)
                    loss_high = bg_ys_norm.new_tensor(0.0)

                if "pixel_mask" in bg_dict:
                    pixel_mask_t = torch.from_numpy(bg_dict["pixel_mask"]).to(device)
                    mse_per_pixel = (bg_pred - bg_ys_norm) ** 2
                    loss_bg = (mse_per_pixel * pixel_mask_t).sum() / pixel_mask_t.sum().clamp_min(1e-8)
                else:
                    loss_bg = mse_loss(bg_pred, bg_ys_norm)
                freq_focus = getattr(cfg, "bg_freq_focus", "low")
                freq_boost = float(getattr(cfg, "bg_freq_boost", 1.0))
                freq_warmup = int(getattr(cfg, "bg_freq_warmup_epochs", 3))
                freq_weight = 0.0 if ep < freq_warmup else float(getattr(cfg, "bg_freq_weight", 0.0))

                loss_freq, _, _ = fft_mag_phase_loss_bg_t(
                    bg_pred,
                    bg_ys_norm,
                    focus=freq_focus,
                    boost=freq_boost,
                    mag_weight=1.0,
                    phase_weight=float(getattr(cfg, "bg_fft_phase_weight", 1.0)),
                )

                loss = loss_bg + freq_weight * loss_freq

                if use_split_bands:
                    low_weight = float(getattr(cfg, "bg_low_weight", 0.2))
                    mid_weight = float(getattr(cfg, "bg_mid_weight", 0.5))
                    high_weight = float(getattr(cfg, "bg_high_weight", 1.0))
                    loss = loss + low_weight * loss_low + mid_weight * loss_mid + high_weight * loss_high

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(float(loss.item()))
            epoch_freq_losses.append(float(loss_freq.item()))
            epoch_low_losses.append(float(loss_low.item()))
            epoch_mid_losses.append(float(loss_mid.item()))
            epoch_high_losses.append(float(loss_high.item()))

        if evaluator is not None:
            epoch_train_wall = time.perf_counter() - epoch_start
            t_eval_start = time.perf_counter()
            eval_res = evaluator(unwrap_bg_model(model))

            if isinstance(eval_res, tuple):
                if len(eval_res) >= 2:
                    cur_p, cur_max_err = eval_res[:2]
                else:
                    cur_p = eval_res[0]
                    cur_max_err = None
            else:
                cur_p = eval_res
                cur_max_err = None

            eval_time_total += time.perf_counter() - t_eval_start
            cum_train_time = time.perf_counter() - t_start_train - eval_time_total

            history["epoch"].append(ep + 1)
            history["loss"].append(float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else 0.0)
            history["psnr"].append((ep + 1, cur_p))
            history["time"].append(cum_train_time)
            history["epoch_wall"].append(float(epoch_train_wall))
            if cur_max_err is not None:
                history["max_err"].append(cur_max_err)

            err_str = f"{cur_max_err:.1f}" if cur_max_err is not None else "N/A"
            freq_str = f" | Freq: {np.mean(epoch_freq_losses):.6f}" if len(epoch_freq_losses) > 0 else ""
            low_str = (
                f" | Low: {np.mean(epoch_low_losses):.6f}"
                if use_split_bands and len(epoch_low_losses) > 0
                else ""
            )
            mid_str = (
                f" | Mid: {np.mean(epoch_mid_losses):.6f}"
                if use_split_bands and len(epoch_mid_losses) > 0
                else ""
            )
            high_str = (
                f" | High: {np.mean(epoch_high_losses):.6f}"
                if use_split_bands and len(epoch_high_losses) > 0
                else ""
            )

            print(
                f"{_log_pfx}Epoch {ep + 1:3d} [BG] | train_wall={epoch_train_wall:.2f}s"
                f" | Loss: {history['loss'][-1]:.6f}"
                f"{freq_str}{low_str}{mid_str}{high_str} | Global: {cur_p:.2f} dB | MaxErr: {err_str}",
                end="",
            )

            if cur_p > best_psnr:
                best_psnr = cur_p
                best_model_weights = copy.deepcopy(unwrap_bg_model(model).state_dict())
                print("  [New Best!]")
            else:
                print()

            if ep == 0:
                _ep0_pure = time.perf_counter() - t_start_train - eval_time_total
                print(
                    f"{_log_pfx}[timing] first_epoch_pure_train≈{_ep0_pure:.3f}s "
                    f"(excludes this epoch's end-of-epoch eval)"
                )
        else:
            if len(epoch_losses) <= 0:
                continue

            epoch_train_wall = time.perf_counter() - epoch_start
            mean_loss = float(np.mean(epoch_losses))
            cum_train_time = time.perf_counter() - t_start_train - eval_time_total

            history["epoch"].append(ep + 1)
            history["loss"].append(mean_loss)
            history["epoch_wall"].append(float(epoch_train_wall))
            history["time"].append(float(cum_train_time))

            freq_str = (
                f" | Freq: {np.mean(epoch_freq_losses):.6f}"
                if len(epoch_freq_losses) > 0
                else ""
            )
            low_str = (
                f" | Low: {np.mean(epoch_low_losses):.6f}"
                if use_split_bands and len(epoch_low_losses) > 0
                else ""
            )
            mid_str = (
                f" | Mid: {np.mean(epoch_mid_losses):.6f}"
                if use_split_bands and len(epoch_mid_losses) > 0
                else ""
            )
            high_str = (
                f" | High: {np.mean(epoch_high_losses):.6f}"
                if use_split_bands and len(epoch_high_losses) > 0
                else ""
            )

            print(
                f"{_log_pfx}Epoch {ep + 1:3d} [BG] | train_wall={epoch_train_wall:.2f}s"
                f" | Loss: {mean_loss:.6f}"
                f"{freq_str}{low_str}{mid_str}{high_str}"
            )

            if ep == 0:
                print(
                    f"{_log_pfx}[timing] first_epoch_train_wall={epoch_train_wall:.3f}s "
                    f"(steps={int(cfg.steps_per_epoch)}, no eval)"
                )

        # ---- Slope-based early stop (opt-in: cfg.bg_early_stop) --------------
        # Fit a line to the last `bg_es_window` epochs of the chosen metric and
        # stop when the slope says training has flattened or reversed:
        #   metric="psnr": stop when PSNR slope < bg_es_min_slope (dB/epoch)
        #   metric="loss": stop when mean-normalized loss slope > -bg_es_min_slope
        if bool(getattr(cfg, "bg_early_stop", False)) and not stop_training:
            es_window = max(3, int(getattr(cfg, "bg_es_window", 5)))
            es_metric = str(
                getattr(cfg, "bg_es_metric", "psnr" if evaluator is not None else "loss")
            ).lower()
            if es_metric == "psnr" and evaluator is not None:
                vals = [p[1] if isinstance(p, tuple) else p for p in history["psnr"]]
                min_slope = float(getattr(cfg, "bg_es_min_slope", 0.02))  # dB/epoch
                if len(vals) >= es_window:
                    y = np.asarray(vals[-es_window:], dtype=np.float64)
                    slope = float(np.polyfit(np.arange(es_window), y, 1)[0])
                    if slope < min_slope:
                        stop_training = True
                        print(
                            f"{_log_pfx}[early-stop] PSNR slope {slope:+.4f} dB/ep < "
                            f"{min_slope:g} over last {es_window} epochs -> stop at epoch {ep + 1}"
                        )
            else:
                # The frequency-loss term switches on at ep >= bg_freq_warmup_epochs,
                # adding a new term so the TOTAL loss jumps ONCE (a change in loss
                # *composition*, not divergence).  A slope/drop across that boundary
                # always looks like a big positive jump -> false early-stop on every
                # config.  So drop the pre-warmup epochs and judge only the
                # composition-stable tail.
                _fw = int(getattr(cfg, "bg_freq_warmup_epochs", 3))
                _freq_on = float(getattr(cfg, "bg_freq_weight", 0.0)) > 0.0
                vals = list(history["loss"])
                # Only slice once training has actually crossed the warmup boundary
                # (some epochs are freq-on).  If len(vals) <= _fw, every recorded
                # epoch is still freq-off — one stable composition — so judge on all
                # of them.  (Without this guard, freq_warmup >= epoch budget would
                # slice vals to empty and the early-stop could never fire.)
                if _freq_on and _fw > 0 and len(vals) > _fw:
                    vals = vals[_fw:]

                if es_metric == "loss_patience":
                    # Aggressive: stop once the per-epoch relative loss drop stays
                    # below bg_es_min_drop for bg_es_patience consecutive epochs.
                    min_drop = float(getattr(cfg, "bg_es_min_drop", 0.01))   # 1%
                    patience = max(1, int(getattr(cfg, "bg_es_patience", 2)))
                    if len(vals) >= patience + 1:
                        drops = [
                            (vals[i - 1] - vals[i]) / max(abs(vals[i - 1]), 1e-12)
                            for i in range(len(vals) - patience, len(vals))
                        ]
                        if all(d < min_drop for d in drops):
                            stop_training = True
                            _ds = ", ".join(f"{d * 100:+.2f}%" for d in drops)
                            print(
                                f"{_log_pfx}[early-stop] loss drop < {min_drop * 100:g}% "
                                f"for {patience} consecutive epochs ({_ds}) "
                                f"-> stop at epoch {ep + 1}"
                            )
                else:
                    # Smoother slope variant (bg_es_metric == "loss").
                    min_slope = float(getattr(cfg, "bg_es_min_slope", 1e-3))  # rel. drop/epoch
                    if len(vals) >= es_window:
                        y = np.asarray(vals[-es_window:], dtype=np.float64)
                        scale = max(abs(float(y.mean())), 1e-12)
                        slope = float(np.polyfit(np.arange(es_window), y / scale, 1)[0])
                        if slope > -min_slope:
                            stop_training = True
                            print(
                                f"{_log_pfx}[early-stop] rel-loss slope {slope:+.5f}/ep > "
                                f"-{min_slope:g} over last {es_window} post-warmup epochs -> stop at epoch {ep + 1}"
                            )

    core_model = unwrap_bg_model(model)
    if evaluator is not None and best_model_weights is not None:
        core_model.load_state_dict(best_model_weights)

    pure_train_time = time.perf_counter() - t_start_train - eval_time_total
    print(f"\n{_log_pfx}--- Experiment [BG_only] finished ---")
    print(f"{_log_pfx}--- Pure training time: {pure_train_time:.2f} s ---")
    _ew = history.get("epoch_wall") or []
    if len(_ew) > 0:
        _ew_arr = np.asarray(_ew, dtype=np.float64)
        print(
            f"{_log_pfx}[timing] epochs={len(_ew)} | train_wall/epoch: "
            f"mean={float(_ew_arr.mean()):.2f}s min={float(_ew_arr.min()):.2f}s "
            f"max={float(_ew_arr.max()):.2f}s | sum={float(_ew_arr.sum()):.2f}s"
        )
    if evaluator is not None:
        print(f"{_log_pfx}--- Best global PSNR: {best_psnr:.2f} dB ---")

    # Extract optimizer state to CPU to prevent memory leak
    opt_state = optimizer.state_dict()
    for state in opt_state['state'].values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cpu()
    history["optimizer_state"] = opt_state

    return core_model, history
