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
import math


from config_io import _error_bounded_post_process, set_deterministic_seed
from frequency_losses import fft_mag_phase_loss_bg_t, masked_fft_mag_l1_t, masked_fft_phase_l1_t

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bg_normalize import (
    _bg_arch_kind, _bg_norm_mode, _bg_norm_eps, _bg_input_norm_mode, _bg_residual_norm_mode,
    _revin_mu_sig, _build_input_norm_tensors, normalize_bg_inputs, normalize_bg_residual_tensor,
    denormalize_bg_residual_tensor, _normalize_bg_batch,
)
from bg_sampling import (
    _gaussian_low_mid_high_split_t, _sample_bg_training_batch, _sample_slice2d_gpu, _to_gpu_volume,
)


def _to_device(v, device):
    """Move a numpy array or tensor to ``device``.

    Tensors (e.g. from the device-resident sampler) are moved as-is; numpy arrays are
    pinned + copied async on CUDA so the host->device transfer overlaps compute.
    """
    if torch.is_tensor(v):
        return v if v.device == torch.device(device) else v.to(device, non_blocking=True)
    t = torch.from_numpy(v)
    if torch.device(device).type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


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
    """Thin wrapper so nn.DataParallel / DDP can scatter the batch over GPUs and gather the
    four outputs (full residual + low/mid/high bands) back on batch dim 0."""

    def __init__(self, core, split_mode=None):
        super().__init__()
        self.core = core
        self.split_mode = split_mode

    def forward(self, xp_norm):
        if self.split_mode == "three":
            pred_low, pred_mid, pred_high, pred = self.core.bg_forward_split(xp_norm)
            return pred, pred_low, pred_mid, pred_high
        pred = self.core.bg_forward(xp_norm)
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
    dp_model = nn.DataParallel(adapter, device_ids=list(range(n_dp)))
    return dp_model, True


def _forward_bg_outputs(model, x_norm, split_mode=None):
    """Run the (possibly DataParallel/DDP-wrapped) model; returns the full residual and,
    with the three-band split, the per-band predictions."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        pred, pred_low, pred_mid, pred_high = model(x_norm)
        return {"pred": pred, "low": pred_low, "mid": pred_mid, "high": pred_high}
    core = unwrap_bg_model(model)
    if split_mode == "three":
        pred_low, pred_mid, pred_high, pred = core.bg_forward_split(x_norm)
        return {"pred": pred, "low": pred_low, "mid": pred_mid, "high": pred_high}
    return {"pred": core.bg_forward(x_norm), "low": None, "mid": None, "high": None}


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

    ai_contribution = np.zeros_like(lq_target, dtype=np.float32)

    mean_t, std_t, min_t, max_t = _build_input_norm_tensors(cfg, model_device, n_fields)

    with torch.no_grad():
        for z in range(z_lo, z_hi):
            y0 = 0
            x0 = 0

            slice_data = np.stack([field[z] for field in Xps], axis=0).astype(np.float32)
            slice_t = torch.from_numpy(slice_data).unsqueeze(0).to(model_device)
            slice_norm = normalize_bg_inputs(slice_t, cfg, mean_t, std_t, min_t, max_t)
            pred_norm = model.bg_forward(slice_norm)

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


def train_bg_only(Xs, Xps, device, cfg, evaluator=None):
    """Train the residual model on one (Xs, Xps) pair.

    Args:
        Xs:  ``[target_field]`` (+ aux fields), each ``(D, H, W)`` ground truth.
        Xps: ``[base_recon]`` (+ the same aux fields); ``n_fields = len(Xps)``.
        device, cfg: torch device and a ``TrainConfig`` (see ``build_bg_only_cfg``).
        evaluator: optional ``callable(model) -> psnr | (psnr, max_err)`` run at every
            epoch end; the best-PSNR weights are restored before returning.

    Trains on whole 2-D slices under a wall-clock budget (``cfg.max_train_time``) and an
    epoch cap (``cfg.epochs``) in bf16 AMP. Loss = spatial MSE + ramped dual-domain
    frequency loss + (with ``bg_split_mode="three"``) the per-band MSEs. The cosine
    schedule is re-planned once the real epoch/step cost is known (opt-in
    ``bg_sched_time_calibrate`` / ``bg_sched_step_calibrate``). An opt-in early stop
    (``cfg.bg_early_stop``) halts when the epoch loss improves by less than
    ``bg_es_min_drop`` (default 2%) for ``bg_es_patience`` (default 2) consecutive
    epochs; it is disabled in all reported experiments.

    Returns ``(model, history)``: ``model`` carries the best-PSNR weights (or the final
    weights without an evaluator); ``history`` holds per-epoch loss / psnr / time lists.
    """
    from siren_fft_backbone_model import UNET_Model

    split_mode = getattr(cfg, "bg_split_mode", None)
    use_split_bands = split_mode == "three"
    seed = getattr(cfg, "seed", 42)
    set_deterministic_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(getattr(cfg, "bg_cudnn_benchmark", True))   # autotune fixed-size convs
        torch.backends.cudnn.deterministic = bool(getattr(cfg, "bg_cudnn_deterministic", False))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    _pfx = (str(getattr(cfg, "bg_log_prefix", "") or "").strip() + " ") if str(getattr(cfg, "bg_log_prefix", "") or "").strip() else ""

    n_fields = len(Xps)
    # Sampling volumes: the residual replaces the target; memmaps are pulled into RAM once
    # so per-step slicing never hits the disk.
    _ram = lambda a: np.array(a) if isinstance(a, np.memmap) else a
    Xs_for_sampling = [_ram(Xs[0] - Xps[0])] + [_ram(a) for a in Xs[1:]]
    Xps = [_ram(a) for a in Xps]

    model = UNET_Model(
        n_fields=n_fields,
        bg_hidden=cfg.bg_h,
        bg_split_bands=bool(getattr(cfg, "bg_split_bands", False)),
        bg_split_mode=split_mode,
    ).to(device)
    model, _ = _maybe_wrap_dataparallel(model, cfg, device)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    total_steps = max(1, cfg.epochs * cfg.steps_per_epoch)
    # Linear warmup capped at 20% of the run (short BO-proxy trials must keep room for
    # the cosine decay), then cosine to 0.
    warmup_steps = max(0, min(int(getattr(cfg, "bg_lr_warmup_steps", 200)), total_steps // 5))
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
                        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    mse_loss = nn.MSELoss()

    # ---- AMP (default bf16; cfg.amp=False -> fp32) ----
    use_amp = bool(getattr(cfg, "amp", True))
    amp_dtype = str(getattr(cfg, "amp_dtype", "bf16")).lower()
    autocast_dtype = torch.bfloat16 if amp_dtype in ("bf16", "bfloat16") else torch.float16
    if not use_amp:
        autocast_dtype = torch.float32

    history = {"epoch": [], "loss": [], "psnr": [], "time": [], "max_err": [], "epoch_wall": [], "total_steps": 0}

    # Replay support: the sampler seed is a pure function of (epoch, step), so re-running
    # the same number of steps reproduces a wall-clock-cut run exactly (cfg.bg_max_steps).
    _max_steps = getattr(cfg, "bg_max_steps", None)
    _max_steps = int(_max_steps) if _max_steps is not None else None
    _steps_done = 0

    def _evaluate():
        res = evaluator(unwrap_bg_model(model))
        if isinstance(res, tuple):
            return float(res[0]), (res[1] if len(res) >= 2 else None)
        return float(res), None

    best_psnr, best_weights = -float("inf"), None
    if evaluator is not None:
        best_psnr, base_max_err = _evaluate()
        best_weights = copy.deepcopy(unwrap_bg_model(model).state_dict())
        history["epoch"].append(0); history["psnr"].append((0, best_psnr)); history["time"].append(0.0)
        if base_max_err is not None:
            history["max_err"].append(base_max_err)
        print(f"{_pfx}[Init] Epoch   0 | Global PSNR: {best_psnr:.2f} dB | MaxErr: "
              f"{base_max_err:.1f}" if base_max_err is not None else f"{_pfx}[Init] Epoch   0 | Global PSNR: {best_psnr:.2f} dB")
    else:
        print(f"{_pfx}[Init] evaluator=None, train only without PSNR tracking.")

    amp_str = "off" if not use_amp else ("bf16" if autocast_dtype is torch.bfloat16 else "fp16")
    _dp_flag = bool(getattr(cfg, "bg_data_parallel", False)) and torch.cuda.device_count() > 1
    print(f"{_pfx}[plan] pure_train_budget={float(cfg.max_train_time):.2f}s | epochs_cap={int(cfg.epochs)} | "
          f"steps/epoch={int(cfg.steps_per_epoch)} | patch={int(cfg.bg_patch_size)} | batch={int(cfg.bg_batch)} | "
          f"sample={str(getattr(cfg, 'bg_sample_mode', 'random')).lower()} | data_parallel={_dp_flag} | amp={amp_str}")
    print(f"{_pfx}[lr-sched] warmup_steps={warmup_steps} (of {total_steps} total, cap=20%) -> cosine decay | "
          f"freq_weight ramps linearly to {getattr(cfg, 'bg_freq_weight', 0.0)} over "
          f"{int(getattr(cfg, 'bg_freq_warmup_epochs', 3))} epoch(s)")
    early_stop = bool(getattr(cfg, "bg_early_stop", False))
    print(f"{_pfx}[early-stop] " + (f"ENABLED | min_drop={getattr(cfg, 'bg_es_min_drop', 0.02)} patience={getattr(cfg, 'bg_es_patience', 2)}"
                                    if early_stop else "DISABLED (cfg.bg_early_stop is False/unset)"))

    t_start_train = time.perf_counter()
    eval_time_total = 0.0
    stop_training = False
    mean_t, std_t, min_t, max_t = _build_input_norm_tensors(cfg, device, n_fields)

    # ---- Device-resident sampling (auto when the volumes fit in 60% of free VRAM) ----
    gpu_sampling = False
    Xs_gpu = Xps_gpu = None
    _gpu_want = getattr(cfg, "bg_gpu_sampling", "auto")
    if _gpu_want is not False:
        _need = int(sum(a.size for a in Xs_for_sampling) + sum(a.size for a in Xps)) * 4
        if _gpu_want is True:
            _enable = True
        elif torch.device(device).type == "cuda":
            try:
                _free = torch.cuda.mem_get_info(device)[0]
            except Exception:
                _free = 0
            _enable = _free > 0 and _need < 0.6 * _free
        else:
            _enable = False
        if _enable:
            Xs_gpu = [_to_gpu_volume(a, device) for a in Xs_for_sampling]
            Xps_gpu = [_to_gpu_volume(a, device) for a in Xps]
            gpu_sampling = True
            print(f"{_pfx}[gpu-sampling] {len(Xps_gpu)} fields resident on {device} (~{_need/1e9:.1f} GB)")

    # Step-level schedule calibration (opt-in): armed only when one epoch could outlast the budget.
    _step_calib = ({"k0": 32, "k1": 96, "t0": None}
                   if (bool(getattr(cfg, "bg_sched_step_calibrate", False))
                       and float(cfg.max_train_time) < 1e8 and int(cfg.steps_per_epoch) > 96) else None)

    freq_focus = getattr(cfg, "bg_freq_focus", "low")
    freq_boost = float(getattr(cfg, "bg_freq_boost", 1.0))
    freq_target = float(getattr(cfg, "bg_freq_weight", 0.0))
    freq_warmup_steps = int(getattr(cfg, "bg_freq_warmup_epochs", 3)) * cfg.steps_per_epoch
    phase_weight = float(getattr(cfg, "bg_fft_phase_weight", 1.0))
    band_w = (float(getattr(cfg, "bg_low_weight", 0.2)), float(getattr(cfg, "bg_mid_weight", 0.5)), float(getattr(cfg, "bg_high_weight", 1.0)))
    sigma_low, sigma_mid = float(getattr(cfg, "bg_sigma_low", 0.08)), float(getattr(cfg, "bg_sigma_mid", 0.18))

    for ep in range(cfg.epochs):
        if stop_training:
            break

        # Epoch-level schedule calibration (opt-in): with a wall-clock budget the epoch cap
        # is huge, so the initial cosine never decays. After epochs 1 and 2 (warmup-inflated,
        # then steady-state) re-plan the cosine over the steps that fit the remaining budget;
        # LambdaLR starts at the current lr factor and clamps at 0 past its horizon.
        if (ep in (1, 2) and bool(getattr(cfg, "bg_sched_time_calibrate", False))
                and float(cfg.max_train_time) < 1e8 and len(history["epoch_wall"]) >= ep):
            _eN = float(history["epoch_wall"][ep - 1])
            if _eN > 0:
                _rem_sec = max(0.0, float(cfg.max_train_time) - float(sum(history["epoch_wall"][:ep])))
                _rem_steps = max(1, int(_rem_sec / _eN * cfg.steps_per_epoch))
                _f0 = float(optimizer.param_groups[0]["lr"]) / max(float(cfg.lr), 1e-12)
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda t, T=_rem_steps, f0=_f0: f0 * 0.5 * (1.0 + math.cos(math.pi * min(t, T) / T)))
                print(f"{_pfx}[lr-sched] time-budget calibration@ep{ep}: epoch={_eN:.2f}s -> cosine {_f0:.2f}->0 "
                      f"over ~{_rem_steps} steps ({_rem_sec:.1f}s of {float(cfg.max_train_time):.1f}s budget)")

        epoch_start = time.perf_counter()
        model.train()
        ep_loss, ep_freq, ep_low, ep_mid, ep_high = [], [], [], [], []

        for step in range(cfg.steps_per_epoch):
            pure_time = time.perf_counter() - t_start_train - eval_time_total
            stop_flag = int(pure_time >= cfg.max_train_time or (_max_steps is not None and _steps_done >= _max_steps))
            if getattr(cfg, "bg_ddp", False):
                import torch.distributed as dist
                if dist.is_initialized():
                    t_flag = torch.tensor([stop_flag], device=device, dtype=torch.int32)
                    dist.all_reduce(t_flag, op=dist.ReduceOp.MAX)
                    stop_flag = int(t_flag.item())
            if stop_flag:
                stop_training = True
                break
            _steps_done += 1
            step_seed = seed + ep * 10000 + step

            # Step-level calibration: probe the steady-state step cost early in epoch 0; if one
            # epoch would eat >80% of the budget, plan the remaining warmup + cosine over the
            # steps that actually fit (QMCPack: 33,120 slices/epoch at a 54 s budget).
            if _step_calib is not None and ep == 0:
                if step == _step_calib["k0"]:
                    _step_calib["t0"] = pure_time
                elif step == _step_calib["k1"] and _step_calib["t0"] is not None:
                    _dt = (pure_time - _step_calib["t0"]) / float(_step_calib["k1"] - _step_calib["k0"])
                    _proj = _dt * float(cfg.steps_per_epoch)
                    if _dt > 0 and _proj > 0.8 * float(cfg.max_train_time):
                        _rem_steps = max(1, int((float(cfg.max_train_time) - pure_time) / _dt))
                        _wu = max(0, int(warmup_steps) - int(step))
                        _T = max(1, _rem_steps - _wu)
                        _f0 = float(optimizer.param_groups[0]["lr"]) / max(float(cfg.lr), 1e-12)
                        def _lam(t, f0=_f0, wu=_wu, T=_T):
                            if t < wu:
                                return f0 + (1.0 - f0) * (t / float(wu))
                            return 0.5 * (1.0 + math.cos(math.pi * min(t - wu, T) / T))
                        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lam)
                        print(f"{_pfx}[lr-sched] step-budget calibration@step{step}: {_dt*1e3:.2f} ms/step -> one epoch ≈ "
                              f"{_proj:.1f}s of a {float(cfg.max_train_time):.1f}s budget; warmup {_f0:.2f}->1 over {_wu} steps, "
                              f"then cosine 1->0 over ~{_T} steps")
                    _step_calib = None

            if gpu_sampling:
                batch = _sample_slice2d_gpu(Xs_gpu, Xps_gpu, cfg, ep, step, seed=step_seed)
            else:
                batch = _sample_bg_training_batch(Xs_for_sampling, Xps, cfg, ep, step, seed=step_seed)
            xs_norm = _normalize_bg_batch(_to_device(batch["xp"], device), cfg, mean_t, std_t, min_t, max_t)
            ys_norm = normalize_bg_residual_tensor(_to_device(batch["x"], device), cfg)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                out = _forward_bg_outputs(model, xs_norm, split_mode=split_mode if use_split_bands else None)
                pred = out["pred"]
                loss_bg = mse_loss(pred, ys_norm)
                g = ep * cfg.steps_per_epoch + step
                freq_weight = freq_target if freq_warmup_steps <= 0 else min(1.0, g / freq_warmup_steps) * freq_target
                loss_freq, _, _ = fft_mag_phase_loss_bg_t(pred, ys_norm, focus=freq_focus, boost=freq_boost,
                                                          mag_weight=1.0, phase_weight=phase_weight)
                loss = loss_bg + freq_weight * loss_freq
                if use_split_bands:
                    tgt_low, tgt_mid, tgt_high = _gaussian_low_mid_high_split_t(ys_norm, sigma_low=sigma_low, sigma_mid=sigma_mid)
                    loss_low, loss_mid, loss_high = mse_loss(out["low"], tgt_low), mse_loss(out["mid"], tgt_mid), mse_loss(out["high"], tgt_high)
                    loss = loss + band_w[0] * loss_low + band_w[1] * loss_mid + band_w[2] * loss_high
                else:
                    loss_low = loss_mid = loss_high = ys_norm.new_tensor(0.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            ep_loss.append(float(loss.item())); ep_freq.append(float(loss_freq.item()))
            ep_low.append(float(loss_low.item())); ep_mid.append(float(loss_mid.item())); ep_high.append(float(loss_high.item()))

        if not ep_loss:
            continue
        epoch_wall = time.perf_counter() - epoch_start
        cur_p, cur_max_err = (None, None)
        if evaluator is not None:
            t0 = time.perf_counter()
            cur_p, cur_max_err = _evaluate()
            eval_time_total += time.perf_counter() - t0
        cum_train_time = time.perf_counter() - t_start_train - eval_time_total

        history["epoch"].append(ep + 1); history["loss"].append(float(np.mean(ep_loss)))
        history["epoch_wall"].append(float(epoch_wall)); history["time"].append(float(cum_train_time))
        if cur_p is not None:
            history["psnr"].append((ep + 1, cur_p))
            if cur_max_err is not None:
                history["max_err"].append(cur_max_err)

        msg = (f"{_pfx}Epoch {ep + 1:3d} [BG] | train_wall={epoch_wall:.2f}s | Loss: {history['loss'][-1]:.6f}"
               f" | Freq: {np.mean(ep_freq):.6f}")
        if use_split_bands:
            msg += f" | Low: {np.mean(ep_low):.6f} | Mid: {np.mean(ep_mid):.6f} | High: {np.mean(ep_high):.6f}"
        if cur_p is not None:
            msg += f" | Global: {cur_p:.2f} dB | MaxErr: " + (f"{cur_max_err:.1f}" if cur_max_err is not None else "N/A")
            if cur_p > best_psnr:
                best_psnr, best_weights = cur_p, copy.deepcopy(unwrap_bg_model(model).state_dict())
                msg += "  [New Best!]"
        print(msg)
        if ep == 0:
            print(f"{_pfx}[timing] first_epoch_pure_train≈{cum_train_time:.3f}s (excludes the end-of-epoch eval)")

        # ---- Early stop (opt-in): loss improved < min_drop for `patience` consecutive epochs ----
        if early_stop and not stop_training:
            min_drop = float(getattr(cfg, "bg_es_min_drop", 0.02))
            patience = max(1, int(getattr(cfg, "bg_es_patience", 2)))
            vals = history["loss"]
            if len(vals) >= patience + 1:
                drops = [(vals[k - 1] - vals[k]) / max(abs(vals[k - 1]), 1e-12) for k in range(len(vals) - patience, len(vals))]
                if all(d < min_drop for d in drops):
                    stop_training = True
                    print(f"{_pfx}[early-stop] loss drop < {min_drop * 100:g}% for {patience} consecutive epochs "
                          f"({', '.join(f'{d * 100:+.2f}%' for d in drops)}) -> stop at epoch {ep + 1}")

    history["total_steps"] = int(_steps_done)
    core_model = unwrap_bg_model(model)
    if best_weights is not None:
        core_model.load_state_dict(best_weights)

    pure_train_time = time.perf_counter() - t_start_train - eval_time_total
    print(f"\n{_pfx}--- Experiment [BG_only] finished ---\n{_pfx}--- Pure training time: {pure_train_time:.2f} s ---")
    if history["epoch_wall"]:
        ew = np.asarray(history["epoch_wall"], dtype=np.float64)
        print(f"{_pfx}[timing] epochs={len(ew)} | train_wall/epoch: mean={ew.mean():.2f}s min={ew.min():.2f}s "
              f"max={ew.max():.2f}s | sum={ew.sum():.2f}s")
    if evaluator is not None:
        print(f"{_pfx}--- Best global PSNR: {best_psnr:.2f} dB ---")
    return core_model, history
