"""SPERR_fft.py — standalone script version of SPERR_fft.ipynb.

Companion to SPERR.py: same 6 methods (SZ3, SZ3+Ours, SPERR, SPERR+Ours, SZ3+NeurLZ,
SPERR+NeurLZ) and same 4 datasets (NYX x3 targets, Miranda, WarpX, Magnetic Reconnection
-- 6 panels total), but the y-axis is FFT magnitude / phase error instead of PSNR:

    mag_err, phase_err = _global_fft_err(x_true, x_hat)

a strided-2D-FFT metric (mean |delta FFT magnitude| and mean |delta wrapped phase| over
~32 z-slices), reused as-is from frequency_head_loss/fft_err.ipynb.

IMPORTANT: SPERR.py's cache only stores CR/PSNR numbers -- no model weights or
reconstructed volumes were ever saved, so FFT error can't be computed "for free" from
that cache. This script re-runs the SAME two-phase BO + training pipeline (same configs,
same seed) so results are apples-to-apples with the PSNR script, but computes FFT error
right when each reconstruction is already in memory (no extra training cost) and caches
its OWN results (sperr_fft_cache/) so future re-runs of THIS script are free.

Same --task CLI as SPERR.py, for SLURM-style per-dataset parallel launches:
  --task {nyx_b,nyx_t,nyx_d,miranda,warpx,mag,plot,all}   (default: all)
  --task plot loads every cached result (expects cache HITs from prior parallel runs)
  and draws the two combined figures.
"""
import os, sys, time, subprocess, hashlib, pickle, io, contextlib, random, argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
SPERR_BIN    = "/home/sam/Halo_Finder/SPERR/build/bin/sperr3d"
SZ3_LIB      = "/home/sam/Data_Compression/SZ3/build/lib64/libSZ3c.so"
PYSZ_PATH    = "/home/sam/Data_Compression/SZ3/tools/pysz"
SCRIPTS_PATH = "/home/sam/Halo_Finder/Final_design/base_script"
for _p in (PYSZ_PATH, SCRIPTS_PATH):
    if _p not in sys.path:
        sys.path.append(_p)

from pysz import SZ
from bg_stage import train_bg_only, run_bg_inference, unwrap_bg_model
from experiment import build_bg_only_cfg, estimate_bg_model_param_bytes
from bg_shard import pick_bg_h_under_budget
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from monai.networks.nets import BasicUNet
from config_io import _error_bounded_post_process


def _sperr_env():
    """Env for the sperr3d subprocess, with LD_LIBRARY_PATH extended to wherever
    libSPERR.so* actually lives. On a fresh machine/cluster the .so is often not on
    the default loader path even though the binary itself is executable, causing
    `error while loading shared libraries: libSPERR.so.*` (subprocess rc=127) --
    which silently empties every SPERR/SPERR+Ours/SPERR+NeurLZ series, since
    run_sperr's failure path just returns None and the caller `continue`s past it.
    We search a few directories up from SPERR_BIN for any libSPERR.so*, so this
    works without hand-editing LD_LIBRARY_PATH on every new machine."""
    import glob
    env = os.environ.copy()
    root = os.path.dirname(SPERR_BIN)
    found = set()
    for _ in range(4):                      # walk up a few levels from bin/
        for hit in glob.glob(os.path.join(root, "**", "libSPERR.so*"), recursive=True):
            found.add(os.path.dirname(hit))
        root = os.path.dirname(root)
        if not root or root == "/":
            break
    if found:
        env["LD_LIBRARY_PATH"] = ":".join(found) + (":" + env["LD_LIBRARY_PATH"]
                                                     if env.get("LD_LIBRARY_PATH") else "")
    return env


_SPERR_ENV = _sperr_env()

device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sz_engine = SZ(SZ3_LIB)
BYTES_PER_PARAM = 2
SEED = 17

def set_seed(s=SEED):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)

DETERMINISTIC = False
torch.backends.cudnn.benchmark = not DETERMINISTIC
torch.backends.cudnn.deterministic = DETERMINISTIC
set_seed(SEED)


def compute_psnr(x_true, x_hat, drange):
    mse = float(np.mean((np.asarray(x_true, np.float64) - np.asarray(x_hat, np.float64)) ** 2))
    return 100.0 if mse == 0 else 20.0 * np.log10(drange) - 10.0 * np.log10(mse)


def _global_fft_err(x_true, x_hat, n_slices=32):
    """Global FFT magnitude / phase L1 error over a strided set of z-slices.
    Reused as-is from frequency_head_loss/fft_err.ipynb."""
    x_true = np.asarray(x_true); x_hat = np.asarray(x_hat)
    D = x_true.shape[0]; step = max(1, D // int(n_slices))
    mag, pha = [], []
    for z in range(0, D, step):
        ft_t = np.fft.rfft2(x_true[z].astype(np.float64), norm="ortho")
        ft_h = np.fft.rfft2(x_hat[z].astype(np.float64), norm="ortho")
        mag.append(float(np.mean(np.abs(np.abs(ft_h) - np.abs(ft_t)))))
        d = np.angle(np.exp(1j * (np.angle(ft_h) - np.angle(ft_t))))
        pha.append(float(np.mean(np.abs(d))))
    return (float(np.mean(mag)) if mag else float("nan"),
            float(np.mean(pha)) if pha else float("nan"))


def bg_h_for_params(budget, shape, n_fields):
    h, _est = pick_bg_h_under_budget(int(budget), shape=shape, n_fields=int(n_fields),
                                     bg_arch="spatial", h_candidates=list(range(3, 256)))
    return int(h)


def run_sperr(data_file, target_gt, shape, drange, target_psnr):
    """SPERR at a target PSNR. Returns (CR, PSNR, recon, nbytes).
    Tmp filenames use os.getpid()+nanosecond-time (NOT numpy's global RNG) so parallel
    dataset runs can never collide on /tmp paths; failures are printed, not silent."""
    W, H, D = shape[2], shape[1], shape[0]
    tag = f"{os.getpid()}_{time.time_ns()}"
    bit = f"/tmp/sperr_{tag}.bit"; rec = f"/tmp/sperr_{tag}.dec.f32"
    p1 = subprocess.run([SPERR_BIN, "-c", "--ftype", "32",
                        "--dims", str(W), str(H), str(D), "--psnr", f"{float(target_psnr):.4f}",
                        "--bitstream", bit, data_file], capture_output=True, text=True, env=_SPERR_ENV)
    if not os.path.exists(bit):
        print(f"  [run_sperr] COMPRESS FAILED (psnr={target_psnr:.2f}) rc={p1.returncode} "
              f"stdout={p1.stdout!r} stderr={p1.stderr!r}")
        return None, None, None, None
    nbytes = os.path.getsize(bit)
    p2 = subprocess.run([SPERR_BIN, "-d", "--decomp_f", rec, bit], capture_output=True, text=True, env=_SPERR_ENV)
    cr = psnr = recon = None
    if os.path.exists(rec):
        recon = np.fromfile(rec, dtype=np.float32).reshape(shape)
        psnr  = compute_psnr(target_gt, recon, drange)
        cr    = (int(np.prod(shape)) * 4) / nbytes
    else:
        print(f"  [run_sperr] DECOMPRESS FAILED (psnr={target_psnr:.2f}) rc={p2.returncode} "
              f"stdout={p2.stdout!r} stderr={p2.stderr!r}")
    for f in (bit, rec):
        if os.path.exists(f):
            os.remove(f)
    return cr, psnr, recon, nbytes


def _sperr_psnr_for_cr(data_file, target_gt, shape, drange, target_cr, lo=1.0, hi=250.0, iters=10):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cr, _, _, _ = run_sperr(data_file, target_gt, shape, drange, mid)
        if cr is None:
            lo = mid; continue
        if cr > target_cr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


PROXY_DS = 2

print("Setup ready | device:", device)

# ─────────────────────────────────────────────────────────────────────────────
# CLI: --task lets a SLURM array/parallel launch compute ONE dataset per process
# (each writes to the shared cache/), then a final `--task plot` process loads
# every cached result and draws the combined figures. No --task (or --task all) =
# original monolithic behavior: run everything in this one process, then plot.
# ─────────────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--task", choices=["nyx_b", "nyx_t", "nyx_d", "miranda", "warpx", "mag", "plot", "all"],
                    default="all")
_parser.add_argument("--save_recons", action="store_true",
                    help="accepted for CLI compatibility; NOT implemented -- cached_bench_field_fft only "
                         "stores CR/PSNR/FFT-error numbers, never model weights or reconstructed volumes.")
args, _unknown = _parser.parse_known_args()
if args.save_recons:
    print("[warn] --save_recons was passed but is not implemented (only CR/PSNR/FFT get cached); ignoring.")
TASK = args.task
print(f"[task] running: {TASK}")

# ─────────────────────────────────────────────────────────────────────────────
# BO config + bench_field_fft
# ─────────────────────────────────────────────────────────────────────────────
BO_ENABLE           = True
BG_BATCH            = 1
SPERR_MAX_CR        = 500
BO_N_TRIALS         = 10
BO_N_STARTUP        = 3
BO_ENQUEUE_LR       = 1e-3
BO_LR_MIN           = 1e-4
BO_LR_MAX           = 1e-2
BO_PHASE1_EPOCHS    = 3
BO_MIN_GAIN_DB      = 0.30
BG_USE_AUX          = True
EPOCHS_OVERRIDE     = None
BO_SEARCH_DIRECTION = True
N_FFT_SLICES        = 32     # strided z-slices used by _global_fft_err

_DIR_FWD = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 0, 1)}   # bring axis k to the front (= slicing axis)
_DIR_INV = {0: (0, 1, 2), 1: (1, 0, 2), 2: (1, 2, 0)}   # inverse of _DIR_FWD[k]


def _perm_view(a, k):
    return np.transpose(np.asarray(a), _DIR_FWD[k])


def _unperm_view(a, k):
    """Inverse of _perm_view: bring a volume that was permuted by axis k back to
    the original (Z, Y, X) orientation, so it can be FFT-compared against target_gt."""
    return np.transpose(np.asarray(a), _DIR_INV[k]) if k != 0 else np.asarray(a)


def _take_perm(a, k, idx):
    return np.ascontiguousarray(np.transpose(np.take(np.asarray(a, np.float32), idx, axis=k), _DIR_FWD[k]))


def _new_series():
    return {"CR": [], "PSNR": [], "fft_mag": [], "fft_phase": []}


def bench_field_fft(name, target_gt, target_file, aux_list, shape, rel_errs,
                    param_budget, epochs, lr=1e-3, sperr_psnr_offset=8.0,
                    sperr_extra_span=0.0, sperr_n_extra=0, full_slice=False, full_slice_axis=0,
                    bg_low_w=0.2, bg_mid_w=0.5, bg_high_w=1.0, bo_axes=None, time_budget=None):
    """Same two-phase BO + SZ3/SPERR/model/NeurLZ pipeline as SPERR.py's bench_field,
    but each series dict also carries fft_mag/fft_phase (via _global_fft_err), computed
    on the SAME reconstruction used for PSNR (no extra training, one extra FFT per point).
    time_budget: if set, Phase-2 "Ours" training (per rel_err) is capped by wall-clock
    seconds instead of `epochs` -- see SPERR.py's bench_field for the full rationale.
    """
    set_seed(SEED)
    target_gt = np.asarray(target_gt, np.float32)
    if EPOCHS_OVERRIDE:
        epochs = int(EPOCHS_OVERRIDE)
    if not BG_USE_AUX:
        aux_list = []
    drange    = float(target_gt.max() - target_gt.min())
    n_fields  = 1 + len(aux_list)
    bg_h      = bg_h_for_params(param_budget, shape, n_fields)
    n_params, nn_bytes = estimate_bg_model_param_bytes(
        n_fields=n_fields, shape=shape, bg_arch="spatial", bg_h=bg_h, dtype_bytes=BYTES_PER_PARAM)
    orig_bytes = int(np.prod(shape)) * 4
    print(f"[{name}] budget {param_budget:,} -> bg_h={bg_h} (~{n_params:,} params, "
          f"{nn_bytes/1e3:.1f} KB) | n_fields={n_fields} | drange={drange:.3g}")
    neurlz_features = None
    if ADD_NEURLZ:
        _nlz_nf = 1 if globals().get("NEURLZ_SINGLE_FIELD", False) else n_fields
        neurlz_features = (_neurlz_features_for_params(n_params, _nlz_nf)
                           if NEURLZ_FEATURES == "match" else NEURLZ_FEATURES)
        print(f"[{name}] NeurLZ BasicUNet features={tuple(neurlz_features)} "
              f"(~{_basicunet_nparams(neurlz_features, _nlz_nf):,} params, in_ch={_nlz_nf})")

    sz3, pipe             = _new_series(), _new_series()
    sperr, sperr_pipe     = _new_series(), _new_series()
    neurlz, sperr_neurlz  = _new_series(), _new_series()

    def _phase1_best(base_recon, base_rel):
        if not BO_ENABLE:
            return (full_slice_axis if full_slice else 0), lr
        full_shape = np.asarray(target_gt).shape
        axes = (list(bo_axes) if bo_axes is not None else
               ([0, 1, 2] if BO_SEARCH_DIRECTION else [full_slice_axis if full_slice else 0]))
        _dsp = ((lambda x: np.ascontiguousarray(x[:, ::PROXY_DS, ::PROXY_DS]))
                if PROXY_DS > 1 else (lambda x: x))

        proxy = {}
        for k in axes:
            Dk  = int(full_shape[k])
            idx = np.arange(0, Dk, PROXY_DS)
            tgt_t  = _dsp(_take_perm(target_gt,  k, idx))
            base_t = _dsp(_take_perm(base_recon, k, idx))
            aux_t  = [_dsp(_take_perm(a, k, idx)) for a in aux_list]
            dr_t   = float(tgt_t.max() - tgt_t.min()) or 1.0
            proxy[k] = ([tgt_t] + aux_t, [base_t] + aux_t, dr_t, compute_psnr(tgt_t, base_t, dr_t))

        def objective(trial):
            k    = trial.suggest_categorical("direction", axes)
            lr_c = trial.suggest_float("lr", BO_LR_MIN, BO_LR_MAX, log=True)
            Xs_t, Xps_t, dr_t, base_p = proxy[k]
            nz, hh, ww = Xs_t[0].shape
            patch_d = int(min(hh, ww))
            cfg = build_bg_only_cfg(
                X_target=Xs_t[0], Xps=Xps_t, max_train_time=1e9, bg_h=bg_h, roi_h=4,
                epochs=int(BO_PHASE1_EPOCHS), steps_per_epoch=nz, bg_patch_size=patch_d,
                bg_batch=1, lr=float(lr_c), bg_freq_weight=0.5, bg_fft_phase_weight=0.5,
                bg_freq_warmup_epochs=1, bg_field_norm="zscore")
            cfg.bg_arch = "spatial"; cfg.bg_split_mode = "three"; cfg.bg_split_bands = True
            cfg.bg_split_sigma = 0.12; cfg.bg_sigma_low = 0.08; cfg.bg_sigma_mid = 0.18
            cfg.bg_low_weight = bg_low_w; cfg.bg_mid_weight = bg_mid_w; cfg.bg_high_weight = bg_high_w
            cfg.bg_cr_rel_err = float(base_rel); cfg.bg_gpu_sampling = True; cfg.seed = SEED
            cfg.bg_full_slice = full_slice
            cfg.bg_cudnn_benchmark = not DETERMINISTIC; cfg.bg_cudnn_deterministic = DETERMINISTIC
            def evt(m, _c=cfg):
                xh = run_bg_inference(unwrap_bg_model(m), Xs_t, Xps_t, _c, float(base_rel))
                return compute_psnr(Xs_t[0], xh, dr_t), 0.0
            set_seed(SEED)
            with contextlib.redirect_stdout(io.StringIO()):
                m, _ = train_bg_only(Xs=Xs_t, Xps=Xps_t, device=device, cfg=cfg, evaluator=evt)
                p = compute_psnr(Xs_t[0], run_bg_inference(unwrap_bg_model(m), Xs_t, Xps_t, cfg, float(base_rel)), dr_t)
            del m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if not np.isfinite(p):
                p = -1e9
            trial.set_user_attr("gain", float(p - base_p))
            return float(p)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=int(BO_N_STARTUP)))
        for k in axes:
            study.enqueue_trial({"direction": k, "lr": float(BO_ENQUEUE_LR)})
        with contextlib.redirect_stdout(io.StringIO()):
            study.optimize(objective, n_trials=int(BO_N_TRIALS))

        best_dir  = int(study.best_params["direction"])
        best_lr   = float(study.best_params["lr"])
        best_gain = float(study.best_trial.user_attrs.get("gain", 0.0))
        per_dir = {}
        for t in study.trials:
            if t.value is None:
                continue
            d = int(t.params["direction"])
            if d not in per_dir or t.value > per_dir[d][1]:
                per_dir[d] = (t.params["lr"], t.value)
        summ = " ".join(f"axis{d}:{per_dir[d][1]:.1f}@{per_dir[d][0]:.0e}" for d in axes if d in per_dir)
        print(f"  [TPE] band={base_rel:.1e} {int(BO_N_TRIALS)} trials -> PICK axis{best_dir} "
              f"lr={best_lr:.1e} proxy={study.best_value:.2f} gain={best_gain:+.2f} | {summ}")
        if best_gain <= BO_MIN_GAIN_DB:
            print(f"  [TPE] => best gain {best_gain:+.2f} <= {BO_MIN_GAIN_DB}dB -> axis0, fixed lr={lr:.1e}")
            return (full_slice_axis if full_slice else 0), lr
        return best_dir, best_lr

    def _train_residual(base_recon, base_rel):
        """Same as SPERR.py's _train_residual, but ALSO returns the enhanced
        reconstruction un-permuted back to (Z,Y,X) so FFT error can be computed
        against the original target_gt."""
        set_seed(SEED)
        best_k, use_lr = _phase1_best(base_recon, base_rel)
        Xs0  = [target_gt] + aux_list
        Xps0 = [np.ascontiguousarray(base_recon, np.float32)] + aux_list
        if best_k == 0:
            Xs, Xps = Xs0, Xps0
        else:
            Xs  = [np.ascontiguousarray(_perm_view(a, best_k)) for a in Xs0]
            Xps = [np.ascontiguousarray(_perm_view(a, best_k)) for a in Xps0]
        dep_d   = int(Xs[0].shape[0])
        patch_d = int(min(Xs[0].shape[1], Xs[0].shape[2]))
        use_time_budget = time_budget is not None and float(time_budget) > 0
        cfg = build_bg_only_cfg(
            X_target=Xs[0], Xps=Xps,
            max_train_time=(float(time_budget) if use_time_budget else 1e9), bg_h=bg_h, roi_h=4,
            epochs=(100000 if use_time_budget else epochs), steps_per_epoch=max(1, dep_d // int(BG_BATCH)),
            bg_patch_size=patch_d, bg_batch=int(BG_BATCH), lr=use_lr,
            bg_freq_weight=0.5, bg_fft_phase_weight=0.5, bg_freq_warmup_epochs=1,
            bg_field_norm="zscore")
        cfg.bg_arch = "spatial"; cfg.bg_split_mode = "three"; cfg.bg_split_bands = True
        cfg.bg_split_sigma = 0.12; cfg.bg_sigma_low = 0.08; cfg.bg_sigma_mid = 0.18
        cfg.bg_low_weight = bg_low_w; cfg.bg_mid_weight = bg_mid_w; cfg.bg_high_weight = bg_high_w
        cfg.bg_cr_rel_err = float(base_rel)
        cfg.bg_gpu_sampling = True
        cfg.seed = SEED
        cfg.bg_full_slice = full_slice
        cfg.bg_cudnn_benchmark = not DETERMINISTIC; cfg.bg_cudnn_deterministic = DETERMINISTIC
        def ev(model, c=cfg, Xs=Xs, Xps=Xps, r=base_rel):
            return compute_psnr(Xs[0], run_bg_inference(model, Xs, Xps, c, float(r)), drange), 0.0
        model, _hist = train_bg_only(Xs=Xs, Xps=Xps, device=device, cfg=cfg, evaluator=ev)
        bg_train_time = float(_hist["time"][-1]) if _hist.get("time") else float("nan")
        x_hat = run_bg_inference(model, Xs, Xps, cfg, float(base_rel))
        p = compute_psnr(Xs[0], x_hat, drange)
        x_hat_orig = _unperm_view(x_hat, best_k)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return p, bg_train_time, best_k, x_hat_orig

    # ── SZ3  and  SZ3 + model ──
    for rel in rel_errs:
        b, _   = sz_engine.compress(target_gt, 1, 0, float(rel), 0)
        sz_len = len(b)
        xq     = sz_engine.decompress(b, shape, np.float32)
        p_sz3  = compute_psnr(target_gt, xq, drange); cr_sz3 = orig_bytes / sz_len
        m_sz3, ph_sz3 = _global_fft_err(target_gt, xq, N_FFT_SLICES)
        sz3["CR"].append(cr_sz3); sz3["PSNR"].append(p_sz3)
        sz3["fft_mag"].append(m_sz3); sz3["fft_phase"].append(ph_sz3)

        p_pipe, bg_time, bg_axis, xhat_pipe = _train_residual(xq, rel)
        cr_pipe = orig_bytes / (sz_len + nn_bytes)
        m_pipe, ph_pipe = _global_fft_err(target_gt, xhat_pipe, N_FFT_SLICES)
        pipe["CR"].append(cr_pipe); pipe["PSNR"].append(p_pipe)
        pipe["fft_mag"].append(m_pipe); pipe["fft_phase"].append(ph_pipe)
        print(f"  rel={rel:.0e} | SZ3 {cr_sz3:6.1f}x/{p_sz3:5.1f}dB/mag={m_sz3:.3g}/pha={ph_sz3:.3g} | "
              f"SZ3+model {cr_pipe:6.1f}x/{p_pipe:5.1f}dB/mag={m_pipe:.3g}/pha={ph_pipe:.3g}")

        if ADD_NEURLZ:
            p_nlz, nlz_params, nlz_time, nlz_eps, _hist_nlz, enh_nlz = run_neurlz(
                target_gt, xq, aux_list, shape, drange, rel,
                int(epochs * NEURLZ_EPOCH_MULT), neurlz_features, time_budget=bg_time,
                slice_axis=NEURLZ_SLICE_AXIS, return_history=True)
            enh_nlz_orig = _unperm_view(enh_nlz, NEURLZ_SLICE_AXIS)
            m_nlz, ph_nlz = _global_fft_err(target_gt, enh_nlz_orig, N_FFT_SLICES)
            cr_nlz = orig_bytes / (sz_len + nlz_params * BYTES_PER_PARAM)
            neurlz["CR"].append(cr_nlz); neurlz["PSNR"].append(p_nlz)
            neurlz["fft_mag"].append(m_nlz); neurlz["fft_phase"].append(ph_nlz)
            print(f"           SZ3+NeurLZ {cr_nlz:6.1f}x/{p_nlz:5.1f}dB/mag={m_nlz:.3g}/pha={ph_nlz:.3g} "
                  f"[{nlz_params:,}p] | BG {epochs}ep {bg_time:.1f}s ≈ NeurLZ {nlz_time:.1f}s/{nlz_eps}ep")

    # ── SPERR  and  SPERR + model ──
    if sz3["PSNR"]:
        hi = max(sz3["PSNR"]) - sperr_psnr_offset
        lo = min(sz3["PSNR"]) - sperr_psnr_offset - sperr_extra_span
        if SPERR_MAX_CR:
            lo = max(lo, _sperr_psnr_for_cr(target_file, target_gt, shape, drange, float(SPERR_MAX_CR)))
        n_pts = len(rel_errs) + sperr_n_extra
        print(f"  SPERR sweep: {n_pts} targets, PSNR {hi:.1f}..{lo:.1f} dB"
              + (f" (CR capped <= {SPERR_MAX_CR})" if SPERR_MAX_CR else ""))
        for tp in np.linspace(hi, lo, n_pts):
            cr_sp, p_sp, recon_sp, sp_bytes = run_sperr(target_file, target_gt, shape, drange, float(tp))
            if recon_sp is None:
                continue
            if SPERR_MAX_CR and cr_sp is not None and cr_sp > SPERR_MAX_CR * 1.03:
                continue
            m_sp, ph_sp = _global_fft_err(target_gt, recon_sp, N_FFT_SLICES)
            sperr["CR"].append(cr_sp); sperr["PSNR"].append(p_sp)
            sperr["fft_mag"].append(m_sp); sperr["fft_phase"].append(ph_sp)

            rel_sp = float(np.abs(target_gt - recon_sp).max()) / max(drange, 1e-12)
            p_spp, bg_time_sp, bg_axis_sp, xhat_spp = _train_residual(recon_sp, rel_sp)
            cr_spp = orig_bytes / (sp_bytes + nn_bytes)
            m_spp, ph_spp = _global_fft_err(target_gt, xhat_spp, N_FFT_SLICES)
            sperr_pipe["CR"].append(cr_spp); sperr_pipe["PSNR"].append(p_spp)
            sperr_pipe["fft_mag"].append(m_spp); sperr_pipe["fft_phase"].append(ph_spp)
            print(f"    SPERR {cr_sp:6.1f}x/{p_sp:5.1f}dB/mag={m_sp:.3g}/pha={ph_sp:.3g} | "
                  f"SPERR+Ours {cr_spp:6.1f}x/{p_spp:5.1f}dB/mag={m_spp:.3g}/pha={ph_spp:.3g}")

            if ADD_NEURLZ:
                p_spn, spn_params, spn_time, spn_eps, _hist_spn, enh_spn = run_neurlz(
                    target_gt, recon_sp, aux_list, shape, drange, rel_sp,
                    int(epochs * NEURLZ_EPOCH_MULT), neurlz_features,
                    time_budget=bg_time_sp, slice_axis=NEURLZ_SLICE_AXIS, return_history=True)
                enh_spn_orig = _unperm_view(enh_spn, NEURLZ_SLICE_AXIS)
                m_spn, ph_spn = _global_fft_err(target_gt, enh_spn_orig, N_FFT_SLICES)
                cr_spn = orig_bytes / (sp_bytes + spn_params * BYTES_PER_PARAM)
                sperr_neurlz["CR"].append(cr_spn); sperr_neurlz["PSNR"].append(p_spn)
                sperr_neurlz["fft_mag"].append(m_spn); sperr_neurlz["fft_phase"].append(ph_spn)
                print(f"    SPERR+NeurLZ {cr_spn:6.1f}x/{p_spn:5.1f}dB/mag={m_spn:.3g}/pha={ph_spn:.3g} "
                      f"[{spn_params:,}p] | BG {bg_time_sp:.1f}s ≈ NeurLZ {spn_time:.1f}s/{spn_eps}ep")
            del recon_sp
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        for D in (sperr, sperr_pipe, sperr_neurlz):
            if not D["CR"]:
                continue
            o = list(np.argsort(D["CR"]))
            for key in ("CR", "PSNR", "fft_mag", "fft_phase"):
                D[key] = [D[key][i] for i in o]

    return dict(sz3=sz3, pipe=pipe, sperr=sperr, sperr_pipe=sperr_pipe,
               neurlz=neurlz, sperr_neurlz=sperr_neurlz, bg_h=bg_h, n_params=n_params)


print("bench_field_fft ready")

# ─────────────────────────────────────────────────────────────────────────────
# FFT results cache (own directory -- doesn't touch SPERR.py's sperr_cache/)
# ─────────────────────────────────────────────────────────────────────────────
FFT_CACHE_DIR = "/home/sam/Halo_Finder/Final_design/SPERR/sperr_fft_cache"
os.makedirs(FFT_CACHE_DIR, exist_ok=True)
FORCE_RETRAIN = False

_CACHE_CFG_KEYS = [
    "SEED", "BO_ENABLE", "BO_N_TRIALS", "BO_N_STARTUP", "BO_ENQUEUE_LR", "BO_LR_MIN",
    "BO_LR_MAX", "BO_PHASE1_EPOCHS", "BO_MIN_GAIN_DB", "BO_SEARCH_DIRECTION",
    "EPOCHS_OVERRIDE", "BG_USE_AUX", "PROXY_DS", "ADD_NEURLZ", "N_FFT_SLICES",
    "NEURLZ_FEATURES", "NEURLZ_LR", "NEURLZ_BATCH", "NEURLZ_SLICE_AXIS",
    "NEURLZ_BEST_GUARD", "NEURLZ_POSTPROCESS", "NEURLZ_SINGLE_FIELD", "NEURLZ_EPOCH_MULT",
    "BYTES_PER_PARAM", "DETERMINISTIC", "BG_BATCH", "SPERR_MAX_CR",
]


def _fft_cache_path(name, target_file, shape, rel_errs, param_budget, epochs, n_aux, kw):
    g = globals()
    cfg = {k: g.get(k, None) for k in _CACHE_CFG_KEYS}
    sig = repr(dict(name=name, file=str(target_file), shape=tuple(shape),
                    rel=list(map(float, rel_errs)), params=int(param_budget),
                    epochs=int(epochs), n_aux=int(n_aux),
                    kw={k: kw[k] for k in sorted(kw)}, cfg=cfg))
    h = hashlib.md5(sig.encode()).hexdigest()[:12]
    safe = name.replace("/", "_").replace(" ", "_")
    return os.path.join(FFT_CACHE_DIR, f"{safe}__{h}.pkl")


def cached_bench_field_fft(name, target_gt, target_file, aux_list, shape, rel_errs,
                           param_budget, epochs, **kw):
    path = _fft_cache_path(name, target_file, shape, rel_errs, param_budget, epochs, len(aux_list), kw)
    if (not FORCE_RETRAIN) and os.path.isfile(path):
        with open(path, "rb") as f:
            r = pickle.load(f)
        print(f"[cache] HIT  {name}: loaded {os.path.basename(path)} (no retrain)")
        return r
    print(f"[cache] MISS {name}: training ...")
    r = bench_field_fft(name, target_gt, target_file, aux_list, shape, rel_errs, param_budget, epochs, **kw)
    with open(path, "wb") as f:
        pickle.dump(r, f)
    print(f"[cache] saved {name} -> {os.path.basename(path)}")
    return r


print("cached_bench_field_fft ready | FFT_CACHE_DIR =", FFT_CACHE_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# NeurLZ baseline (faithful re-impl of neurlz/train.py's recipe, MSE variant)
# ─────────────────────────────────────────────────────────────────────────────
ADD_NEURLZ        = True
NEURLZ_LR         = 1e-2
NEURLZ_SLICE_AXIS = 0
NEURLZ_BATCH      = 10
NEURLZ_MAX_PIXELS_PER_BATCH = 16 * 1024 * 1024
NEURLZ_VERBOSE    = True
NEURLZ_EVAL_EVERY = 1
NEURLZ_EPOCH_MULT = 1
NEURLZ_POSTPROCESS = False
NEURLZ_BEST_GUARD  = False
NEURLZ_FEATURES    = (4, 4, 4, 4, 4, 4)
NEURLZ_SINGLE_FIELD = False


def _basicunet_nparams(features, n_fields):
    with contextlib.redirect_stdout(io.StringIO()):
        m = BasicUNet(spatial_dims=2, features=tuple(features), act="gelu",
                      in_channels=int(n_fields), out_channels=1)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad); del m
    return n


def _neurlz_features_for_params(target_params, n_fields, lo=4, hi=384):
    a, b, best = lo, hi, lo
    while a <= b:
        mid = (a + b) // 2
        if _basicunet_nparams((mid,) * 6, n_fields) <= target_params:
            best = mid; a = mid + 1
        else:
            b = mid - 1
    w_hi = min(best + 1, hi)
    n_lo = _basicunet_nparams((best,) * 6, n_fields)
    n_hi = _basicunet_nparams((w_hi,) * 6, n_fields)
    w = best if abs(n_lo - target_params) <= abs(n_hi - target_params) else w_hi
    return (w,) * 6


def _mm(x, eps=1e-8):
    lo, hi = float(np.min(x)), float(np.max(x))
    return ((np.asarray(x, np.float32) - lo) / (hi - lo + eps)).astype(np.float32), (lo, hi)


def run_neurlz(target_gt, base_recon, aux_list, shape, drange, rel, epochs, features,
              time_budget=None, slice_axis=0, return_history=False):
    """SZ3 + NeurLZ on one base reconstruction.
    Returns (psnr, n_params, train_time, epochs[, hist, enh])."""
    if NEURLZ_SINGLE_FIELD:
        aux_list = []
    if int(slice_axis) != 0:
        _p = {1: (1, 0, 2), 2: (2, 0, 1)}[int(slice_axis)]
        target_gt  = np.ascontiguousarray(np.transpose(np.asarray(target_gt, np.float32), _p))
        base_recon = np.ascontiguousarray(np.transpose(np.asarray(base_recon, np.float32), _p))
        aux_list   = [np.ascontiguousarray(np.transpose(np.asarray(a, np.float32), _p)) for a in aux_list]
        shape = target_gt.shape
    D, H, W = shape
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    eff_batch = max(1, min(NEURLZ_BATCH, NEURLZ_MAX_PIXELS_PER_BATCH // (H * W)))
    if eff_batch < NEURLZ_BATCH:
        print(f"           [neurlz] slice {H}x{W}: batch {NEURLZ_BATCH} -> {eff_batch} (memory)")
    tgt = np.asarray(target_gt, np.float32)
    lq  = np.ascontiguousarray(base_recon, np.float32)
    fields = [lq] + [np.asarray(a, np.float32) for a in aux_list]
    n_fields = len(fields)
    lq_n = np.stack([_mm(f)[0] for f in fields], axis=1)
    err_n, (e_lo, e_hi) = _mm(tgt - lq)
    ph, pw = (-H) % 16, (-W) % 16
    pad = ((0, 0), (0, 0), (0, ph), (0, pw))
    Xlq  = torch.from_numpy(np.pad(lq_n, pad, mode="reflect"))
    Yerr = torch.from_numpy(np.pad(err_n[:, None], pad, mode="reflect"))

    set_seed(SEED)
    with contextlib.redirect_stdout(io.StringIO()):
        model = BasicUNet(spatial_dims=2, features=tuple(features), act="gelu",
                          in_channels=n_fields, out_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt   = torch.optim.Adam(model.parameters(), lr=NEURLZ_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
    mse    = torch.nn.MSELoss()
    idx = np.arange(D)

    def _enhanced():
        model.eval()
        out = lq.copy()
        with torch.no_grad():
            for st in range(0, D, eff_batch):
                bi = list(range(st, min(st + eff_batch, D)))
                pred = model(Xlq[bi].to(device)).cpu().numpy()[:, 0, :H, :W]
                out[bi] = lq[bi] + (pred * (e_hi - e_lo + 1e-8) + e_lo)
        model.train()
        if NEURLZ_POSTPROCESS:
            return _error_bounded_post_process(x_enhanced=out, x_prime=lq, absolute_error_bound=0.0,
                                               relative_error_bound=float(rel), verbose=False, a=1.0)
        return out

    use_budget = time_budget is not None and float(time_budget) > 0
    ep_cap     = 100000 if use_budget else int(epochs)
    budget_str = f"{float(time_budget):.1f}s" if use_budget else f"{int(epochs)}ep"
    base_psnr  = compute_psnr(tgt, lq, drange)
    best_psnr  = base_psnr
    best_state = None
    train_time, ep = 0.0, 0
    hist_t, hist_p = [], []
    model.train()
    while ep < ep_cap:
        np.random.shuffle(idx)
        tot, nb = 0.0, 0
        t_ep = time.perf_counter()
        for st in range(0, D, eff_batch):
            bi = idx[st:st + eff_batch]
            loss = mse(model(Xlq[bi].to(device)), Yerr[bi].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.item())
            nb += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t_ep
        ep += 1
        done = use_budget and train_time >= float(time_budget)
        eval_now = (ep % NEURLZ_EVAL_EVERY == 0) or done or (not use_budget and ep == int(epochs))
        if eval_now and (NEURLZ_BEST_GUARD or NEURLZ_VERBOSE or return_history):
            pe = compute_psnr(tgt, _enhanced(), drange)
            if NEURLZ_BEST_GUARD and pe > best_psnr:
                best_psnr  = float(pe)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if return_history:
                hist_t.append(float(train_time)); hist_p.append(float(pe))
            if NEURLZ_VERBOSE:
                _bs = f" | best {best_psnr:.2f}" if NEURLZ_BEST_GUARD else ""
                print(f"           [neurlz] ep {ep:3d} | {train_time:5.1f}/{budget_str} | MSE {tot/max(nb,1):.6f} | PSNR {pe:.2f} dB{_bs}")
        elif NEURLZ_VERBOSE:
            print(f"           [neurlz] ep {ep:3d} | {train_time:5.1f}/{budget_str} | MSE {tot/max(nb,1):.6f}")
        if done:
            break

    if NEURLZ_BEST_GUARD:
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            enh = _enhanced()
        else:
            enh = lq.copy()
        out_psnr = float(max(best_psnr, base_psnr))
    else:
        enh = _enhanced()
        out_psnr = compute_psnr(tgt, enh, drange)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if return_history:
        return out_psnr, int(n_params), float(train_time), int(ep), {"time": hist_t, "psnr": hist_p}, enh
    return out_psnr, int(n_params), float(train_time), int(ep)


print("run_neurlz ready | ADD_NEURLZ =", ADD_NEURLZ, "| NEURLZ_FEATURES =", NEURLZ_FEATURES)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset configs + runs
# Each block is gated by TASK: it runs when TASK=="all" (legacy, everything in one
# process), TASK==<its own key> (its dedicated SLURM process), or TASK=="plot"
# (re-invoked here so the plot process gets a cache HIT and can rebuild `results`
# without duplicating the config elsewhere).
# ─────────────────────────────────────────────────────────────────────────────
results = {}   # label -> r dict, in the order we want plotted

# ── NYX 512^3 (3 targets) — same configs as SPERR.py, for apples-to-apples CR points ──
NYX_DIR   = "/home/sam/Halo_Finder/halo_finder_v1/SDRBENCH-EXASKY-NYX-512x512x512/origin/"
NYX_SHAPE = (512, 512, 512)
NYX_ALL   = ["baryon_density", "dark_matter_density", "temperature",
             "velocity_x", "velocity_y", "velocity_z"]
NYX_REL = {
    "baryon_density":      [1e-6, 3e-6, 5e-6, 7e-6, 9e-6],
    "temperature":         [1e-4, 3e-4, 5e-4, 7e-4, 8e-4],
    "dark_matter_density": [1e-4, 3e-4, 5e-4, 7e-4, 8e-4],
}
NYX_PARAMS, NYX_EPOCHS, NYX_SPERR_OFF = 30000, 10, 17.0
NYX_SPERR_EXTRA_SPAN, NYX_SPERR_NEXTRA = 7, 0
NYX_TIME_BUDGET = 10.0   # seconds of Phase-2 "Ours" training per rel_err (NeurLZ matches this)
NYX_TASK_KEY = {"baryon_density": "nyx_b", "temperature": "nyx_t", "dark_matter_density": "nyx_d"}

for tname, tkey in NYX_TASK_KEY.items():
    if TASK not in ("all", "plot", tkey):
        continue
    set_seed(SEED)
    tgt = np.fromfile(NYX_DIR + tname + ".f32", dtype=np.float32).reshape(NYX_SHAPE)
    aux = [np.memmap(NYX_DIR + a + ".f32", dtype=np.float32, mode="r", shape=NYX_SHAPE)
           for a in NYX_ALL if a != tname]
    r = cached_bench_field_fft(f"NYX/{tname}", tgt, NYX_DIR + tname + ".f32", aux, NYX_SHAPE,
                               NYX_REL[tname], NYX_PARAMS, NYX_EPOCHS, sperr_psnr_offset=NYX_SPERR_OFF,
                               sperr_extra_span=NYX_SPERR_EXTRA_SPAN, sperr_n_extra=NYX_SPERR_NEXTRA,
                               time_budget=NYX_TIME_BUDGET)
    results[f"NYX — {tname}"] = r
    del tgt, aux
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ── Miranda 1024^3 ──
MIR_FILE  = "/home/sam/Halo_Finder/halo_finder_v1/miranda_1024x1024x1024_float32.raw"
MIR_SHAPE = (1024, 1024, 1024)
MIR_REL   = [5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2]
MIR_PARAMS, MIR_EPOCHS = 240000, 5
MIR_TIME_BUDGET = 80.0   # seconds of Phase-2 "Ours" training per rel_err (NeurLZ matches this)

if TASK in ("all", "plot", "miranda"):
    set_seed(SEED)
    mir = np.fromfile(MIR_FILE, dtype=np.float32).reshape(MIR_SHAPE)
    results["Miranda 1024³"] = cached_bench_field_fft("Miranda", mir, MIR_FILE, [], MIR_SHAPE,
                                                      MIR_REL, MIR_PARAMS, MIR_EPOCHS,
                                                      time_budget=MIR_TIME_BUDGET)
    del mir
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ── WarpX (wpx) 2048x256x256, forced axis2 (fast + clean; see SPERR.py notes) ──
WPX_RAW       = "/home/sam/Halo_Finder/halo_finder_v1/wpx-256_256_2048_double.raw"
WPX_REL = [1e-2, 2e-2, 3e-2, 4e-2, 5e-2]
WPX_PARAMS, WPX_EPOCHS = 30000, 10
WPX_TIME_BUDGET = 10.0   # seconds of Phase-2 "Ours" training per rel_err (NeurLZ matches this)

if TASK in ("all", "plot", "warpx"):
    _wpx = np.fromfile(WPX_RAW, dtype=np.float64).reshape(WPX_SRC_SHAPE).astype(np.float32)
    _wpx = np.ascontiguousarray(np.transpose(_wpx, (2, 0, 1)))   # -> (2048, 256, 256)
    WPX_SHAPE = _wpx.shape
    WPX_F32 = "/tmp/wpx_2048_256_256_f32.raw"
    _wpx.tofile(WPX_F32)
    set_seed(SEED)
    results["WarpX"] = cached_bench_field_fft("wpx", _wpx, WPX_F32, [], WPX_SHAPE, WPX_REL, WPX_PARAMS, WPX_EPOCHS,
                                              full_slice=True, full_slice_axis=2, bo_axes=[2],
                                              time_budget=WPX_TIME_BUDGET)
    del _wpx
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ── Magnetic Reconnection 512^3 ──
MAG_FILE  = "/home/sam/Halo_Finder/halo_finder_v1/magnetic_reconnection_512x512x512_float32.raw"
MAG_SHAPE = (512, 512, 512)
MAG_REL   = [8e-3, 9.5e-3, 1.5e-2, 2e-2, 2.5e-2]
MAG_PARAMS, MAG_EPOCHS = 30000, 10
MAG_TIME_BUDGET = 10.0   # seconds of Phase-2 "Ours" training per rel_err (NeurLZ matches this)

if TASK in ("all", "plot", "mag"):
    set_seed(SEED)
    mag = np.fromfile(MAG_FILE, dtype=np.float32).reshape(MAG_SHAPE)
    results["Magnetic Reconnection"] = cached_bench_field_fft("Magnetic", mag, MAG_FILE, [], MAG_SHAPE,
                                                              MAG_REL, MAG_PARAMS, MAG_EPOCHS,
                                                              time_budget=MAG_TIME_BUDGET)
    del mag
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ─────────────────────────────────────────────────────────────────────────────
# Combined 2x3 figures: FFT magnitude error vs CR, and FFT phase error vs CR.
# Same panel layout / hollow-marker restyle as SPERR.py's PSNR figure, just swap
# the y-axis metric. Two separate figures (magnitude, phase) rather than cramming
# 2 sub-axes into each of 6 panels, so each figure stays as readable as the PSNR
# one. Only runs for TASK in ("all", "plot") -- a single-dataset process (e.g.
# --task nyx_b) has an incomplete `results` and must not try to draw the figure.
# ─────────────────────────────────────────────────────────────────────────────
if TASK in ("all", "plot"):
    PANEL_ORDER = [
        "NYX — baryon_density", "NYX — temperature", "NYX — dark_matter_density",
        "Miranda 1024³", "WarpX", "Magnetic Reconnection",
    ]
    PANEL_TITLES = ["Baryon density", "Temperature", "Dark matter density",
                   "Miranda 1024³", "WarpX", "Magnetic Reconnection"]

    _SERIES_DEFS = [
        ("sz3", "SZ3"), ("pipe", "SZ3 + Ours"), ("sperr", "SPERR"), ("sperr_pipe", "SPERR + Ours"),
        ("neurlz", "SZ3 + NeurLZ"), ("sperr_neurlz", "SPERR + NeurLZ"),
    ]

    def _panel_series(r, metric_key):
        """List of (key, x, y, default_label) for every series with data, in canonical order.
        metric_key: 'fft_mag' or 'fft_phase'."""
        out = []
        for key, lbl in _SERIES_DEFS:
            d = r.get(key, {})
            if d.get("CR"):
                out.append((key, d["CR"], d[metric_key], lbl))
        return out

    FIGSIZE      = (18, 10)
    DPI          = 150
    TITLE_FS     = 20
    LABEL_FS     = 20
    TICK_FS      = 20
    LEGEND_FS    = 20
    MARKER_SCALE = 2.0
    LINE_SCALE   = 1.0
    GRID_ALPHA   = 0.30
    XLOG         = False
    YLOG         = True     # FFT error often spans orders of magnitude across the CR sweep

    # Same family-grouped styling as SPERR.py's PSNR figure: blue shades = SZ3
    # family, red/orange shades = SPERR family; marker+linestyle encode the role
    # (baseline / +Ours / +NeurLZ) and match across families.
    _SERIES_STYLE = {
        "sz3":          dict(color="#08519c", marker="o", ls="-"),
        "pipe":         dict(color="#4355b9", marker="s", ls="--"),
        "neurlz":       dict(color="#4292c6", marker="^", ls=":"),
        "sperr":        dict(color="#a50f15", marker="o", ls="-"),
        "sperr_pipe":   dict(color="#e6550d", marker="s", ls="--"),
        "sperr_neurlz": dict(color="#fdae6b", marker="^", ls=":"),
    }

    CUSTOM_LABELS = {
        "sz3": "SZ3", "pipe": "SZ3 + Ours", "sperr": "SPERR", "sperr_pipe": "SPERR + Ours",
        "neurlz": "SZ3 + NeurLZ", "sperr_neurlz": "SPERR + NeurLZ",
    }

    def plot_fft_metric(metric_key, ylabel, suptitle, save_name):
        fig, axes = plt.subplots(2, 3, figsize=FIGSIZE)
        axes_flat = axes.ravel()
        for ax, label, title in zip(axes_flat, PANEL_ORDER, PANEL_TITLES):
            series = _panel_series(results[label], metric_key)
            for key, x, y, default_lbl in series:
                style = _SERIES_STYLE[key]
                lbl = CUSTOM_LABELS.get(key, default_lbl) if CUSTOM_LABELS else default_lbl
                ax.plot(x, y, marker=style["marker"], linestyle=style["ls"], color=style["color"],
                        linewidth=1.8 * LINE_SCALE, markersize=7 * MARKER_SCALE,
                        markerfacecolor="none", markeredgewidth=2.0, label=lbl)
            if XLOG:
                ax.set_xscale("log")
            if YLOG:
                ax.set_yscale("log")
            ax.set_title(title, fontsize=TITLE_FS)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.grid(True, alpha=GRID_ALPHA, which="both")

        fig.supxlabel("Effective Compression Ratio", fontsize=LABEL_FS, fontweight="bold")
        fig.supylabel(ylabel, fontsize=LABEL_FS, fontweight="bold")

        _legend_ax = max(axes_flat, key=lambda a: len(a.get_lines()))
        handles, labels = _legend_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.96),
                  fontsize=LEGEND_FS, ncol=len(labels), frameon=True)
        fig.suptitle(suptitle, fontsize=15, y=1.02)

        plt.tight_layout()
        plt.savefig(save_name, dpi=DPI, bbox_inches="tight")
        plt.show()
        print("Saved:", save_name)


    plot_fft_metric("fft_mag", "FFT magnitude error", "",
                    "/home/sam/Halo_Finder/SPERR/sperr_fft_mag_all6.pdf")
    plot_fft_metric("fft_phase", "FFT phase error", "",
                    "/home/sam/Halo_Finder/SPERR/sperr_fft_phase_all6.pdf")


# # ─────────────────────────────────────────────────────────────────────────────
# # Combined 2x3 figures: FFT magnitude error vs CR, and FFT phase error vs CR.
# # Same panel layout / hollow-marker restyle as SPERR.py's PSNR figure, just swap
# # the y-axis metric. Two separate figures (magnitude, phase) rather than cramming
# # 2 sub-axes into each of 6 panels, so each figure stays as readable as the PSNR
# # one. Only runs for TASK in ("all", "plot") -- a single-dataset process (e.g.
# # --task nyx_b) has an incomplete `results` and must not try to draw the figure.
# # ─────────────────────────────────────────────────────────────────────────────
# if TASK in ("all", "plot"):
#     PANEL_ORDER = [
#         "NYX — baryon_density", "NYX — temperature", "NYX — dark_matter_density",
#         "Miranda 1024³", "WarpX", "Magnetic Reconnection",
#     ]
#     PANEL_TITLES = ["Baryon density", "Temperature", "Dark matter density",
#                    "Miranda 1024³", "WarpX", "Magnetic Reconnection"]

#     _SERIES_DEFS = [
#         ("sz3", "SZ3"), ("pipe", "SZ3 + Ours"), ("sperr", "SPERR"), ("sperr_pipe", "SPERR + Ours"),
#         ("neurlz", "SZ3 + NeurLZ"), ("sperr_neurlz", "SPERR + NeurLZ"),
#     ]

#     def _panel_series(r, metric_key):
#         """List of (x, y, default_label) for every series with data, in canonical order.
#         metric_key: 'fft_mag' or 'fft_phase'."""
#         out = []
#         for key, lbl in _SERIES_DEFS:
#             d = r.get(key, {})
#             if d.get("CR"):
#                 out.append((d["CR"], d[metric_key], lbl))
#         return out

#     FIGSIZE      = (18, 10)
#     DPI          = 150
#     TITLE_FS     = 20
#     LABEL_FS     = 20
#     TICK_FS      = 20
#     LEGEND_FS    = 20
#     MARKER_SCALE = 2.0
#     LINE_SCALE   = 1.0
#     GRID_ALPHA   = 0.30
#     XLOG         = False
#     YLOG         = True     # FFT error often spans orders of magnitude across the CR sweep

#     CUSTOM_MARKERS    = ["o", "s", "^", "D", "v", "P", "*", "X"]
#     CUSTOM_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
#     CUSTOM_LABELS = ["SZ3", "SZ3 + Ours", "SPERR", "SPERR + Ours", "SZ3 + NeurLZ", "SPERR + NeurLZ"]

#     def plot_fft_metric(metric_key, ylabel, suptitle, save_name):
#         fig, axes = plt.subplots(2, 3, figsize=FIGSIZE)
#         axes_flat = axes.ravel()
#         for ax, label, title in zip(axes_flat, PANEL_ORDER, PANEL_TITLES):
#             series = _panel_series(results[label], metric_key)
#             for i, (x, y, default_lbl) in enumerate(series):
#                 m  = CUSTOM_MARKERS[i % len(CUSTOM_MARKERS)]
#                 ls = CUSTOM_LINESTYLES[i % len(CUSTOM_LINESTYLES)]
#                 lbl = CUSTOM_LABELS[i] if CUSTOM_LABELS and i < len(CUSTOM_LABELS) else default_lbl
#                 ax.plot(x, y, marker=m, linestyle=ls, linewidth=1.8 * LINE_SCALE,
#                         markersize=7 * MARKER_SCALE, markerfacecolor="none", markeredgewidth=2.0,
#                         label=lbl)
#             if XLOG:
#                 ax.set_xscale("log")
#             if YLOG:
#                 ax.set_yscale("log")
#             ax.set_title(title, fontsize=TITLE_FS)
#             ax.tick_params(axis="both", labelsize=TICK_FS)
#             ax.grid(True, alpha=GRID_ALPHA, which="both")

#         fig.supxlabel("Effective Compression Ratio", fontsize=LABEL_FS, fontweight="bold")
#         fig.supylabel(ylabel, fontsize=LABEL_FS, fontweight="bold")

#         _legend_ax = max(axes_flat, key=lambda a: len(a.get_lines()))
#         handles, labels = _legend_ax.get_legend_handles_labels()
#         fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.96),
#                   fontsize=LEGEND_FS, ncol=len(labels), frameon=True)
#         fig.suptitle(suptitle, fontsize=15, y=1.02)

#         plt.tight_layout()
#         plt.savefig(save_name, dpi=DPI, bbox_inches="tight")
#         plt.show()
#         print("Saved:", save_name)


#     plot_fft_metric("fft_mag", "FFT magnitude error", "",
#                     "/Users/923714256/Final_design_for_BO_adaptive_compression/SPERR/sperr_fft_mag_all6.pdf")
#     plot_fft_metric("fft_phase", "FFT phase error", "",
#                     "/Users/923714256/Final_design_for_BO_adaptive_compression/sperr_fft_phase_all6.pdf")
