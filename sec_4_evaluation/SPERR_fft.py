"""SPERR_fft.py — standalone script version of SPERR_fft.ipynb.

Companion to SPERR.py: same 6 methods (SZ3, SZ3+Ours, SPERR, SPERR+Ours, SZ3+NeurLZ,
SPERR+NeurLZ) and same 4 datasets (NYX x3 targets, Miranda, WarpX, Magnetic Reconnection
-- 6 panels total), but the y-axis is FFT magnitude / phase error instead of PSNR:

    mag_err, phase_err = _global_fft_err(x_true, x_hat)

a strided-2D-FFT metric (mean |delta FFT magnitude| and mean |delta wrapped phase| over
~32 z-slices), reused as-is from sec_3_3_frequency_loss/fft_err.ipynb.

IMPORTANT: SPERR.py's cache only stores CR/PSNR numbers -- no model weights or
reconstructed volumes were ever saved, so FFT error can't be computed "for free" from
that cache. This script re-runs the SAME two-phase BO + training pipeline (same configs,
same seed) so results are apples-to-apples with the PSNR script, but computes FFT error
right when each reconstruction is already in memory (no extra training cost) and caches
its OWN results (sperr_fft_cache/) so future re-runs of THIS script are free.

Same --task CLI as SPERR.py, for SLURM-style per-dataset parallel launches:
  --task {nyx_b,nyx_t,nyx_d,miranda,mag,qmcpack,aux_prep,neurlz_long,all}   (default: all)
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
SCRIPTS_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "base_script"))
if SCRIPTS_PATH not in sys.path:
    sys.path.append(SCRIPTS_PATH)
from local_paths import P          # env var > <repo>/local_paths.env > placeholder (see local_paths.env.example)
SPERR_BIN = P("ADAMIT_SPERR_BIN")
SZ3_LIB   = P("ADAMIT_SZ3_LIB")
PYSZ_PATH = P("ADAMIT_PYSZ")
if PYSZ_PATH not in sys.path:
    sys.path.append(PYSZ_PATH)

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

# Deterministic cuDNN, not autotune: with autotune ON, two runs of the identical
# config chose different algorithms, which moved the Phase-1 trial scores by ~0.008 dB
# -- enough to flip the argmax and the adopted slice direction. Measured free in steady
# state (28 s/epoch either way on Miranda); autotune's first epoch is actually slower.
# Autotune (not deterministic cuDNN) is the default: deterministic fp32 convolutions
# are slower here, so the fixed wall-clock budget buys fewer steps -- measured ~2 dB
# lower PSNR across all ten NYX operating points. The cost of that speed is that two
# runs of an identical config can pick different Phase-1 configurations (trial scores
# move by ~0.008 dB, enough to flip an argmax); repeat runs and report a median rather
# than paying 2 dB for a determinism that would still not pin Phase 2's step count.
# SPERR_DET=1 switches to deterministic cuDNN.
DETERMINISTIC = bool(int(os.environ.get('SPERR_DET', '0')))
torch.backends.cudnn.benchmark = not DETERMINISTIC
torch.backends.cudnn.deterministic = DETERMINISTIC
set_seed(SEED)


def compute_psnr(x_true, x_hat, drange):
    mse = float(np.mean((np.asarray(x_true, np.float64) - np.asarray(x_hat, np.float64)) ** 2))
    return 100.0 if mse == 0 else 20.0 * np.log10(drange) - 10.0 * np.log10(mse)


def _global_fft_err(x_true, x_hat, n_slices=None):
    """Global FFT magnitude / phase L1 error over z-slices.

    n_slices=None uses EVERY slice. The previous default of 32 came from
    sec_3_3_frequency_loss/fft_err.ipynb and was never re-checked for this paper.
    Measured against the full-volume value on NYX baryon_density (512^3, SZ3 @
    rel=4.02e-06): 32 slices give magnitude to within 0.89% -- fine -- but phase to
    only 11.6%, and the phase estimate is still non-monotonic at N=128 (16.0% / 11.6%
    / 15.5% / 7.6% at N=16/32/64/128), i.e. it has not converged. Full volume costs
    4.4 s against 0.27 s for 32 slices, negligible next to a 10-80 s training budget,
    so the sampling is dropped entirely."""
    x_true = np.asarray(x_true); x_hat = np.asarray(x_hat)
    D = x_true.shape[0]
    step = 1 if n_slices is None else max(1, D // int(n_slices))
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


def run_sperr(data_file, target_gt, shape, drange, target_psnr, dtype=np.float32):
    """SPERR at a target PSNR. Returns (CR, PSNR, recon, nbytes).
    dtype: precision of BOTH the on-disk data_file and the decompressed output --
    np.float32 (default) or np.float64 for a dataset (WarpX) whose native format is
    double; must match whatever dtype data_file was actually written in.
    Tmp filenames use os.getpid()+nanosecond-time (NOT numpy's global RNG) so parallel
    dataset runs can never collide on /tmp paths; failures are printed, not silent."""
    W, H, D = shape[2], shape[1], shape[0]
    is64 = np.dtype(dtype) == np.dtype(np.float64)
    bpe  = 8 if is64 else 4
    tag = f"{os.getpid()}_{time.time_ns()}"
    bit = f"/tmp/sperr_{tag}.bit"; rec = f"/tmp/sperr_{tag}.dec.{'f64' if is64 else 'f32'}"
    p1 = subprocess.run([SPERR_BIN, "-c", "--ftype", ("64" if is64 else "32"),
                        "--dims", str(W), str(H), str(D), "--psnr", f"{float(target_psnr):.4f}",
                        "--bitstream", bit, data_file], capture_output=True, text=True, env=_SPERR_ENV)
    if not os.path.exists(bit):
        print(f"  [run_sperr] COMPRESS FAILED (psnr={target_psnr:.2f}) rc={p1.returncode} "
              f"stdout={p1.stdout!r} stderr={p1.stderr!r}")
        return None, None, None, None
    nbytes = os.path.getsize(bit)
    p2 = subprocess.run([SPERR_BIN, "-d", ("--decomp_d" if is64 else "--decomp_f"), rec, bit],
                        capture_output=True, text=True, env=_SPERR_ENV)
    cr = psnr = recon = None
    if os.path.exists(rec):
        recon = np.fromfile(rec, dtype=dtype).reshape(shape)
        psnr  = compute_psnr(target_gt, recon, drange)
        cr    = (int(np.prod(shape)) * bpe) / nbytes
    else:
        print(f"  [run_sperr] DECOMPRESS FAILED (psnr={target_psnr:.2f}) rc={p2.returncode} "
              f"stdout={p2.stdout!r} stderr={p2.stderr!r}")
    for f in (bit, rec):
        if os.path.exists(f):
            os.remove(f)
    return cr, psnr, recon, nbytes


def _sperr_psnr_for_cr(data_file, target_gt, shape, drange, target_cr, lo=1.0, hi=250.0, iters=10, dtype=np.float32):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cr, _, _, _ = run_sperr(data_file, target_gt, shape, drange, mid, dtype=dtype)
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
_parser.add_argument("--task", choices=["nyx_b", "nyx_t", "nyx_d", "miranda", "mag", "qmcpack", "all", "aux_prep", "neurlz_long"],
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
BO_LR_MIN           = float(os.environ.get('SPERR_LR_MIN', '1e-3'))   # paper-final (2026-08-27): search [1e-3, 1e-2]
# 3e-3, not 1e-2. Every configuration that failed outright in our runs -- four SPERR
# points on NYX with the trust gates disabled, and two Miranda points where the gates
# DID pass -- had adopted an lr at an endpoint of the search interval. A proxy trial is
# short enough that a large lr always looks best on it, but at full resolution the same
# lr fails to converge inside the budget and the error-bounded clamp returns the base
# reconstruction (+0.00 dB while still paying the model's CR overhead). Narrowing the
# top of the interval removes that failure mode without adding any mechanism.
BO_LR_WARMUP_FRAC   = 5      # Phase-1 trial warmup = steps_per_epoch // this
BO_LR_MAX           = float(os.environ.get('SPERR_LR_MAX', '1e-2'))   # paper (Sec. 3.5): search [1e-3, 1e-2]; author
                             # decision 2026-08-28 (late): the paper's window is kept as written. For the record: the
                             # plain argmax occasionally adopts an lr at the top of the window that does not converge
                             # inside the Phase-2 budget and the guard then returns the base reconstruction (+0.00 dB) --
                             # 2 of 120 operating points on 08-28 at 1e-2 (Miranda-SPERR CR~500 @ 6.7e-3, QMCPack
                             # @ 8.9e-3), and the same thing happened once at a 5e-3 cap (QMCPack CR~500 @ 4.9e-3).
                             # lr-spread trust gates do not help (they catch 2/7 collapses while overriding 76 healthy picks).
BO_PHASE1_EPOCHS    = 3
BO_MIN_GAIN_DB      = 0.30
# Trust gate for Phase 1. The old gate compared the BEST trial against the proxy's own
# base reconstruction (best_gain = p - base_p) and refused to act unless the model beat
# the base by BO_MIN_GAIN_DB on the proxy. That is the wrong quantity: a proxy trial
# trains for a fraction of a second, so it essentially never beats the base -- measured
# +0.00 dB on 170 of ~300 logged decisions, and the gate fired on 56/60 operating points
# of the six-field run, discarding the search almost every time. What the gate should
# ask is whether the CONFIGURATIONS are distinguishable from each other, which is what
# the archived standalone BO study (NYX_Magnetic.ipynb) actually observes: there the
# three slice directions score 79.75 / 81.08 / 92.66 dB on the proxy -- a 13 dB spread --
# and its pick lands within 0.23 dB of the true full-resolution optimum. So the gate now
# tests the spread across completed trials.
BO_MIN_SPREAD_DB    = float(os.environ.get('SPERR_TAU_AXIS', '-1'))   # paper-final: gates DISABLED (plain argmax, Sec. 3.5); -1 never fires
BO_MIN_LR_SPREAD_DB = float(os.environ.get('SPERR_TAU_LR', '-1'))   # paper-final: disabled
# A proxy that ranks the SMALLEST learning rate highest is not reporting a good learning
# rate -- it is reporting that it has not trained long enough to distinguish anything, so
# the least-perturbed model wins by staying closest to the base. Phase 2 then gets 85% of
# the budget, for which "barely train" cannot be the right advice. Measured on Miranda,
# whose 0.72 s proxy trials sit in exactly that regime: the three lowest learning rates
# all tie at the top and adopting one costs 2.99 dB at full resolution, against 0.18 dB
# for falling back to the default. NYX, whose proxy prefers a LARGE learning rate, is
# unaffected. The test is on the adopted value's rank, not on a new threshold.
BO_LR_LOW_REJECT_FRAC = -1.0   # paper-final: disabled (with the 1e-3 floor there is no damage regime)
# The gate is split because the two search dimensions carry very different signal. On
# NYX the three directions score 102.5 / 102.5 / 99.1 dB on the proxy -- a real 3.4 dB
# axis spread -- while the lr values tried on the winning axis land within ~0.05 dB of
# each other. A single combined gate therefore passed on the strength of the axis
# signal and then adopted whatever lr happened to top the noise: two SPERR points
# picked the interval endpoints (1e-2 and 1e-4) against 124.6 / 120.8 dB baselines,
# learned nothing in 9 s, and the error-bound clamp returned the baseline -- +0.00 dB
# while still paying the model's CR overhead. Both quantities are read off the trials
# already completed, so splitting the gate costs no extra time.
# 0.15, not 0.10. At 10% a NYX trial got 0.06 s of pure training, which buys ~20 steps
# on the proxy -- too few for the learning rate to matter: the ten trials on the winning
# direction spanned 0.00-0.04 dB, i.e. the proxy could not rank lr at all and the trust
# gate (correctly) fell back to the default every time. Raising the split to 15% is the
# smallest change that gives the search something to measure; Phase 2 still keeps 85%.
# Paper decision (2026-08-14): the six-field results, Table 2 and Figs 11/12 were all
# produced at the 10%/90% split, so 10% is the reported protocol; the 15% experiment
# above is documented for the record and can be re-enabled with SPERR_P1_SPLIT=0.15.
BO_TIME_SPLIT       = float(os.environ.get('SPERR_P1_SPLIT', '0.10'))
BG_SCHED_TIME_CALIBRATE = True  # time-budget mode: after epoch 1, re-fit the cosine lr to the steps that fit the budget
BG_USE_AUX          = bool(int(os.environ.get('SPERR_USE_AUX', '1')))   # 0 = single-field ablation (no sibling channels; NeurLZ in_ch=1 too)
EPOCHS_OVERRIDE     = None
BO_SEARCH_DIRECTION = True
N_FFT_SLICES        = None   # z-slices used by _global_fft_err; None = EVERY slice (paper
                             # decision 2026-08-14: full-volume metric; the earlier 32-slice
                             # shortcut agreed to 1-4% in absolute error but was never principled)

BG_FREQ_WEIGHT      = 0.5    # w_freq   (paper Sec. 3.3.2)
BG_FFT_PHASE_WEIGHT = float(os.environ.get('SPERR_PHASE_W', '1.0'))   # lambda_phase (paper Fig. 5: 1); 0.5 in the 08-07 runs
BG_SAMPLE_MODE      = os.environ.get('SPERR_SAMPLE_MODE', 'shuffled')  # paper-final (2026-08-28 evening, Sec. 3.1): every epoch
                             # visits every depth slice exactly once in a random order (bg_sampling._shuffled_z; the
                             # same loop NeurLZ uses). 'sequential' (in depth order) was the 08-27 setting: identical
                             # within noise on NYX/Miranda/Magnetic, but on QMCPack (33120 slices, ~3 ms/step) the 54 s
                             # budget covers only ~18k steps, so in-order sampling never trains the second half of the
                             # orbitals and 5/10 operating points collapsed to +0.00 dB; shuffled gives +2.4-2.6 dB.
AUX_MODE            = os.environ.get('SPERR_AUX_MODE', 'cr_matched')   # sibling-field protocol (Sec. 4.1):
                             # 'cr_matched' (paper-final 2026-08-28): the siblings are archived by the target's
                             #   own base compressor at the AUX_CR_LEVELS entry closest to the target's CR, and
                             #   BOTH training and inference use those decompressed siblings -- the model never
                             #   sees anything the decoder cannot reproduce (NeurLZ's reference code likewise
                             #   feeds lossy fields on every input channel).
                             # 'orig': originals everywhere (the pre-08-27 runs; not decoder-reproducible).
                             # The 08-27 variant (originals at training, siblings SZ3-decompressed at rel=1e-5
                             # at inference) collapsed NYX temperature to +0.00 dB: rel=1e-5 of baryon density's
                             # 1.16e5 range is 1.16, i.e. 2.3x that field's MEDIAN value -- its bulk is wiped
                             # out while its PSNR still reads 105 dB, so a model trained on the original could
                             # not use it. Train/inference consistency is what NeurLZ has and what we need.
AUX_CR_LEVELS       = tuple(int(x) for x in os.environ.get('SPERR_AUX_CR_LEVELS', '100,200,300,400,500,600').split(','))
BG_SCHED_STEP_CALIB = bool(int(os.environ.get('SPERR_STEP_CALIB', '1')))   # Phase-2 cosine planned from the measured
                             # step cost when ONE epoch would exceed 80% of the budget (bg_stage.py). Only QMCPack
                             # trips it (33120 sequential slices/epoch ≈ the whole 54 s budget): before this, its
                             # lr never decayed and every band picked above ~2e-3 ended at +0.00 dB.

# ── visualization export (opt-in via env SPERR_SAVE_RECONS_DIR) ─────────────────────────────
# Writes every operating point's base / +Ours / +NeurLZ reconstruction as float32 raw (.f32, same
# (z,y,x) layout as the source files) plus legacy-VTK STRUCTURED_POINTS (.vtk, opens directly in
# ParaView), and signed error volumes vs the original (err_*.f32). SPERR_ONLY_BANDS="4" (indices into
# the dataset's rel list) restricts a run to the chosen band(s); the SPERR side is then bisected to
# the SAME CR as that SZ3 band (an aligned-CR pair) instead of the 5-point PSNR sweep. The cache hash
# covers rel_errs, so such runs never collide with the paper's pkls.
SAVE_RECONS_DIR  = os.environ.get('SPERR_SAVE_RECONS_DIR')
SAVE_RECONS_ONLY = [x.strip() for x in os.environ.get('SPERR_SAVE_RECONS_ONLY', '').split(',') if x.strip()]   # e.g. "ours": only that volume, .f32 only
AUX_FIELDS       = [x.strip() for x in os.environ.get('SPERR_AUX_FIELDS', '').split(',') if x.strip()]         # ablation: restrict the NYX sibling set
AUX_ENHANCED_DIR = os.environ.get('SPERR_AUX_ENHANCED_DIR')   # cascade: a sibling that has an exported enhanced volume
                             # (<name>_<compressor>_cr*/ours.f32 under this dir, from an earlier stage's SPERR_SAVE_RECONS_DIR)
                             # is fed ENHANCED instead of merely decompressed -- the decoder holds it after that stage.
ONLY_BANDS      = [int(x) for x in os.environ.get('SPERR_ONLY_BANDS', '').split(',') if x.strip() != '']


def _write_vtk(path, arr, name="value"):
    a = np.ascontiguousarray(np.asarray(arr, np.float32))
    D, H, W = a.shape
    with open(path, "wb") as fh:
        fh.write((f"# vtk DataFile Version 3.0\n{name}\nBINARY\nDATASET STRUCTURED_POINTS\n"
                  f"DIMENSIONS {W} {H} {D}\nORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA {D * H * W}\n"
                  f"SCALARS {name} float 1\nLOOKUP_TABLE default\n").encode())
        fh.write(a.astype('>f4').tobytes())


def _save_recons(name, base_tag, cr_base, gt, meta, **vols):
    if not SAVE_RECONS_DIR:
        return
    import json
    d = os.path.join(SAVE_RECONS_DIR, f"{name.replace('/', '_')}_{base_tag}_cr{float(cr_base):.0f}")
    os.makedirs(d, exist_ok=True)
    mp = os.path.join(d, "meta.json")
    m = json.load(open(mp)) if os.path.isfile(mp) else {}
    m.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in meta.items()})
    gt32 = np.asarray(gt, np.float32)
    m.update(dict(shape=list(gt32.shape), dtype="float32", layout="(z,y,x) C-order, x fastest; .vtk DIMENSIONS = W H D"))
    for k, v in vols.items():
        if v is None or (SAVE_RECONS_ONLY and k not in SAVE_RECONS_ONLY):
            continue
        v32 = np.ascontiguousarray(np.asarray(v, np.float32))
        v32.tofile(os.path.join(d, f"{k}.f32"))
        if not SAVE_RECONS_ONLY:                      # full export: ParaView file + signed error volume
            _write_vtk(os.path.join(d, f"{k}.vtk"), v32, k)
            np.ascontiguousarray(v32 - gt32).tofile(os.path.join(d, f"err_{k}.f32"))
        m[f"psnr_{k}"] = float(compute_psnr(gt32, v32, float(gt32.max() - gt32.min()) or 1.0))
    json.dump(m, open(mp, "w"), indent=1)
    print(f"  [viz] saved {list(vols)} -> {d}")


def _enhanced_sibling(a_file, compressor, cr_base, shape):
    """Cascade lookup: the enhanced volume of sibling `a_file` exported by an earlier stage
    (<NYX_stem>_<compressor>_cr*/ours.f32 under AUX_ENHANCED_DIR) whose CR is closest (log) to
    the target's CR. Returns (volume, dir) or None."""
    import glob as _glob, re as _re
    stem = os.path.basename(str(a_file)).rsplit('.', 1)[0]
    cands = []
    for d in _glob.glob(os.path.join(AUX_ENHANCED_DIR, f"*_{stem}_{compressor}_cr*")):
        m = _re.search(r"_cr(\d+)$", d)
        if m and os.path.isfile(os.path.join(d, "ours.f32")):
            cands.append((abs(np.log(float(m.group(1))) - np.log(max(cr_base, 1e-9))), d))
    if not cands:
        return None
    d = min(cands)[1]
    return np.ascontiguousarray(np.fromfile(os.path.join(d, "ours.f32"), np.float32).reshape(shape)), d


def _sperr_compress_file(data_file, shape, dtype, target_psnr):
    W, H, D = shape[2], shape[1], shape[0]
    is64 = np.dtype(dtype) == np.dtype(np.float64)
    bit = f"/tmp/sperr_aux_{os.getpid()}_{time.time_ns()}.bit"
    subprocess.run([SPERR_BIN, "-c", "--ftype", ("64" if is64 else "32"), "--dims", str(W), str(H), str(D),
                    "--psnr", f"{float(target_psnr):.4f}", "--bitstream", bit, str(data_file)],
                   capture_output=True, text=True, env=_SPERR_ENV)
    return (bit, os.path.getsize(bit)) if os.path.exists(bit) else (None, None)


def _sperr_decompress_file(bit, shape, dtype):
    is64 = np.dtype(dtype) == np.dtype(np.float64)
    rec = f"/tmp/sperr_aux_{os.getpid()}_{time.time_ns()}.dec"
    subprocess.run([SPERR_BIN, "-d", ("--decomp_d" if is64 else "--decomp_f"), rec, bit],
                   capture_output=True, text=True, env=_SPERR_ENV)
    arr = np.ascontiguousarray(np.fromfile(rec, dtype=dtype).reshape(shape), np.float32)
    os.remove(rec)
    return arr


def _aux_stream_paths(a_file, level, compressor):
    d = os.path.join(FFT_CACHE_DIR, "aux_streams")
    os.makedirs(d, exist_ok=True)
    base = os.path.basename(str(a_file)).replace(".", "_")
    stem = os.path.join(d, f"{compressor}_{base}_cr{int(level)}")
    return stem + ".bin", stem + ".json"


def _aux_at_cr_level(a, a_file, shape, level, compressor, dtype=np.float32, tol=0.03, iters=12):
    """Sibling `a` as the decoder holds it: archived by `compressor` ('sz3' | 'sperr') at
    CR ~= level (bisection, within tol). The bitstream is cached on disk under
    (compressor, file, level), so every target field and every run reuses the same
    archive. Returns (decompressed float32 volume, achieved CR, compressor knob)."""
    import json
    bit, meta = _aux_stream_paths(a_file, level, compressor)
    nbytes_orig = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if os.path.isfile(bit) and os.path.isfile(meta):
        m = json.load(open(meta))
        if compressor == "sz3":
            dec = np.ascontiguousarray(sz_engine.decompress(np.fromfile(bit, np.uint8), shape, np.float32), np.float32)
        else:
            dec = _sperr_decompress_file(bit, shape, dtype)
        return dec, float(m["cr"]), float(m["knob"])
    best = None   # (|log(cr/level)|, cr, knob, stream)
    if compressor == "sz3":
        a32 = np.ascontiguousarray(a, np.float32)
        lo, hi = -8.0, -1.0                      # log10(rel)
        for _ in range(iters):
            mid = 0.5 * (lo + hi); rel = 10.0 ** mid
            b, _ = sz_engine.compress(a32, 1, 0, float(rel), 0)
            cr = nbytes_orig / len(b); d = abs(np.log(cr / level))
            if best is None or d < best[0]:
                best = (d, cr, rel, np.asarray(b, np.uint8).copy())
            if d < tol:
                break
            if cr > level: hi = mid              # too coarse -> tighter bound
            else:          lo = mid
        best[3].tofile(bit)
        dec = np.ascontiguousarray(sz_engine.decompress(best[3], shape, np.float32), np.float32)
    elif compressor == "sperr":
        lo, hi = 1.0, 250.0                      # sperr3d --psnr
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            tmp, nb = _sperr_compress_file(a_file, shape, dtype, mid)
            if tmp is None:
                lo = mid; continue
            cr = nbytes_orig / nb; d = abs(np.log(cr / level))
            if best is None or d < best[0]:
                if best is not None and os.path.exists(best[3]):
                    os.remove(best[3])
                best = (d, cr, mid, tmp)
            else:
                os.remove(tmp)
            if d < tol:
                break
            if cr > level: lo = mid              # too coarse -> higher PSNR target
            else:          hi = mid
        os.replace(best[3], bit)
        dec = _sperr_decompress_file(bit, shape, dtype)
    else:
        raise ValueError(f"unknown compressor {compressor!r}")
    json.dump({"cr": float(best[1]), "knob": float(best[2]), "level": int(level), "compressor": compressor,
               "file": str(a_file), "nbytes": int(os.path.getsize(bit))}, open(meta, "w"))
    return dec, float(best[1]), float(best[2])


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
                    bg_low_w=0.2, bg_mid_w=0.5, bg_high_w=1.0, bo_axes=None, time_budget=None,
                    src_dtype=np.float32, bo_proxy_stride=None,
                    bo_proxy_spatial=None, bo_eval_slices=None, aux_files=None):
    """Same two-phase BO + SZ3/SPERR/model/NeurLZ pipeline as SPERR.py's bench_field,
    aux_files: on-disk paths of the aux siblings (same order as aux_list); needed by the
    AUX_MODE='cr_matched' protocol (SPERR archives from a file, and the streams are
    cached on disk under the file name).
    (bo_proxy_stride: see SPERR.py's bench_field docstring -- same per-axis proxy-stride
    override, same default.)
    bo_proxy_spatial / bo_eval_slices: opt-in fast Phase-1. Setting bo_eval_slices
    switches _phase1_best to a lean trial loop: spatial stride bo_proxy_spatial
    (instead of PROXY_DS), no per-epoch evaluator inside train_bg_only, and each trial
    scored by one inference over the middle bo_eval_slices proxy slices only. Measured
    on NYX (proxy 64x128x128, steady state past cuDNN warmup): 0.07 s/trial, so all
    BO_N_TRIALS=10 trials fit inside the 10% Phase-1 allocation of a 10 s budget --
    the default path fits only 1-2 because a trial costs ~0.44 s and the timeout only
    stops NEW trials. One-time per-dataset costs (target/aux proxy construction, cuDNN
    warmup) are excluded from the per-point Phase-1 clock, like SZ3 compression itself.
    but each series dict also carries fft_mag/fft_phase (via _global_fft_err), computed
    on the SAME reconstruction used for PSNR (no extra training, one extra FFT per point).
    time_budget: if set, Phase-2 "Ours" training (per rel_err) is capped by wall-clock
    seconds instead of `epochs` -- see SPERR.py's bench_field for the full rationale.
    src_dtype: dataset's native on-disk precision (np.float32 default, np.float64 for
    WarpX) -- see SPERR.py's bench_field docstring for the full rationale; the BO
    proxy / neural-net paths already downcast to fp32 internally regardless.
    """
    set_seed(SEED)
    target_gt = np.asarray(target_gt, src_dtype)
    _bpe = np.dtype(src_dtype).itemsize
    if EPOCHS_OVERRIDE:
        epochs = int(EPOCHS_OVERRIDE)
    if not BG_USE_AUX:
        aux_list = []
    drange    = float(target_gt.max() - target_gt.min())
    n_fields  = 1 + len(aux_list)
    bg_h      = bg_h_for_params(param_budget, shape, n_fields)
    n_params, nn_bytes = estimate_bg_model_param_bytes(
        n_fields=n_fields, shape=shape, bg_arch="spatial", bg_h=bg_h, dtype_bytes=BYTES_PER_PARAM)
    orig_bytes = int(np.prod(shape)) * _bpe
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

    _bo_elapsed = [0.0]   # actual Phase-1 (BO) wall time of the most recent _phase1_best call

    _bo_ta_cache = {}     # fast path: axis -> (tgt_proxy, aux_proxies, idx); target/aux
                          # never change across the 10 operating points, so build once
    _bo_warmed = [False]  # fast path: first trial pays cuDNN plan creation (~1.3 s);
                          # run one throwaway trial before starting the per-point clock

    def _phase1_best_fast(base_recon, base_rel):
        """Lean Phase-1: same TPE search, enqueue, gain gate and fallback semantics as
        _phase1_best below, but with the three cost sinks removed that made the default
        path overshoot its cap ~8x on NYX (proxy rebuilt from scratch every point: 6.5 s;
        evaluator running full-proxy inference at init + every epoch + once more at the
        end: ~3x 0.3 s; all outside the Optuna timeout, which only stops NEW trials)."""
        axes = (list(bo_axes) if bo_axes is not None else
               ([0, 1, 2] if BO_SEARCH_DIRECTION else [full_slice_axis if full_slice else 0]))
        sp = int(bo_proxy_spatial) if bo_proxy_spatial else PROXY_DS
        _dspf = ((lambda x: np.ascontiguousarray(x[:, ::sp, ::sp])) if sp > 1 else (lambda x: x))
        full_shape = np.asarray(target_gt).shape

        for k in axes:                      # one-time per dataset, outside the clock
            if k in _bo_ta_cache:
                continue
            Dk     = int(full_shape[k])
            stride = int((bo_proxy_stride or {}).get(k, PROXY_DS))
            idx    = np.arange(0, Dk, stride)
            _bo_ta_cache[k] = (_dspf(_take_perm(target_gt, k, idx)),
                               [_dspf(_take_perm(a, k, idx)) for a in aux_list], idx)

        _p1_cap = (max(1.0, BO_TIME_SPLIT * float(time_budget))
                   if (time_budget is not None and float(time_budget) > 0) else None)
        # 0.6x head-room: the cap only governs pure training; inference + setup eat the
        # rest of each trial's slot, and 10 full trials must land under _p1_cap total.
        _trial_cap = ((_p1_cap / max(1, int(BO_N_TRIALS))) * 0.6) if _p1_cap else None

        def _mk_cfg(Xs_t, Xps_t, lr_c):
            nz, hh, ww = Xs_t[0].shape
            cfg = build_bg_only_cfg(
                X_target=Xs_t[0], Xps=Xps_t, max_train_time=(_trial_cap if _trial_cap else 1e9),
                bg_h=bg_h, roi_h=4,
                epochs=(100000 if _trial_cap else int(BO_PHASE1_EPOCHS)),
                steps_per_epoch=max(1, nz // int(BG_BATCH)), bg_patch_size=int(min(hh, ww)),
                bg_batch=int(BG_BATCH), lr=float(lr_c), bg_freq_weight=BG_FREQ_WEIGHT, bg_fft_phase_weight=BG_FFT_PHASE_WEIGHT,
                bg_freq_warmup_epochs=1, bg_field_norm="zscore")
            cfg.bg_arch = "spatial"; cfg.bg_split_mode = "three"; cfg.bg_split_bands = True
            cfg.bg_split_sigma = 0.12; cfg.bg_sigma_low = 0.08; cfg.bg_sigma_mid = 0.18
            cfg.bg_low_weight = bg_low_w; cfg.bg_mid_weight = bg_mid_w; cfg.bg_high_weight = bg_high_w
            cfg.bg_cr_rel_err = float(base_rel); cfg.bg_gpu_sampling = True; cfg.seed = SEED; cfg.bg_sample_mode = BG_SAMPLE_MODE
            cfg.bg_full_slice = full_slice
            cfg.bg_cudnn_benchmark = not DETERMINISTIC; cfg.bg_cudnn_deterministic = DETERMINISTIC
            # Size the lr warmup to the trial, not to the (meaningless) 100k-epoch plan.
            # A time-capped trial runs ~20-50 steps, while bg_stage's default warmup is
            # 200 steps: the whole trial then sits on the warmup ramp and effectively
            # tests lr/9, so different learning rates score identically. Measured on the
            # NYX proxy at 0.06 s/trial: the spread across lr in [1e-4, 3e-3] goes from
            # 0.001 dB (default) to 0.011 dB, and at Miranda's 0.48 s/trial the effect is
            # far larger -- the difference between a search that can rank lr and one that
            # cannot.
            cfg.bg_lr_warmup_steps = max(2, int(cfg.steps_per_epoch) // int(BO_LR_WARMUP_FRAC))
            return cfg

        def _run_trial(proxy, k, lr_c):
            Xs_t, Xps_t, dr_t = proxy[k]
            nz  = Xs_t[0].shape[0]
            evs = int(min(int(bo_eval_slices), nz))
            z0  = max(0, nz // 2 - evs // 2); z1 = min(nz, z0 + evs)
            cfg = _mk_cfg(Xs_t, Xps_t, lr_c)
            set_seed(SEED)
            with contextlib.redirect_stdout(io.StringIO()):
                m, _ = train_bg_only(Xs=Xs_t, Xps=Xps_t, device=device, cfg=cfg, evaluator=None)
                xh   = run_bg_inference(unwrap_bg_model(m), Xs_t, Xps_t, cfg, float(base_rel),
                                        z_start=z0, z_stop=z1)
            p  = compute_psnr(Xs_t[0][z0:z1], xh[z0:z1], dr_t)
            pb = compute_psnr(Xs_t[0][z0:z1], Xps_t[0][z0:z1], dr_t)
            del m
            return (float(p) if np.isfinite(p) else -1e9), float(pb)

        def _mk_proxy():
            proxy = {}
            for k in axes:
                tgt_t, aux_t, idx = _bo_ta_cache[k]
                base_t = _dspf(_take_perm(base_recon, k, idx))
                dr_t   = float(tgt_t.max() - tgt_t.min()) or 1.0
                proxy[k] = ([tgt_t] + aux_t, [base_t] + aux_t, dr_t)
            return proxy

        proxy = _mk_proxy()
        if not _bo_warmed[0]:               # cuDNN warmup, once per dataset, off the clock
            _run_trial(proxy, axes[0], float(BO_ENQUEUE_LR))
            _bo_warmed[0] = True

        _t_bo = time.time()
        def objective(trial):
            k    = trial.suggest_categorical("direction", axes)
            lr_c = trial.suggest_float("lr", BO_LR_MIN, BO_LR_MAX, log=True)
            p, pb = _run_trial(proxy, k, lr_c)
            trial.set_user_attr("gain", float(p - pb))
            return float(p)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=int(BO_N_STARTUP)))
        for k in axes:
            study.enqueue_trial({"direction": k, "lr": float(BO_ENQUEUE_LR)})
        with contextlib.redirect_stdout(io.StringIO()):
            study.optimize(objective, n_trials=int(BO_N_TRIALS))   # no timeout: all BO_N_TRIALS run; per-trial cap keeps Phase 1 ~= its share
        _bo_elapsed[0] = time.time() - _t_bo

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
        # axis spread: best-per-axis, i.e. is any direction distinguishable from another
        _axis_best = [per_dir[d][1] for d in axes if d in per_dir]
        axis_spread = (max(_axis_best) - min(_axis_best)) if len(_axis_best) > 1 else 0.0
        # lr spread: only among trials that ran on the axis BO wants to pick
        _on_axis = [t.value for t in study.trials
                    if t.value is not None and int(t.params["direction"]) == best_dir]
        lr_spread = (max(_on_axis) - min(_on_axis)) if len(_on_axis) > 1 else 0.0
        summ = " ".join(f"axis{d}:{per_dir[d][1]:.1f}@{per_dir[d][0]:.0e}" for d in axes if d in per_dir)
        _cap_s = f" (cap {_p1_cap:.1f}s, fast)" if _p1_cap else " (fast)"
        _n_done = sum(1 for t in study.trials if t.state.name == "COMPLETE")
        print(f"  [TPE] band={base_rel:.1e} {_n_done} trials in {_bo_elapsed[0]:.1f}s{_cap_s} "
              f"-> PICK axis{best_dir} lr={best_lr:.1e} proxy={study.best_value:.2f} gain={best_gain:+.2f} axis_spread={axis_spread:.2f} lr_spread={lr_spread:.2f} | {summ}")
        if axis_spread <= BO_MIN_SPREAD_DB:
            fallback_axis = axes[0]
            # The 16-slice slab scoring is noisier than the full-proxy scoring of the
            # default path, and it showed the known failure mode immediately: a proxy
            # picked lr=9.4e-3 for a 113 dB SPERR base, Phase-2 learned nothing, and
            # the error-bound clamp returned the baseline -- +0.00 dB but a WORSE
            # effective CR (model bytes for free). So a non-default lr from the proxy
            # is only adopted if it beat the enqueued default on the SAME axis by the
            # same trust threshold the axis decision uses; within-noise margins fall
            # back to the default lr.
            fb = per_dir.get(fallback_axis)
            enq_score = None
            for t in study.trials:
                if (t.value is not None and int(t.params["direction"]) == fallback_axis
                        and abs(float(t.params["lr"]) - float(BO_ENQUEUE_LR)) < 1e-12):
                    enq_score = t.value if enq_score is None else max(enq_score, t.value)
            if fb is None:
                fallback_lr, _why = lr, "axis never tried"
            elif enq_score is not None and (fb[1] - enq_score) < BO_MIN_GAIN_DB:
                fallback_lr, _why = float(BO_ENQUEUE_LR), (
                    f"best-tried {fb[0]:.1e} only +{fb[1]-enq_score:.2f}dB over default on proxy")
            else:
                fallback_lr, _why = fb[0], "best tried for that axis"
            print(f"  [TPE] => axis_spread {axis_spread:.2f} <= {BO_MIN_SPREAD_DB}dB "
                  f"(directions indistinguishable) -> axis{fallback_axis}, lr={fallback_lr:.1e} ({_why})")
            return fallback_axis, fallback_lr
        if lr_spread <= BO_MIN_LR_SPREAD_DB:
            print(f"  [TPE] => axis{best_dir} accepted (axis_spread {axis_spread:.2f}dB), but "
                  f"lr_spread {lr_spread:.2f} <= {BO_MIN_LR_SPREAD_DB}dB -> lr={BO_ENQUEUE_LR:.1e} (default)")
            return best_dir, float(BO_ENQUEUE_LR)
        _lo = np.log10(BO_LR_MIN); _span = np.log10(BO_LR_MAX) - _lo
        if (np.log10(best_lr) - _lo) <= float(BO_LR_LOW_REJECT_FRAC) * _span:
            print(f"  [TPE] => axis{best_dir} accepted, but lr={best_lr:.1e} sits in the bottom "
                  f"{BO_LR_LOW_REJECT_FRAC:.0%} of the search range (proxy is still in its "
                  f"damage regime, not ranking lr) -> lr={BO_ENQUEUE_LR:.1e} (default)")
            return best_dir, float(BO_ENQUEUE_LR)
        return best_dir, best_lr

    def _phase1_best(base_recon, base_rel):
        if not BO_ENABLE:
            _bo_elapsed[0] = 0.0
            return (full_slice_axis if full_slice else 0), lr
        if bo_eval_slices:
            return _phase1_best_fast(base_recon, base_rel)
        _t_bo = time.time()
        full_shape = np.asarray(target_gt).shape
        axes = (list(bo_axes) if bo_axes is not None else
               ([0, 1, 2] if BO_SEARCH_DIRECTION else [full_slice_axis if full_slice else 0]))
        _dsp = ((lambda x: np.ascontiguousarray(x[:, ::PROXY_DS, ::PROXY_DS]))
                if PROXY_DS > 1 else (lambda x: x))

        proxy = {}
        for k in axes:
            Dk     = int(full_shape[k])
            stride = int((bo_proxy_stride or {}).get(k, PROXY_DS))
            idx    = np.arange(0, Dk, stride)
            tgt_t  = _dsp(_take_perm(target_gt,  k, idx))
            base_t = _dsp(_take_perm(base_recon, k, idx))
            aux_t  = [_dsp(_take_perm(a, k, idx)) for a in aux_list]
            dr_t   = float(tgt_t.max() - tgt_t.min()) or 1.0
            proxy[k] = ([tgt_t] + aux_t, [base_t] + aux_t, dr_t, compute_psnr(tgt_t, base_t, dr_t))

        # Per-trial wall-clock cap so a slow candidate axis can't eat the whole BO
        # budget before the others are even tried. optuna's own `timeout=` (below)
        # only stops NEW trials from starting -- it never interrupts one already
        # running, so on a dataset where a single proxy trial costs minutes (e.g.
        # QMCPACK's axis0: steps_per_epoch=nz is huge for a thin-slice axis), the
        # FIRST enqueued axis silently consumes the entire cap and every other axis
        # goes untested -- "PICK axis0" then means "axis0 was the only one that got
        # to run," not "axis0 won a comparison." Splitting the cap across the
        # BO_N_STARTUP enqueued candidates (matching len(axes) in the common case)
        # gives each one a fair shot within roughly the intended total time.
        _p1_cap = (max(1.0, BO_TIME_SPLIT * float(time_budget))
                   if (time_budget is not None and float(time_budget) > 0) else None)
        _trial_cap = (_p1_cap / max(1, int(BO_N_TRIALS))) if _p1_cap else None

        def objective(trial):
            k    = trial.suggest_categorical("direction", axes)
            lr_c = trial.suggest_float("lr", BO_LR_MIN, BO_LR_MAX, log=True)
            Xs_t, Xps_t, dr_t, base_p = proxy[k]
            nz, hh, ww = Xs_t[0].shape
            patch_d = int(min(hh, ww))
            cfg = build_bg_only_cfg(
                X_target=Xs_t[0], Xps=Xps_t, max_train_time=(_trial_cap if _trial_cap else 1e9),
                bg_h=bg_h, roi_h=4,
                epochs=(100000 if _trial_cap else int(BO_PHASE1_EPOCHS)),
                steps_per_epoch=max(1, nz // int(BG_BATCH)), bg_patch_size=patch_d,
                bg_batch=int(BG_BATCH), lr=float(lr_c), bg_freq_weight=BG_FREQ_WEIGHT, bg_fft_phase_weight=BG_FFT_PHASE_WEIGHT,
                bg_freq_warmup_epochs=1, bg_field_norm="zscore")
            cfg.bg_arch = "spatial"; cfg.bg_split_mode = "three"; cfg.bg_split_bands = True
            cfg.bg_split_sigma = 0.12; cfg.bg_sigma_low = 0.08; cfg.bg_sigma_mid = 0.18
            cfg.bg_low_weight = bg_low_w; cfg.bg_mid_weight = bg_mid_w; cfg.bg_high_weight = bg_high_w
            cfg.bg_cr_rel_err = float(base_rel); cfg.bg_gpu_sampling = True; cfg.seed = SEED; cfg.bg_sample_mode = BG_SAMPLE_MODE
            cfg.bg_full_slice = full_slice
            cfg.bg_cudnn_benchmark = not DETERMINISTIC; cfg.bg_cudnn_deterministic = DETERMINISTIC
            cfg.bg_lr_warmup_steps = max(2, int(cfg.steps_per_epoch) // int(BO_LR_WARMUP_FRAC))   # see fast path
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
            study.optimize(objective, n_trials=int(BO_N_TRIALS))   # no timeout: all BO_N_TRIALS run; per-trial cap keeps Phase 1 ~= its share
        _bo_elapsed[0] = time.time() - _t_bo

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
        # axis spread: best-per-axis, i.e. is any direction distinguishable from another
        _axis_best = [per_dir[d][1] for d in axes if d in per_dir]
        axis_spread = (max(_axis_best) - min(_axis_best)) if len(_axis_best) > 1 else 0.0
        # lr spread: only among trials that ran on the axis BO wants to pick
        _on_axis = [t.value for t in study.trials
                    if t.value is not None and int(t.params["direction"]) == best_dir]
        lr_spread = (max(_on_axis) - min(_on_axis)) if len(_on_axis) > 1 else 0.0
        summ = " ".join(f"axis{d}:{per_dir[d][1]:.1f}@{per_dir[d][0]:.0e}" for d in axes if d in per_dir)
        _cap_s = f" (cap {_p1_cap:.1f}s)" if _p1_cap else ""
        _n_done = sum(1 for t in study.trials if t.state.name == "COMPLETE")
        print(f"  [TPE] band={base_rel:.1e} {_n_done} trials in {_bo_elapsed[0]:.1f}s{_cap_s} "
              f"-> PICK axis{best_dir} lr={best_lr:.1e} proxy={study.best_value:.2f} gain={best_gain:+.2f} axis_spread={axis_spread:.2f} lr_spread={lr_spread:.2f} | {summ}")
        if axis_spread <= BO_MIN_SPREAD_DB:
            # axes[0] (NOT a hardcoded 0): when bo_axes restricts the candidate set
            # (e.g. bo_axes=[2]), axis 0 may not even be a legal choice for this
            # dataset -- falling back to it unconditionally silently broke that
            # constraint. axes[0] degrades to the old hardcoded behavior exactly
            # when bo_axes is None (axes is then [0,1,2] or [full_slice_axis or 0]).
            fallback_axis = axes[0]
            # The gain gate says "no axis beat the baseline enough to trust switching
            # away from the default" -- that does NOT mean the lr search inside
            # fallback_axis's own trials was noise too. per_dir[fallback_axis] already
            # holds whichever lr scored highest AMONG the trials actually run on that
            # axis; with bo_axes restricting the search to one axis (e.g. QMCPACK's
            # bo_axes=[0]), enough trials fit in budget to genuinely compare several lr
            # values there (real example: 3.9e-3 and 9.4e-3 beat the enqueued 1e-3 by
            # a real margin on QMCPACK's proxy) -- discarding that back to the hardcoded
            # default on every gain-gate trip was throwing away a real finding. Only
            # falls through to the hardcoded default if fallback_axis was never tried
            # at all (shouldn't happen -- it's always enqueued -- but keeps this safe).
            fallback_lr = per_dir[fallback_axis][0] if fallback_axis in per_dir else lr
            print(f"  [TPE] => axis_spread {axis_spread:.2f} <= {BO_MIN_SPREAD_DB}dB (configs indistinguishable) "
                  f"-> axis{fallback_axis}, lr={fallback_lr:.1e} (best tried for that axis)")
            return fallback_axis, fallback_lr
        return best_dir, best_lr

    def _train_residual(base_recon, base_rel):
        """Same as SPERR.py's _train_residual, but ALSO returns the enhanced
        reconstruction un-permuted back to (Z,Y,X) so FFT error can be computed
        against the original target_gt."""
        set_seed(SEED)
        best_k, use_lr = _phase1_best(base_recon, base_rel)
        # aux_list already holds the siblings as the decoder has them (_use_aux_for_cr)
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
            max_train_time=((1.0 - BO_TIME_SPLIT) * float(time_budget) if use_time_budget else 1e9),
            bg_h=bg_h, roi_h=4,
            epochs=(100000 if use_time_budget else epochs), steps_per_epoch=max(1, dep_d // int(BG_BATCH)),
            bg_patch_size=patch_d, bg_batch=int(BG_BATCH), lr=use_lr,
            bg_freq_weight=BG_FREQ_WEIGHT, bg_fft_phase_weight=BG_FFT_PHASE_WEIGHT, bg_freq_warmup_epochs=1,
            bg_field_norm="zscore")
        cfg.bg_arch = "spatial"; cfg.bg_split_mode = "three"; cfg.bg_split_bands = True
        cfg.bg_split_sigma = 0.12; cfg.bg_sigma_low = 0.08; cfg.bg_sigma_mid = 0.18
        cfg.bg_low_weight = bg_low_w; cfg.bg_mid_weight = bg_mid_w; cfg.bg_high_weight = bg_high_w
        cfg.bg_cr_rel_err = float(base_rel)
        cfg.bg_sched_time_calibrate = use_time_budget and bool(BG_SCHED_TIME_CALIBRATE)
        cfg.bg_sched_step_calibrate = use_time_budget and bool(BG_SCHED_STEP_CALIB)
        cfg.bg_gpu_sampling = True
        cfg.bg_sample_mode = BG_SAMPLE_MODE
        cfg.seed = SEED
        cfg.bg_full_slice = full_slice
        cfg.bg_cudnn_benchmark = not DETERMINISTIC; cfg.bg_cudnn_deterministic = DETERMINISTIC
        def ev(model, c=cfg, Xs=Xs, Xps=Xps, r=base_rel):
            return compute_psnr(Xs[0], run_bg_inference(model, Xs, Xps, c, float(r)), drange), 0.0
        model, _hist = train_bg_only(Xs=Xs, Xps=Xps, device=device, cfg=cfg, evaluator=ev)
        bg_train_time = float(_hist["time"][-1]) if _hist.get("time") else float("nan")
        if use_time_budget:
            print(f"      [time] phase1(BO) {_bo_elapsed[0]:.1f}s | phase2(train) {bg_train_time:.1f}s "
                  f"(caps {BO_TIME_SPLIT*float(time_budget):.1f}s / {(1.0-BO_TIME_SPLIT)*float(time_budget):.1f}s)")
        x_hat = run_bg_inference(model, Xs, Xps, cfg, float(base_rel))
        p = compute_psnr(Xs[0], x_hat, drange)
        x_hat_orig = _unperm_view(x_hat, best_k)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # NeurLZ is given AdaMit's ALL-PHASE time, not just Phase 2. _bo_elapsed[0] is
        # the Phase-1 (BO) wall time actually spent for this operating point, so the
        # returned budget is Phase 1 + Phase 2. Returning only bg_train_time would hand
        # NeurLZ ~10% less wall clock than AdaMit consumed end-to-end, making the
        # iso-time claim in the paper false in our own favour.
        total_pipeline_time = bg_train_time + float(_bo_elapsed[0])
        return p, total_pipeline_time, best_k, x_hat_orig

    # ── sibling fields at the target's compression level (AUX_MODE='cr_matched') ──
    # aux_list is swapped IN PLACE per operating point, so every consumer (Phase-1
    # proxies, Phase-2 training, inference, NeurLZ) sees the same decompressed
    # siblings; the originals are kept only as compression input.
    aux_clean = list(aux_list)
    _aux_state = {"key": None}

    def _use_aux_for_cr(cr_base, compressor):
        if not aux_clean or AUX_MODE != "cr_matched":
            return
        level = min(AUX_CR_LEVELS, key=lambda L: abs(np.log(L) - np.log(max(float(cr_base), 1e-9))))
        key = (compressor, int(level))
        if _aux_state["key"] == key:
            return
        new = []
        for _ai, _a in enumerate(aux_clean):
            _f = aux_files[_ai] if aux_files else None
            _dr = float(np.max(_a) - np.min(_a)) or 1.0
            _enh = _enhanced_sibling(_f, compressor, float(cr_base), shape) if (AUX_ENHANCED_DIR and _f) else None
            if _enh is not None:
                _ad, _src = _enh
                print(f"  [aux] target CR {float(cr_base):6.1f} ({compressor}) | sibling {_ai}: ENHANCED by an earlier stage "
                      f"({os.path.basename(_src)}), PSNR {compute_psnr(np.asarray(_a, np.float32), _ad, _dr):.1f} dB")
            else:
                _ad, _cr, _knob = _aux_at_cr_level(_a, _f, shape, level, compressor, dtype=src_dtype)
                print(f"  [aux] target CR {float(cr_base):6.1f} -> level {level} ({compressor}) | sibling {_ai}: "
                      f"CR {_cr:6.1f}  PSNR {compute_psnr(np.asarray(_a, np.float32), _ad, _dr):.1f} dB  knob {_knob:.4g}")
            new.append(_ad)
        aux_list[:] = new
        _aux_state["key"] = key
        _bo_ta_cache.clear()          # Phase-1 proxies must be rebuilt from the new siblings

    # ── SZ3  and  SZ3 + model ──
    for rel in rel_errs:
        b, _   = sz_engine.compress(target_gt, 1, 0, float(rel), 0)
        sz_len = len(b)
        xq     = sz_engine.decompress(b, shape, src_dtype)
        p_sz3  = compute_psnr(target_gt, xq, drange); cr_sz3 = orig_bytes / sz_len
        _use_aux_for_cr(cr_sz3, "sz3")
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
        _save_recons(name, "sz3", cr_sz3, target_gt,
                     dict(original_file=str(target_file), rel=float(rel), cr_base=cr_sz3, psnr_base=p_sz3, cr_ours=cr_pipe, psnr_ours=p_pipe,
                          fft_mag_base=m_sz3, fft_mag_ours=m_pipe, fft_phase_base=ph_sz3, fft_phase_ours=ph_pipe),
                     base=xq, ours=xhat_pipe)
        del xhat_pipe   # 4 GB, FFT already computed -> free before the memory-heavy run_neurlz

        if ADD_NEURLZ:
            p_nlz, nlz_params, nlz_time, nlz_eps, _hist_nlz, enh_nlz = run_neurlz(
                target_gt, xq, aux_list, shape, drange, rel,
                int(epochs * NEURLZ_EPOCH_MULT), neurlz_features, time_budget=bg_time,
                slice_axis=bg_axis, return_history=True)
            enh_nlz_orig = _unperm_view(enh_nlz, bg_axis)
            m_nlz, ph_nlz = _global_fft_err(target_gt, enh_nlz_orig, N_FFT_SLICES)
            cr_nlz = orig_bytes / (sz_len + nlz_params * BYTES_PER_PARAM)
            _save_recons(name, "sz3", cr_sz3, target_gt,
                         dict(cr_neurlz=cr_nlz, psnr_neurlz=p_nlz, fft_mag_neurlz=m_nlz, fft_phase_neurlz=ph_nlz), neurlz=enh_nlz_orig)
            neurlz["CR"].append(cr_nlz); neurlz["PSNR"].append(p_nlz)
            neurlz["fft_mag"].append(m_nlz); neurlz["fft_phase"].append(ph_nlz)
            print(f"           SZ3+NeurLZ {cr_nlz:6.1f}x/{p_nlz:5.1f}dB/mag={m_nlz:.3g}/pha={ph_nlz:.3g} "
                  f"[{nlz_params:,}p] | BG {epochs}ep {bg_time:.1f}s ≈ NeurLZ {nlz_time:.1f}s/{nlz_eps}ep")

    # ── SPERR  and  SPERR + model ──
    if sz3["PSNR"]:
        hi = max(sz3["PSNR"]) - sperr_psnr_offset
        lo = min(sz3["PSNR"]) - sperr_psnr_offset - sperr_extra_span
        if SPERR_MAX_CR:
            lo = max(lo, _sperr_psnr_for_cr(target_file, target_gt, shape, drange, float(SPERR_MAX_CR), dtype=src_dtype))
        n_pts = len(rel_errs) + sperr_n_extra
        print(f"  SPERR sweep: {n_pts} targets, PSNR {hi:.1f}..{lo:.1f} dB"
              + (f" (CR capped <= {SPERR_MAX_CR})" if SPERR_MAX_CR else ""))
        _targets = (list(np.linspace(hi, lo, n_pts)) if not ONLY_BANDS else
                    [_sperr_psnr_for_cr(target_file, target_gt, shape, drange, float(c), dtype=src_dtype) for c in sz3["CR"]])   # viz: SPERR aligned to the SZ3 band's CR
        for tp in _targets:
            cr_sp, p_sp, recon_sp, sp_bytes = run_sperr(target_file, target_gt, shape, drange, float(tp), dtype=src_dtype)
            if recon_sp is None:
                continue
            if SPERR_MAX_CR and cr_sp is not None and cr_sp > SPERR_MAX_CR * 1.03:
                continue
            _use_aux_for_cr(cr_sp, "sperr")
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
            _save_recons(name, "sperr", cr_sp, target_gt,
                         dict(original_file=str(target_file), sperr_psnr_target=float(tp), rel_equiv=rel_sp, cr_base=cr_sp, psnr_base=p_sp, cr_ours=cr_spp, psnr_ours=p_spp,
                              fft_mag_base=m_sp, fft_mag_ours=m_spp, fft_phase_base=ph_sp, fft_phase_ours=ph_spp),
                         base=recon_sp, ours=xhat_spp)
            del xhat_spp   # 4 GB, FFT already computed -> free before the memory-heavy run_neurlz

            if ADD_NEURLZ:
                p_spn, spn_params, spn_time, spn_eps, _hist_spn, enh_spn = run_neurlz(
                    target_gt, recon_sp, aux_list, shape, drange, rel_sp,
                    int(epochs * NEURLZ_EPOCH_MULT), neurlz_features,
                    time_budget=bg_time_sp, slice_axis=bg_axis_sp, return_history=True)
                enh_spn_orig = _unperm_view(enh_spn, bg_axis_sp)
                m_spn, ph_spn = _global_fft_err(target_gt, enh_spn_orig, N_FFT_SLICES)
                cr_spn = orig_bytes / (sp_bytes + spn_params * BYTES_PER_PARAM)
                _save_recons(name, "sperr", cr_sp, target_gt,
                             dict(cr_neurlz=cr_spn, psnr_neurlz=p_spn, fft_mag_neurlz=m_spn, fft_phase_neurlz=ph_spn), neurlz=enh_spn_orig)
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
FFT_CACHE_DIR = P("ADAMIT_CACHE_DIR")
os.makedirs(FFT_CACHE_DIR, exist_ok=True)
FORCE_RETRAIN = bool(int(os.environ.get("SPERR_FORCE_RETRAIN", "0")))

_CACHE_CFG_KEYS = [
    "SEED", "BO_ENABLE", "BO_N_TRIALS", "BO_N_STARTUP", "BO_ENQUEUE_LR", "BO_LR_MIN",
    "BO_LR_MAX", "BO_PHASE1_EPOCHS", "BO_MIN_GAIN_DB", "BO_MIN_SPREAD_DB", "BO_MIN_LR_SPREAD_DB", "BO_LR_LOW_REJECT_FRAC", "BO_SEARCH_DIRECTION", "BO_TIME_SPLIT", "BG_SCHED_TIME_CALIBRATE",
    "EPOCHS_OVERRIDE", "BG_USE_AUX", "PROXY_DS", "ADD_NEURLZ", "N_FFT_SLICES",
    "NEURLZ_FEATURES", "NEURLZ_LR", "NEURLZ_BATCH", "NEURLZ_MATCH_OURS_AXIS",
    "NEURLZ_BEST_GUARD", "NEURLZ_POSTPROCESS", "NEURLZ_SINGLE_FIELD", "NEURLZ_EPOCH_MULT",
    "BO_LR_WARMUP_FRAC", "BYTES_PER_PARAM", "DETERMINISTIC", "BG_BATCH", "SPERR_MAX_CR",
    "BG_FREQ_WEIGHT", "BG_FFT_PHASE_WEIGHT", "BG_SAMPLE_MODE", "BG_SCHED_STEP_CALIB",
]


def _fft_cache_path(name, target_file, shape, rel_errs, param_budget, epochs, n_aux, kw):
    g = globals()
    cfg = {k: g.get(k, None) for k in _CACHE_CFG_KEYS}
    if int(n_aux) > 0:                                    # sibling protocol (aux datasets only)
        cfg["AUX_MODE"] = g.get("AUX_MODE", None)
        if g.get("AUX_MODE") == "cr_matched":
            cfg["AUX_CR_LEVELS"] = tuple(int(x) for x in g.get("AUX_CR_LEVELS", ()))
        if g.get("AUX_FIELDS"):
            cfg["AUX_FIELDS"] = tuple(g["AUX_FIELDS"])
        if g.get("AUX_ENHANCED_DIR"):
            cfg["AUX_ENHANCED_DIR"] = str(g["AUX_ENHANCED_DIR"])
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
# NeurLZ now slices along whichever axis "Ours" actually trained on for THIS point
# (bg_axis/bg_axis_sp, returned by _train_residual) instead of a fixed axis -- see
# SPERR.py's comment for the full rationale. This marker exists purely so the cache
# signature changes and old (wrong-axis) NeurLZ results get recomputed.
NEURLZ_MATCH_OURS_AXIS = True
NEURLZ_BATCH      = 10
NEURLZ_MAX_PIXELS_PER_BATCH = 16 * 1024 * 1024
NEURLZ_VERBOSE    = True
NEURLZ_EVAL_EVERY = int(os.environ.get('SPERR_NLZ_EVAL_EVERY', '1'))   # neurlz_long: evaluate every N epochs
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
              time_budget=None, slice_axis=0, return_history=False, aux_infer=None):
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
    # single field: skip np.stack (it copies a full extra volume; n_fields>1 needs it)
    if n_fields == 1:
        lq_n = _mm(lq)[0][:, None]                 # (D,1,H,W): add-axis view, no copy
        _field_stats = None
    else:
        _normed = [_mm(f) for f in fields]
        lq_n = np.stack([t[0] for t in _normed], axis=1)
        _field_stats = [t[1] for t in _normed]     # per-channel (lo, hi) for decode-side swap
        del _normed
    _diff = tgt - lq
    err_n, (e_lo, e_hi) = _mm(_diff)
    del _diff                                      # free the 4 GB residual immediately
    ph, pw = (-H) % 16, (-W) % 16
    pad = ((0, 0), (0, 0), (0, ph), (0, pw))
    # np.pad(reflect) copies the ENTIRE volume even when the pad width is 0 (e.g.
    # Miranda 1024^3 is already a multiple of 16). On a 4 GB field that's two wasted
    # full-size copies -> OOM (peak RSS measured 52 GB for one Miranda run_neurlz).
    # When there's nothing to pad, from_numpy just views lq_n / err_n instead.
    if ph == 0 and pw == 0:
        Xlq  = torch.from_numpy(lq_n)
        Yerr = torch.from_numpy(err_n[:, None])
    else:
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
        _t_inf = time.perf_counter()
        out = lq.copy()
        with torch.no_grad():
            for st in range(0, D, eff_batch):
                bi = list(range(st, min(st + eff_batch, D)))
                pred = model(Xlq[bi].to(device)).cpu().numpy()[:, 0, :H, :W]
                out[bi] = lq[bi] + (pred * (e_hi - e_lo + 1e-8) + e_lo)
        model.train()
        if NEURLZ_POSTPROCESS:
            out = _error_bounded_post_process(x_enhanced=out, x_prime=lq, absolute_error_bound=0.0,
                                              relative_error_bound=float(rel), verbose=False, a=1.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        globals()["NEURLZ_LAST_INFER_S"] = time.perf_counter() - _t_inf   # neurlz_long: inference cost
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
        _stop_at = globals().get("NEURLZ_STOP_PSNR")          # neurlz_long: stop once AdaMit's PSNR is reached
        if eval_now and (NEURLZ_BEST_GUARD or NEURLZ_VERBOSE or return_history):
            pe = compute_psnr(tgt, _enhanced(), drange)
            if NEURLZ_BEST_GUARD and pe > best_psnr:
                best_psnr  = float(pe)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if return_history:
                hist_t.append(float(train_time)); hist_p.append(float(pe))
            if _stop_at is not None and pe >= float(_stop_at):
                print(f"           [neurlz] reached target {float(_stop_at):.2f} dB at ep {ep} / {train_time:.1f}s -> stop", flush=True)
                done = True
            if NEURLZ_VERBOSE:
                _bs = f" | best {best_psnr:.2f}" if NEURLZ_BEST_GUARD else ""
                print(f"           [neurlz] ep {ep:3d} | {train_time:5.1f}/{budget_str} | MSE {tot/max(nb,1):.6f} | PSNR {pe:.2f} dB{_bs}")
        elif NEURLZ_VERBOSE:
            print(f"           [neurlz] ep {ep:3d} | {train_time:5.1f}/{budget_str} | MSE {tot/max(nb,1):.6f}")
        if done:
            break

    # decode-side protocol: the REPORTED inference feeds the decompressed siblings,
    # normalized with the training-time statistics (which the decoder stores).
    if aux_infer is not None and n_fields > 1:
        for _ci, _adec in enumerate(aux_infer, start=1):
            _lo, _hi = _field_stats[_ci]
            _a = np.asarray(_adec, np.float32)
            if int(slice_axis) != 0:
                _a = np.ascontiguousarray(np.transpose(_a, _p))
            Xlq[:, _ci, :H, :W] = torch.from_numpy((_a - _lo) / (_hi - _lo + 1e-8))
    if NEURLZ_BEST_GUARD:
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            enh = _enhanced()
            out_psnr = compute_psnr(tgt, enh, drange)
        else:
            enh = lq.copy()                     # guard: never report worse than the base
            out_psnr = float(base_psnr)
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
# process), TASK==<its own key> (its dedicated SLURM process)
# (re-invoked here so the plot process gets a cache HIT and can rebuild `results`
# without duplicating the config elsewhere).
# ─────────────────────────────────────────────────────────────────────────────
results = {}   # label -> r dict, in the order we want plotted

# ── NYX 512^3 (3 targets) — same configs as SPERR.py, for apples-to-apples CR points ──
NYX_DIR   = P("ADAMIT_NYX_DIR")
NYX_SHAPE = (512, 512, 512)
NYX_ALL   = ["baryon_density", "dark_matter_density", "temperature",
             "velocity_x", "velocity_y", "velocity_z"]
# Real SZ3+SPERR binary-search realignment to CR 100-500 (was ~50-400 before, not
# aligned across datasets) -- see align_all_to_100_500.py. Each field gets its OWN
# SPERR_OFF/EXTRA_SPAN now (previously all three shared one number, which only
# happened to work when they were on the old CR range together).
NYX_REL = {
    "baryon_density":      [1.3827e-06, 2.3587e-06, 4.0235e-06, 6.8636e-06, 1.1708e-05],
    "temperature":         [1.4458e-04, 2.2308e-04, 3.4420e-04, 5.3109e-04, 8.1945e-04],
    "dark_matter_density": [1.3496e-04, 2.0941e-04, 3.2493e-04, 5.0419e-04, 7.8233e-04],
}
NYX_SPERR_OFF = {
    "baryon_density": 22.86, "temperature": 23.07, "dark_matter_density": 18.78,
}
NYX_SPERR_EXTRA_SPAN = {
    "baryon_density": 7.15, "temperature": 4.45, "dark_matter_density": 6.80,
}
NYX_PARAMS, NYX_EPOCHS = 30000, 10
NYX_SPERR_NEXTRA = 0
NYX_TIME_BUDGET = 10.0   # seconds of Phase-2 "Ours" training per rel_err (NeurLZ matches this)
NYX_TASK_KEY = {"baryon_density": "nyx_b", "temperature": "nyx_t", "dark_matter_density": "nyx_d"}

if TASK == "aux_prep":
    # One-time archive of every NYX sibling at every (compressor, CR level): pure CPU
    # (SZ3 / sperr3d bisection), so it can run alongside a GPU job; the NYX tasks then
    # find the streams on disk. Level 600 is left to be built lazily if a target ever
    # lands closer to it than to 500.
    _lv = [int(x) for x in os.environ.get("SPERR_AUX_PREP_LEVELS", "100,200,300,400,500").split(",")]
    for _an in NYX_ALL:
        _arr = np.memmap(NYX_DIR + _an + ".f32", dtype=np.float32, mode="r", shape=NYX_SHAPE)
        _dr = float(np.max(_arr) - np.min(_arr))
        for _comp in ("sz3", "sperr"):
            for _L in _lv:
                _t0 = time.time()
                _d, _cr, _k = _aux_at_cr_level(_arr, NYX_DIR + _an + ".f32", NYX_SHAPE, _L, _comp)
                print(f"[aux_prep] {_an:20s} {_comp:5s} level {_L}: CR {_cr:6.1f}  knob {_k:.4g}  "
                      f"PSNR {compute_psnr(np.asarray(_arr, np.float32), _d, _dr):.1f} dB  ({time.time()-_t0:.0f}s)", flush=True)
                del _d
        del _arr
    sys.exit(0)

for tname, tkey in NYX_TASK_KEY.items():
    if TASK not in ("all", tkey):
        continue
    set_seed(SEED)
    tgt = np.fromfile(NYX_DIR + tname + ".f32", dtype=np.float32).reshape(NYX_SHAPE)
    _sibs = [a for a in NYX_ALL if a != tname and (not AUX_FIELDS or a in AUX_FIELDS)]   # SPERR_AUX_FIELDS: ablation subset
    aux = [np.memmap(NYX_DIR + a + ".f32", dtype=np.float32, mode="r", shape=NYX_SHAPE) for a in _sibs]
    _rels = ([float(x) for x in os.environ['SPERR_NYX_RELS'].split(',')] if os.environ.get('SPERR_NYX_RELS')
             else ([NYX_REL[tname][i] for i in ONLY_BANDS] if ONLY_BANDS else NYX_REL[tname]))   # SPERR_NYX_RELS: custom bands (viz); SPERR_ONLY_BANDS: subset
    r = cached_bench_field_fft(f"NYX/{tname}", tgt, NYX_DIR + tname + ".f32", aux, NYX_SHAPE,
                               _rels,
                               NYX_PARAMS, NYX_EPOCHS, sperr_psnr_offset=NYX_SPERR_OFF[tname],
                               sperr_extra_span=NYX_SPERR_EXTRA_SPAN[tname], sperr_n_extra=NYX_SPERR_NEXTRA,
                               time_budget=NYX_TIME_BUDGET,
                               aux_files=[NYX_DIR + a + ".f32" for a in _sibs],
                               # fast Phase-1: depth stride 8 + spatial /4 -> 64x128x128
                               # proxies, trials scored on the middle 16 slices. 0.07 s
                               # per trial measured, so all 10 trials fit the 1.0 s cap
                               # (default path: proxy rebuild alone cost 6.5 s/point and
                               # only 1-2 trials ever ran). New kwargs change the cache
                               # hash, so NYX retrains under the new Phase-1.
                               bo_proxy_stride={0: 8, 1: 8, 2: 8}, bo_proxy_spatial=4,
                               bo_eval_slices=16)
    results[f"NYX — {tname}"] = r
    del tgt, aux
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ── Miranda 1024^3 ──
MIR_FILE  = P("ADAMIT_MIRANDA_FILE")
MIR_SHAPE = (1024, 1024, 1024)
# Real SZ3+SPERR binary-search realignment to CR 100-500 (was 108-370, and SPERR's
# side was never actually calibrated here -- this call used to omit
# sperr_psnr_offset/extra_span entirely, silently defaulting to bench_field_fft's
# uncalibrated 8.0/0.0). See align_all_to_100_500.py.
MIR_REL   = [4.5523e-03, 6.9948e-03, 1.0748e-02, 1.6514e-02, 2.5374e-02]
MIR_PARAMS, MIR_EPOCHS = 240000, 5
MIR_TIME_BUDGET = 80.0   # seconds of Phase-2 "Ours" training per rel_err (NeurLZ matches this)
MIR_SPERR_OFF, MIR_SPERR_EXTRA_SPAN = 3.88, 6.89

if TASK in ("all", "miranda"):
    set_seed(SEED)
    mir = np.fromfile(MIR_FILE, dtype=np.float32).reshape(MIR_SHAPE)
    results["Miranda 1024³"] = cached_bench_field_fft("Miranda", mir, MIR_FILE, [], MIR_SHAPE,
                                                      ([MIR_REL[i] for i in ONLY_BANDS] if ONLY_BANDS else MIR_REL), MIR_PARAMS, MIR_EPOCHS,
                                                      sperr_psnr_offset=MIR_SPERR_OFF,
                                                      sperr_extra_span=MIR_SPERR_EXTRA_SPAN,
                                                      time_budget=MIR_TIME_BUDGET,
                                                      # fast Phase-1: (128,256,256) proxy,
                                                      # 0.57 s/trial measured -> 10 trials
                                                      # in 5.7 s under the 8.0 s cap. The
                                                      # default path rebuilt a 512^3 proxy
                                                      # every point and ran ~2 trials in
                                                      # 28.7 s (3.6x over cap).
                                                      bo_proxy_stride={0: 8, 1: 8, 2: 8},
                                                      bo_proxy_spatial=4, bo_eval_slices=16)
    del mir
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ── Magnetic Reconnection 512^3 ──
MAG_FILE  = P("ADAMIT_MAGNETIC_FILE")
MAG_SHAPE = (512, 512, 512)
# Real SZ3+SPERR realignment to CR 100-500 (was 173-547, and like Miranda above, SPERR's
# side had never actually been calibrated in THIS file -- omitted sperr_psnr_offset/
# extra_span, silently defaulting to bench_field_fft's uncalibrated 8.0/0.0).
MAG_REL   = [4.0613e-03, 6.2596e-03, 9.6476e-03, 1.4869e-02, 2.2918e-02]
MAG_PARAMS, MAG_EPOCHS = 30000, 10
MAG_TIME_BUDGET = 10.0   # seconds of Phase-2 "Ours" training per rel_err (NeurLZ matches this)
MAG_SPERR_OFF, MAG_SPERR_EXTRA_SPAN = 5.66, 4.76

if TASK in ("all", "mag"):
    set_seed(SEED)
    mag = np.fromfile(MAG_FILE, dtype=np.float32).reshape(MAG_SHAPE)
    results["Magnetic Reconnection"] = cached_bench_field_fft("Magnetic", mag, MAG_FILE, [], MAG_SHAPE,
                                                              ([MAG_REL[i] for i in ONLY_BANDS] if ONLY_BANDS else MAG_REL), MAG_PARAMS, MAG_EPOCHS,
                                                              sperr_psnr_offset=MAG_SPERR_OFF,
                                                              sperr_extra_span=MAG_SPERR_EXTRA_SPAN,
                                                              time_budget=MAG_TIME_BUDGET,
                                                              # fast Phase-1: (64,128,128)
                                                              # proxy, 0.071 s/trial -> 10
                                                              # trials in 0.71 s under the
                                                              # 1.0 s cap (default path:
                                                              # 0.35 s/trial, 3.5 s).
                                                              bo_proxy_stride={0: 8, 1: 8, 2: 8},
                                                              bo_proxy_spatial=4,
                                                              bo_eval_slices=16)
    del mag
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
# ── QMCPACK einspline B-spline coefficient table, folded into a (33120,69,69) stack ──
# See SPERR.py's QMCPACK block for the full rationale (why it's a fold of 288 orbitals
# along axis0, why axis0 vs axis1/2 are not equivalent, why bo_axes/bo_proxy_stride are
# needed for BO to be tractable). Already aligned to CR 100-500 across the 5-point
# QMC_REL sweep -- no recalibration needed here.
QMC_FILE  = P("ADAMIT_QMC_FILE")
QMC_SHAPE = (33120, 69, 69)
QMC_REL   = [1.9698e-04, 3.9741e-04, 8.0177e-04, 1.6176e-03, 3.2634e-03]
QMC_PARAMS, QMC_EPOCHS = int(os.environ.get('SPERR_QMC_PARAMS', '35000')), 10   # 35k (2026-08-19 decision) -> bg_h=23, 34,504 params; 30k was bg_h=21 / 28,858
QMC_TIME_BUDGET = 60.0   # seconds of Phase-2 "Ours" training per rel_err (NeurLZ matches this)
QMC_SPERR_OFF, QMC_SPERR_EXTRA_SPAN = 10.81, 7.22
QMC_BO_AXES = [0]
QMC_BO_PROXY_STRIDE = {0: 200}

if TASK in ("all", "qmcpack"):
    set_seed(SEED)
    _qm = np.fromfile(QMC_FILE, dtype=np.float32).reshape(QMC_SHAPE)
    results["QMCPACK — einspline"] = cached_bench_field_fft(
        "QMCPack", _qm, QMC_FILE, [], QMC_SHAPE, ([QMC_REL[i] for i in ONLY_BANDS] if ONLY_BANDS else QMC_REL), QMC_PARAMS, QMC_EPOCHS,
        sperr_psnr_offset=QMC_SPERR_OFF, sperr_extra_span=QMC_SPERR_EXTRA_SPAN,
        time_budget=QMC_TIME_BUDGET, bo_axes=QMC_BO_AXES, bo_proxy_stride=QMC_BO_PROXY_STRIDE,
        # fast Phase-1: keeps the existing axis-0 stride of 200 (proxy 166x35x35) and
        # only swaps the scoring -- 0.375 s/trial measured, so all 10 trials fit the
        # 6.0 s cap. Training dominates here (0.36 s of the 0.375 s), so shrinking the
        # proxy further buys nothing; 800/2 measured the same 0.37 s.
        bo_proxy_spatial=2, bo_eval_slices=32)
    del _qm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ─────────────────────────────────────────────────────────────────────────────
# Combined 2x3 figures: FFT magnitude error vs CR, and FFT phase error vs CR.
# Same panel layout / hollow-marker restyle as SPERR.py's PSNR figure, just swap
# the y-axis metric. Two separate figures (magnitude, phase) rather than cramming
# 2 sub-axes into each of 6 panels, so each figure stays as readable as the PSNR
# one. Only runs for TASK in ("all", "plot") -- a single-dataset process (e.g.
# --task nyx_b) has an incomplete `results` and must not try to draw the figure.
# (WarpX, Hurricane/CLOUDf48, Miranda 256³/diffusivity, SCALE-LETKF/W (with aux),
# SCALE-LETKF/T, and S3D are still benchmarked above via --task warpx/cloud/mir256/
# scale/scale_t/s3d/all, just
# not in this panel --
# swap them back into PANEL_ORDER/PANEL_TITLES if you want them in the figure again.)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# --task neurlz_long : how long does NeurLZ need to reach AdaMit's Table-2 PSNR?
# One SZ3 operating point per dataset (the Table-2 band = last rel), NeurLZ with its
# default configuration (Z slicing, matched-CR decompressed siblings) trained for a
# long budget with a PSNR evaluation after every epoch. Records the trajectory, the
# first time it reaches AdaMit's PSNR (from the pinned pkls), and its inference time.
#   SPERR_NLZ_DATASETS=nyx_b,nyx_t,nyx_d,mag,qmcpack,miranda   SPERR_NLZ_CAPS="nyx:300,mag:300,qmcpack:600,miranda:800"
#   SPERR_NLZ_TARGET_PIN=PAPER_ENHANCED_MIXED_CACHES.json       SPERR_NLZ_OUT=<json path>
if TASK == "neurlz_long":
    import json as _json
    _ds_list = [x.strip() for x in os.environ.get("SPERR_NLZ_DATASETS", "nyx_b,nyx_t,nyx_d,mag,qmcpack,miranda").split(",") if x.strip()]
    _caps = dict(nyx=300.0, mag=300.0, qmcpack=600.0, miranda=800.0)
    for kv in os.environ.get("SPERR_NLZ_CAPS", "").split(","):
        if ":" in kv:
            k, v = kv.split(":"); _caps[k.strip()] = float(v)
    _pin = _json.load(open(os.path.join(FFT_CACHE_DIR, os.environ.get("SPERR_NLZ_TARGET_PIN", "PAPER_ENHANCED_MIXED_CACHES.json"))))
    _out_path = os.environ.get("SPERR_NLZ_OUT", os.path.join(FFT_CACHE_DIR, "neurlz_long.json"))
    _out = _json.load(open(_out_path)) if os.path.isfile(_out_path) else {}
    _spec = {  # task -> (pin key, cap key, loader)
        "nyx_b":   ("Baryon",   "nyx",     None), "nyx_t": ("Temp", "nyx", None), "nyx_d": ("DMD", "nyx", None),
        "mag":     ("Magnetic", "mag",     lambda: (np.fromfile(MAG_FILE, dtype=np.float32).reshape(MAG_SHAPE), MAG_FILE, MAG_SHAPE, MAG_REL[-1], MAG_TIME_BUDGET, [], [])),
        "qmcpack": ("QMC",      "qmcpack", lambda: (np.fromfile(QMC_FILE, dtype=np.float32).reshape(QMC_SHAPE), QMC_FILE, QMC_SHAPE, QMC_REL[-1], QMC_TIME_BUDGET, [], [])),
        "miranda": ("Miranda",  "miranda", lambda: (np.fromfile(MIR_FILE, dtype=np.float32).reshape(MIR_SHAPE), MIR_FILE, MIR_SHAPE, MIR_REL[-1], MIR_TIME_BUDGET, [], [])),
    }
    _nyx_key = {"nyx_b": "baryon_density", "nyx_t": "temperature", "nyx_d": "dark_matter_density"}
    for _ds in _ds_list:
        pin_key, cap_key, loader = _spec[_ds]
        set_seed(SEED)
        if _ds in _nyx_key:
            tname = _nyx_key[_ds]
            gt = np.fromfile(NYX_DIR + tname + ".f32", dtype=np.float32).reshape(NYX_SHAPE)
            _sibs = [a for a in NYX_ALL if a != tname]
            aux_files = [NYX_DIR + a + ".f32" for a in _sibs]
            aux_raw = [np.memmap(f, dtype=np.float32, mode="r", shape=NYX_SHAPE) for f in aux_files]
            tfile, shape, rel, budget = NYX_DIR + tname + ".f32", NYX_SHAPE, NYX_REL[tname][-1], NYX_TIME_BUDGET
        else:
            gt, tfile, shape, rel, budget, aux_raw, aux_files = loader()
        drange = float(gt.max() - gt.min())
        _side = os.environ.get("SPERR_NLZ_SIDE", "sz3")                  # sz3 | sperr (base compressor of the operating point)
        _tp = pickle.load(open(os.path.join(FFT_CACHE_DIR, _pin[pin_key]), "rb"))
        if _side == "sperr":
            _cr_want = float(_tp["sperr"]["CR"][-1])                       # Table-2 SPERR point (aligned to the SZ3 band's CR)
            _tpsnr = _sperr_psnr_for_cr(tfile, gt, shape, drange, _cr_want, dtype=np.float32)
            cr_base, p_base, xq, sz_len = run_sperr(tfile, gt, shape, drange, float(_tpsnr), dtype=np.float32)
            xq = np.ascontiguousarray(xq, np.float32)
            rel = float(np.abs(gt - xq).max()) / max(drange, 1e-12)        # equivalent rel bound for NeurLZ's clamp
            target = float(_tp["sperr_pipe"]["PSNR"][-1]); ours_cr = float(_tp["sperr_pipe"]["CR"][-1])
        else:
            b, _ = sz_engine.compress(gt, 1, 0, float(rel), 0); sz_len = len(b)
            xq = np.ascontiguousarray(sz_engine.decompress(b, shape, np.float32), np.float32); del b
            cr_base = gt.nbytes / sz_len; p_base = compute_psnr(gt, xq, drange)
            target = float(_tp["pipe"]["PSNR"][-1]); ours_cr = float(_tp["pipe"]["CR"][-1])
        del _tp
        aux = []
        if aux_raw:
            level = min(AUX_CR_LEVELS, key=lambda L: abs(np.log(L) - np.log(cr_base)))
            for _a, _f in zip(aux_raw, aux_files):
                _ad, _cr, _k = _aux_at_cr_level(_a, _f, shape, level, _side)
                aux.append(np.ascontiguousarray(_ad, np.float32))
                print(f"  [aux] {os.path.basename(_f):24s} {_side} level {level}: CR {_cr:6.1f}", flush=True)
            del aux_raw
        cap = float(_caps[cap_key])
        print(f"\n[neurlz_long] {_ds} ({_side}): base CR {cr_base:.1f} / {p_base:.2f} dB | AdaMit target {target:.2f} dB (+{target-p_base:.2f}) @ CR {ours_cr:.0f} "
              f"| AdaMit budget {budget:.0f}s | NeurLZ cap {cap:.0f}s", flush=True)
        globals()["NEURLZ_VERBOSE"] = True
        globals()["NEURLZ_STOP_PSNR"] = target if int(os.environ.get("SPERR_NLZ_STOP_AT_TARGET", "1")) else None
        _ep_fixed = int(os.environ.get("SPERR_NLZ_EPOCHS", "0"))          # >0: fixed epoch count (NeurLZ's default 100) instead of a time cap
        p_nlz, nlz_params, t_train, n_ep, hist, enh = run_neurlz(gt, xq, aux, shape, drange, float(rel), _ep_fixed, NEURLZ_FEATURES,
                                                                 time_budget=(None if _ep_fixed else cap), slice_axis=0, return_history=True)
        t_inf = float(globals().get("NEURLZ_LAST_INFER_S", float("nan")))
        globals()["NEURLZ_STOP_PSNR"] = None
        del enh
        ht, hp = hist["time"], hist["psnr"]
        def _first(th):
            for t, p_ in zip(ht, hp):
                if p_ >= th: return float(t)
            return None
        at_budget = max([p_ for t, p_ in zip(ht, hp) if t <= budget], default=None)
        rec = dict(rel=float(rel), cr_base=cr_base, psnr_base=float(p_base), psnr_target=target, ours_cr=ours_cr,
                   adamit_budget_s=budget, cap_s=cap, nlz_params=int(nlz_params), nlz_cr=gt.nbytes / (sz_len + nlz_params * BYTES_PER_PARAM),
                   nlz_epochs=int(n_ep), nlz_train_s=float(t_train), nlz_infer_s=t_inf,
                   nlz_psnr_at_adamit_budget=at_budget, nlz_best_psnr=float(max(hp)) if hp else None,
                   nlz_time_to_best=float(ht[int(np.argmax(hp))]) if hp else None,
                   nlz_time_to_target=_first(target), nlz_time_to_half_gain=_first(p_base + 0.5 * (target - p_base)),
                   hist_time=ht, hist_psnr=hp)
        rec["side"] = _side
        _out[_ds if _side == "sz3" else f"{_ds}_sperr"] = rec
        _json.dump(_out, open(_out_path, "w"), indent=1)
        print(f"[neurlz_long] {_ds}: NeurLZ {n_ep} ep / {t_train:.0f}s -> best {rec['nlz_best_psnr']:.2f} dB at {rec['nlz_time_to_best']:.0f}s | "
              f"@AdaMit budget {at_budget} | reach target: {rec['nlz_time_to_target']} s | half gain: {rec['nlz_time_to_half_gain']} s | infer {t_inf:.2f}s", flush=True)
        del gt, xq, aux
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    sys.exit(0)
