"""End-to-end compression / decompression timing for the 6 paper datasets.

Every dataset is binary-searched to CR 500 and trained for 10 epochs
(iso-epoch). The single-GPU job also runs Phase 1 -- a TPE search over the
slice axis and the learning rate, the same space SPERR.py searches -- and the
4-GPU job is handed the winner, so the two differ only in GPU count. Both take
the same number of slices per optimizer update (iso-batch); that number is set
per dataset from the slice size, since 69x69 slices need far more per update
than 512x512 ones to keep the GPU busy.

    compression time   = base-codec compress + residual-model training
    decompression time = base-codec decompress + full-volume neural inference

--codec picks the base codec the residual model corrects: sz3 (default) or
sperr, giving the "SZ3 + Ours" and "SPERR + Ours" halves of the table. Both are
binary-searched to the same CR, so the two halves sit at one operating point.
Everything after the codec -- Phase 1, the model, the shard split, the clamp --
is identical; only the base reconstruction differs.

Two things are NOT comparable across codecs, and the table note says so:

  * Timing method. SZ3 is called in-process through pysz on an in-memory array.
    SPERR ships as a CLI, so its number is a subprocess: fork + read the input
    file + write the bitstream. That overhead is real for a user, but it is not
    the same measurement, and at CR 500 the streams are small enough that file
    I/O is a visible share of it.
  * Error envelope. run_bg_inference clamps the correction to the base codec's
    error bound. For SZ3 that is the rel_err the bisection asked for; SPERR
    targets a PSNR, not an L-inf bound, so we use the max relative error its
    reconstruction actually achieved (what SPERR.py does). At CR 500 on NYX
    baryon density that envelope is ~30x looser than SZ3's, so the model has
    correspondingly more room -- a property of the codecs, not of the training.

The 4-GPU column is not reported at 10 epochs but at the wall time it first
matches the single-GPU 10-epoch PSNR, so both columns describe the same output
quality. Per-epoch PSNR is evaluated to locate that crossing; train_bg_only
excludes evaluation from its own clock, so the reported training time is clean.

    # 1 GPU, 10 epochs, Phase-1 TPE over (axis, lr)
    torchrun --nproc_per_node=1 bench_compress_table.py --dataset nyx_b --tag nyx_b_n1
    # 4 GPUs, same axis/lr/update count
    torchrun --nproc_per_node=4 bench_compress_table.py --dataset nyx_b \
        --tag nyx_b_n4 --axis 0 --lr_abs 3.2e-3
    # same, on top of SPERR
    torchrun --nproc_per_node=1 bench_compress_table.py --dataset nyx_b \
        --codec sperr --tag nyx_b_n1
    # build the table (mixing codecs is fine; rows are grouped by codec)
    python bench_compress_table.py --table out/*.json
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Machine-specific locations come from base_script/local_paths.py (env var >
# <repo>/local_paths.env > placeholder), like every other script in the repo.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # repository root
sys.path.append(os.path.join(ROOT, "base_script"))
from local_paths import P  # noqa: E402
sys.path.append(P("ADAMIT_PYSZ"))
SZ3_LIB = P("ADAMIT_SZ3_LIB")
SPERR_BIN = P("ADAMIT_SPERR_BIN")
NYX_DIR = os.path.join(P("ADAMIT_NYX_DIR"), "")
from aux_siblings import (NYX_ALL, CASCADE_ORDER, load_siblings,
                          export_enhanced, available_levels, closest_level)

# name: (label, target, shape, aux stems, param budget)
DATASETS = {
    # train_s is the paper's pure-training wall-clock budget for the single-GPU
    # run; --iso-time uses it instead of a fixed epoch count.
    "nyx_b":   dict(label="NYX — baryon density", stem="baryon_density",
                    shape=(512, 512, 512), budget=30000, nyx=True, train_s=10.0),
    "nyx_t":   dict(label="NYX — temperature", stem="temperature",
                    shape=(512, 512, 512), budget=30000, nyx=True, train_s=10.0),
    "nyx_d":   dict(label="NYX — dark matter density", stem="dark_matter_density",
                    shape=(512, 512, 512), budget=30000, nyx=True, train_s=10.0),
    # Random sampling for the same reason as QMCPack below: sequential batches
    # (4 CONSECUTIVE slices per update) stall the single-GPU baseline -- on the
    # 238k Miranda corrector it plateaued 1.4 dB below the 4-GPU run, whose
    # one-slice-per-shard updates cover the volume by construction. Random
    # draws give the single GPU the same coverage, so the 1-vs-4 comparison
    # measures hardware, not the sampler.
    "miranda": dict(label="Miranda", path=P("ADAMIT_MIRANDA_FILE"),
                    shape=(1024, 1024, 1024), budget=240000, nyx=False,
                    sample="random", train_s=80.0),
    "warpx":   dict(label="WarpX", path=P("ADAMIT_WARPX_FILE", "/path/to/data/wpx-256_256_2048_double.raw"),
                    shape=(2048, 256, 256), budget=30000, nyx=False,
                    src_shape=(256, 256, 2048), dtype=np.float64, transpose=(2, 0, 1)),
    # SDRBENCH treats einspline as 69x69x33120 (sz -3 69 69 33120), i.e. 33120
    # slices of 69x69 -- the layout its published numbers use, and 6.4 dB better
    # at CR 500 than tiling the 288 orbitals into one plane (tiling puts 288
    # artificial discontinuities where SZ3's predictor breaks down).
    # Slices are tiny, so batch scales up to keep per-update work comparable to
    # the other datasets: 256 slices/update = 1.22 M px, vs NYX's 4 x 512^2.
    # Random sampling, not sequential: with depth 33120 a sequential batch of
    # 256 is 256 CONSECUTIVE slices, i.e. ~2 of the 288 orbitals, and the model
    # trained on it ends 1.7 dB BELOW the SZ3 base. Random draws cover the
    # volume, which is what the 4-GPU shards give for free.
    "qmc":     dict(label="QMCPack", nyx=False, batch=256, budget=30000,
                    sample="random",
                    path=P("ADAMIT_QMC_FILE"),
                    shape=(33120, 69, 69)),
    # Random sampling too: with sequential batches the single-GPU curve dipped
    # 3 dB mid-run and only recovered on the last epoch, making its
    # time-to-quality meaningless.
    "mag":     dict(label="Magnetic Reconnection",
                    path=P("ADAMIT_MAGNETIC_FILE"),
                    shape=(512, 512, 512), budget=30000, nyx=False,
                    sample="random"),
}


# Phase-1 search space, mirroring SPERR.py: bring axis k to the front and let
# TPE pick (axis, lr) on a downsampled proxy. The lr arm is [1e-3, 1e-2], the
# paper-final window. It used to start at 1e-4, but across six NYX runs TPE
# never chose below 7.8e-3 -- the extra decade only diluted a 10-trial budget.
# Note the winners crowd the upper bound (9.95e-3, 9.42e-3), so the optimum may
# well sit above 1e-2; that is a question for an ablation, not this table.
_DIR_FWD = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 0, 1)}
H_CANDIDATES = tuple(range(3, 128))
PROXY_DS = 2
BO_N_TRIALS = int(os.environ.get("BO_N_TRIALS", 10))
BO_N_STARTUP = 3
# The paper range is [1e-3, 1e-2]; the env override exists to probe whether
# the upper bound binds (five of six enhanced runs picked lr within 0.3% of
# the cap) without changing what a stock run does.
BO_LR_MIN = float(os.environ.get("BO_LR_MIN", 1e-3))
BO_LR_MAX = float(os.environ.get("BO_LR_MAX", 1e-2))
BO_ENQUEUE_LR = 1e-3
BO_PHASE1_EPOCHS, BO_MIN_GAIN_DB = 3, 0.30


def _perm(a, k):
    return a if k == 0 else np.ascontiguousarray(np.transpose(np.asarray(a), _DIR_FWD[k]))


def set_seed(s=17):
    import torch
    torch.manual_seed(s); np.random.seed(s); random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def psnr_of(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    dr = float(a.max() - a.min()) or 1.0
    mse = float(np.mean((a - b) ** 2))
    return 100.0 if mse <= 0 else 20 * np.log10(dr) - 10 * np.log10(mse)


def load_target(spec):
    """Return the target volume as a contiguous float32 array of spec['shape']."""
    if spec["nyx"]:
        return np.fromfile(NYX_DIR + spec["stem"] + ".f32",
                           dtype=np.float32).reshape(spec["shape"])
    dt = spec.get("dtype", np.float32)
    src = spec.get("src_shape", spec["shape"])
    v = np.fromfile(spec["path"], dtype=dt).reshape(src)
    if dt != np.float32:
        v = v.astype(np.float32)
    if spec.get("transpose"):
        v = np.ascontiguousarray(np.transpose(v, spec["transpose"]))
    return v


def rel_for_cr(sz, gt, shape, target_cr, iters=22):
    """Geometric bisection on rel_err; CR is monotone in it."""
    lo, hi, best, bd = 1e-9, 1e-1, None, float("inf")
    orig = int(np.prod(shape)) * 4
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        b, _ = sz.compress(gt, 1, 0, mid, 0)
        cr = orig / len(b)
        if abs(cr - target_cr) < bd:
            best, bd = mid, abs(cr - target_cr)
        if cr < target_cr:
            lo = mid
        else:
            hi = mid
    return best


# ── SPERR as the base codec ────────────────────────────────────────────────
# sperr3d is a CLI, so unlike SZ3 (in-process via pysz) every call here is a
# subprocess over files. See the module docstring for what that costs.

def _sperr_env():
    """Env for sperr3d, with libSPERR.so on LD_LIBRARY_PATH.

    The build tree keeps the .so in a sibling of bin/, and it is usually not on
    the system path -- without this the subprocess dies with rc=127 before it
    ever looks at the data. Same search SPERR.py does.
    """
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(SPERR_BIN))
    dirs = {os.path.dirname(h)
            for h in glob.glob(os.path.join(root, "**", "libSPERR.so*"), recursive=True)}
    if dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            sorted(dirs) + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    return env


_SPERR_ENV = _sperr_env()


def sperr_source_path(spec, tmp, tag):
    """(path, is_temp) of the raw float32 volume sperr3d should read.

    Most datasets already are exactly that on disk and are handed over
    untouched. WarpX is float64 in a different axis order, so it has to be
    written out once -- hence the flag, which tells the caller both to
    materialize it and to delete it afterwards. Deriving the path without the
    data lets every rank agree on it while only rank 0 does the writing.
    """
    if not spec.get("dtype") and not spec.get("transpose"):
        return (NYX_DIR + spec["stem"] + ".f32" if spec["nyx"] else spec["path"]), False
    return os.path.join(tmp, f"sperr_src_{tag}.f32"), True


def sperr_compress(src, shape, q, bit, omp):
    """Encode src at PSNR target q. Returns (bytes on disk, wall seconds)."""
    d, h, w = int(shape[0]), int(shape[1]), int(shape[2])
    cmd = [SPERR_BIN, "-c", "--ftype", "32",
           "--dims", str(w), str(h), str(d),      # sperr3d wants fastest-varying first
           "--psnr", f"{float(q):.6f}", "--omp", str(int(omp)),
           "--bitstream", bit, src]
    if os.path.exists(bit):
        os.remove(bit)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=_SPERR_ENV)
    dt = time.perf_counter() - t0
    if not os.path.exists(bit):
        raise RuntimeError(f"sperr3d compress failed (rc={p.returncode}) "
                           f"stdout={p.stdout!r} stderr={p.stderr!r}")
    return os.path.getsize(bit), dt


def sperr_decompress(bit, dec, omp):
    """Decode bit to the float32 file dec. Returns wall seconds."""
    if os.path.exists(dec):
        os.remove(dec)
    t0 = time.perf_counter()
    p = subprocess.run([SPERR_BIN, "-d", "--decomp_f", dec, "--omp", str(int(omp)), bit],
                       capture_output=True, text=True, env=_SPERR_ENV)
    dt = time.perf_counter() - t0
    if not os.path.exists(dec):
        raise RuntimeError(f"sperr3d decompress failed (rc={p.returncode}) "
                           f"stdout={p.stdout!r} stderr={p.stderr!r}")
    return dt


def sperr_q_for_cr(src, shape, target_cr, tmp, omp, lo=20.0, hi=200.0, iters=16):
    """Bisect the --psnr target whose CR ~= target_cr; CR falls as q rises.

    Linear, not geometric like rel_for_cr: SPERR's knob is already in dB, so
    the quantity being searched is logarithmic in the error to begin with.
    """
    orig = int(np.prod(shape)) * 4
    bit = os.path.join(tmp, "sperr_cr_probe.bit")
    best, bd = None, float("inf")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        n, _ = sperr_compress(src, shape, mid, bit, omp)
        cr = orig / n
        if abs(cr - target_cr) < bd:
            best, bd = mid, abs(cr - target_cr)
        if cr > target_cr:
            lo = mid
        else:
            hi = mid
    if os.path.exists(bit):
        os.remove(bit)
    return best


def phase1_bo(gt, x_lq, aux, spec, device, rel, batch, log):
    """TPE over (slice axis, lr) on a PROXY_DS-downsampled proxy.

    Follows SPERR.py's Phase 1, with one adaptation: the proxy trains at the
    dataset's own batch rather than batch=1. The lr that wins at batch=1 is not
    the lr that wins at batch=256, so matching the batch is what makes the
    search transferable to the full run (and it keeps QMCPack's 16k-slice
    axis-0 proxy from dominating the search cost).
    """
    import optuna
    import torch
    from experiment import build_bg_only_cfg
    from bg_stage import train_bg_only, run_bg_inference, unwrap_bg_model
    from bg_shard import pick_bg_h_under_budget

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    shape = np.asarray(gt).shape
    n_fields = 1 + len(aux)
    bg_h, _ = pick_bg_h_under_budget(int(spec["budget"]), shape=shape,
                                     n_fields=n_fields, bg_arch="spatial",
                                     h_candidates=H_CANDIDATES)
    bg_h = int(bg_h)

    proxy = {}
    for k in (0, 1, 2):
        idx = np.arange(0, int(shape[k]), PROXY_DS)
        def take(a):
            v = np.transpose(np.take(np.asarray(a, np.float32), idx, axis=k), _DIR_FWD[k])
            return np.ascontiguousarray(v[:, ::PROXY_DS, ::PROXY_DS])
        t, b = take(gt), take(x_lq)
        ax = [take(a) for a in aux]
        proxy[k] = ([t] + ax, [b] + ax, psnr_of(t, b))
        log(f"  [bo] axis{k} proxy {t.shape} base={proxy[k][2]:.2f} dB")

    def objective(trial):
        k = trial.suggest_categorical("direction", [0, 1, 2])
        lr_c = trial.suggest_float("lr", BO_LR_MIN, BO_LR_MAX, log=True)
        Xs_t, Xps_t, base_p = proxy[k]
        nz, hh, ww = Xs_t[0].shape
        nb = max(1, min(int(batch), nz))
        set_seed(17)
        cfg = build_bg_only_cfg(
            X_target=Xs_t[0], Xps=Xps_t, max_train_time=1e9, bg_h=bg_h, roi_h=4,
            epochs=BO_PHASE1_EPOCHS, steps_per_epoch=max(1, nz // nb),
            bg_patch_size=int(min(hh, ww)), bg_batch=nb, lr=float(lr_c),
            bg_freq_weight=0.5, bg_fft_phase_weight=0.5, bg_freq_warmup_epochs=1,
            bg_field_norm="zscore")
        cfg.bg_arch = "spatial"; cfg.bg_split_mode = "three"; cfg.bg_split_bands = True
        cfg.bg_split_sigma = 0.12; cfg.bg_sigma_low = 0.08; cfg.bg_sigma_mid = 0.18
        cfg.bg_low_weight = 0.2; cfg.bg_mid_weight = 0.5; cfg.bg_high_weight = 1.0
        cfg.bg_cr_rel_err = rel; cfg.rel_err = rel
        cfg.bg_gpu_sampling = True; cfg.seed = 17
        cfg.bg_sample_mode = str(spec.get("sample", "sequential"))
        cfg.amp = True; cfg.amp_dtype = "bf16"
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            m, _ = train_bg_only(Xs=Xs_t, Xps=Xps_t, device=device, cfg=cfg,
                                 evaluator=None)
            p = psnr_of(Xs_t[0], run_bg_inference(unwrap_bg_model(m), Xs_t, Xps_t,
                                                  cfg, rel))
        del m
        torch.cuda.empty_cache()
        if not np.isfinite(p):
            p = -1e9
        trial.set_user_attr("gain", float(p - base_p))
        return float(p)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=17, n_startup_trials=BO_N_STARTUP))
    for k in (0, 1, 2):
        study.enqueue_trial({"direction": k, "lr": BO_ENQUEUE_LR})
    study.optimize(objective, n_trials=BO_N_TRIALS)

    per_dir = {}
    for t in study.trials:
        if t.value is None:
            continue
        d = int(t.params["direction"])
        if d not in per_dir or t.value > per_dir[d][1]:
            per_dir[d] = (t.params["lr"], t.value)
    summ = " ".join(f"axis{d}:{v:.2f}@{l:.1e}" for d, (l, v) in sorted(per_dir.items()))
    gain = float(study.best_trial.user_attrs.get("gain", 0.0))
    best_k, best_lr = int(study.best_params["direction"]), float(study.best_params["lr"])
    log(f"  [bo] {BO_N_TRIALS} trials -> axis{best_k} lr={best_lr:.2e} "
        f"gain={gain:+.2f} dB | {summ}")
    if gain <= BO_MIN_GAIN_DB:
        log(f"  [bo] gain {gain:+.2f} <= {BO_MIN_GAIN_DB} dB -> fall back to axis0, lr=1e-3")
        return 0, 1e-3, gain, summ
    return best_k, best_lr, gain, summ


def run(args):
    import torch
    import torch.distributed as dist
    from pysz import SZ
    from experiment import build_bg_only_cfg
    from bg_stage import train_bg_only, run_bg_inference, unwrap_bg_model
    from bg_shard import pick_bg_h_under_budget

    spec = DATASETS[args.dataset]
    SHAPE = spec["shape"]
    depth = SHAPE[0]

    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        rank = 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    assert depth % world == 0, f"depth {depth} not divisible by world {world}"

    gt = load_target(spec)

    def bcast(vals):
        """Broadcast a list of floats from rank 0 and return it everywhere."""
        t = torch.tensor([float(v) for v in vals], dtype=torch.float64, device=device)
        if world > 1:
            dist.broadcast(t, src=0)
        return t.tolist()

    def barrier():
        if world > 1:
            dist.barrier()

    sperr_q = 0.0
    # Built unconditionally: the sibling archive is read even on the SPERR
    # path, and constructing the ctypes wrapper costs nothing.
    sz = SZ(SZ3_LIB)
    if args.codec == "sz3":
        # rel is the requested L-inf bound, and doubles as the clamp envelope
        # run_bg_inference holds the correction inside.
        rel, = bcast([args.rel if args.rel > 0 else rel_for_cr(sz, gt, SHAPE, args.cr)]
                     if rank == 0 else [0.0])

        # ---- timed compress / decompress (the codec half of the pipeline) ----
        # Every rank runs it: the call is in-process and cheap, and this is how
        # each rank gets x_lq without a multi-GB broadcast.
        t0 = time.perf_counter()
        b, _ = sz.compress(gt, 1, 0, rel, 0)
        t_base_comp = time.perf_counter() - t0
        t0 = time.perf_counter()
        x_lq = sz.decompress(b, SHAPE, np.float32)
        t_base_decomp = time.perf_counter() - t0
        base_bytes = len(b)
    else:
        # sperr3d is a subprocess over files, so unlike SZ3 it is NOT run once
        # per rank: four concurrent encodes would contend for the same cores and
        # inflate the number we report. Rank 0 runs it alone and the others read
        # the decoded volume it leaves in the node-local tmp dir.
        src, src_tmp = sperr_source_path(spec, args.tmp, args.tag)
        if rank == 0 and src_tmp:
            np.ascontiguousarray(gt, dtype=np.float32).tofile(src)
        barrier()

        q, = bcast([args.sperr_q if args.sperr_q > 0
                    else sperr_q_for_cr(src, SHAPE, args.cr, args.tmp, args.sperr_omp)]
                   if rank == 0 else [0.0])
        sperr_q = q

        bit = os.path.join(args.tmp, f"sperr_{args.tag}.bit")
        dec = os.path.join(args.tmp, f"sperr_{args.tag}.dec.f32")
        if rank == 0:
            nb, t_base_comp = sperr_compress(src, SHAPE, q, bit, args.sperr_omp)
            t_base_decomp = sperr_decompress(bit, dec, args.sperr_omp)
        else:
            nb = t_base_comp = t_base_decomp = 0.0
        barrier()
        nb, t_base_comp, t_base_decomp = bcast([nb, t_base_comp, t_base_decomp])
        base_bytes = int(nb)
        x_lq = np.fromfile(dec, dtype=np.float32).reshape(SHAPE)
        barrier()
        if rank == 0:
            for f in (bit, dec) + ((src,) if src_tmp else ()):
                if os.path.exists(f):
                    os.remove(f)

        # SPERR targets a PSNR, not an L-inf bound, so there is no requested
        # rel to clamp against -- use the max relative error it actually left,
        # which is the envelope SPERR.py hands the residual model.
        rel, = bcast([float(np.abs(gt - x_lq).max())
                      / (float(gt.max() - gt.min()) or 1.0)] if rank == 0 else [0.0])

    cr_base = (int(np.prod(SHAPE)) * 4) / base_bytes

    # ---- siblings ------------------------------------------------------
    # At decode time only the ARCHIVED siblings exist, so the model has to be
    # trained on the same lossy volumes it will later be handed. Feeding the
    # originals here inflates the result and is not decoder-reproducible.
    aux, aux_names = [], []
    if spec["nyx"]:
        def _sperr_sib(bin_path, shp):
            dec = os.path.join(args.tmp, f"sib_{os.getpid()}.f32")
            sperr_decompress(bin_path, dec, args.sperr_omp)
            v = np.fromfile(dec, dtype=np.float32).reshape(shp)
            os.remove(dec)
            return v

        aux, aux_names = load_siblings(
            spec["stem"], cr_base, SHAPE, sz=sz, codec=args.codec,
            mode=args.aux_mode, enhanced_dir=(args.aux_enhanced_dir or None),
            sperr_decode=_sperr_sib,
            log=(log if rank == 0 else (lambda *a, **k: None)))
    # ---- Phase 1: pick (slice axis, lr). The base codec already ran on the
    # ORIGINAL layout, so CR and base PSNR are axis-independent; only the
    # residual model sees the permuted volume, exactly as SPERR.py does. The
    # 4-GPU job is given the axis/lr the 1-GPU job found, so both train the
    # same configuration.
    bo_gain, bo_summ = None, None
    if args.axis >= 0:
        axis, use_lr = int(args.axis), float(args.lr_abs)
        log(f"[phase1] using given axis{axis} lr={use_lr:.2e}")
    elif args.bo:
        t_bo = time.perf_counter()
        axis, use_lr, bo_gain, bo_summ = phase1_bo(
            gt, x_lq, aux, spec, device, rel, int(spec.get("batch", args.batch)), log)
        log(f"[phase1] BO took {time.perf_counter() - t_bo:.1f}s")
    else:
        axis, use_lr = 0, args.lr * args.lr_mult

    if axis != 0:
        gt, x_lq = _perm(gt, axis), _perm(x_lq, axis)
        aux = [_perm(a, axis) for a in aux]
        SHAPE = tuple(gt.shape)
        depth = SHAPE[0]

    Xs_full, Xps_full = [gt] + aux, [x_lq] + aux
    n_fields = len(Xs_full)

    # pick_bg_h_under_budget defaults to h <= 29, which silently leaves most of a
    # large budget unspent: Miranda's 240k budget picked h=29 = 54k parameters.
    # Widen the candidates so the budget is what actually binds.
    bg_h, n_params = pick_bg_h_under_budget(int(spec["budget"]), shape=SHAPE,
                                            n_fields=n_fields, bg_arch="spatial",
                                            h_candidates=H_CANDIDATES)
    bg_h = int(bg_h)
    # effective CR counts the model as part of the payload (fp32 weights)
    cr_eff = (int(np.prod(SHAPE)) * 4) / (base_bytes + int(n_params) * 4)

    # Train on a depth that divides by 4 whatever the GPU count, so the 1-GPU
    # and 4-GPU runs see the same slices (axis 1/2 can give an odd depth, e.g.
    # QMCPack's 69). Evaluation always covers the full volume.
    train_depth = (depth // 4) * 4 or depth
    per = train_depth // world
    z0, z1 = rank * per, (rank + 1) * per
    Xs_c = [np.ascontiguousarray(x[z0:z1]) for x in Xs_full]
    Xps_c = [np.ascontiguousarray(x[z0:z1]) for x in Xps_full]

    # Only rank 0 evaluates on the full volume; the others drop their copies so
    # four ranks do not each hold two full volumes (8 GB apiece on Miranda).
    if rank != 0:
        Xs_full = Xps_full = None
        aux = None
        gt = x_lq = None

    # iso-batch: the same number of slices per optimizer update either way, so
    # 1 GPU takes all of them and each of 4 GPUs takes a quarter. The total is
    # per-dataset because it tracks slice size -- 69x69 slices need many more
    # per update than 512x512 ones to keep the GPU busy.
    # --tot-batch overrides everything: it exists so QMCPack can be run in the
    # weak-scaling convention (every GPU keeps its natural batch of 256, total
    # 1024) instead of splitting the spec batch across ranks.
    if int(getattr(args, "tot_batch", 0)) > 0:
        tot_batch = int(args.tot_batch)
    else:
        tot_batch = int(spec.get("batch", args.batch))
    batch = tot_batch if world == 1 else max(1, tot_batch // world)
    steps = max(1, per // max(1, batch))

    # Budget. The paper trains for a fixed wall-clock time per dataset, with the
    # epoch count only as a cap; --epochs alone reproduces the earlier
    # epoch-budgeted runs.
    if args.train_s > 0 or (args.iso_time and spec.get("train_s")):
        train_budget = float(args.train_s if args.train_s > 0 else spec["train_s"])
        epoch_cap = int(args.epoch_cap)
    else:
        train_budget, epoch_cap = 1e9, int(args.epochs)

    log(f"[setup] {torch.cuda.get_device_name(local_rank)} x{world} | {spec['label']} "
        f"{SHAPE} | codec={args.codec} rel={rel:.4e} -> CR_base={cr_base:.1f} "
        f"CR_eff={cr_eff:.1f} | "
        f"bg_h={bg_h} params={n_params:,} n_fields={n_fields} | "
        f"batch={batch}/rank (total {tot_batch}) steps/rank={steps} "
        + (f"epochs={epoch_cap} (cap) budget={train_budget:.1f}s"
           if train_budget < 1e8 else f"epochs={epoch_cap}")
        + f" | axis{axis} lr={use_lr:.2e}")
    log(f"[codec] {args.codec} compress {t_base_comp:.2f}s | decompress {t_base_decomp:.2f}s"
        + (f" | --psnr {sperr_q:.3f}" if args.codec == "sperr" else "")
        + f" | stream {base_bytes/1e6:.2f} MB")

    set_seed(args.seed)
    cfg = build_bg_only_cfg(
        X_target=Xs_c[0], Xps=Xps_c, max_train_time=train_budget, bg_h=bg_h, roi_h=4,
        epochs=epoch_cap, steps_per_epoch=steps,
        bg_patch_size=min(SHAPE[1], SHAPE[2]), bg_batch=batch,
        lr=use_lr, bg_freq_weight=0.5, bg_fft_phase_weight=0.5,
        bg_freq_warmup_epochs=1, bg_field_norm="zscore")

    # Under DDP every rank must normalize inputs and scale the residual target
    # with the SAME statistics; per-shard stats silently cost ~1 dB.
    if world > 1:
        st = torch.zeros(4 + 4 * n_fields, dtype=torch.float64, device=device)
        if rank == 0:
            _r = Xs_full[0] - Xps_full[0]
            vals = [float(np.mean(_r)), float(np.std(_r)) + 1e-8,
                    float(np.min(_r)), float(np.max(_r))]
            for f in Xps_full:
                vals += [float(np.mean(f)), float(np.std(f)) + 1e-8,
                         float(np.min(f)), float(np.max(f))]
            st[:] = torch.tensor(vals, dtype=torch.float64, device=device)
            del _r
        dist.broadcast(st, src=0)
        v = st.tolist()
        cfg.res_mean, cfg.res_std, cfg.res_min, cfg.res_max = v[0], v[1], v[2], v[3]
        cfg.input_means = [v[4 + 4 * i] for i in range(n_fields)]
        cfg.input_stds = [v[5 + 4 * i] for i in range(n_fields)]
        cfg.input_mins = [v[6 + 4 * i] for i in range(n_fields)]
        cfg.input_maxs = [v[7 + 4 * i] for i in range(n_fields)]

    cfg.bg_arch = "spatial"; cfg.bg_split_mode = "three"; cfg.bg_split_bands = True
    cfg.bg_split_sigma = 0.12; cfg.bg_sigma_low = 0.08; cfg.bg_sigma_mid = 0.18
    cfg.bg_low_weight = 0.2; cfg.bg_mid_weight = 0.5; cfg.bg_high_weight = 1.0
    cfg.bg_cr_rel_err = rel; cfg.rel_err = rel
    cfg.bg_gpu_sampling = True; cfg.seed = args.seed
    cfg.bg_sample_mode = str(spec.get("sample", "sequential"))
    cfg.amp = True; cfg.amp_dtype = "bf16"
    cfg.bg_ddp = world > 1
    cfg.bg_data_parallel = False
    # Local SGD: each rank trains an independent batch-1 replica on its own
    # z-shard and the weights are averaged every K steps. Keeps the batch-1
    # update dynamics that DDP's gradient averaging (total batch = world)
    # gives up -- the single-GPU batch-1 baseline beats total-batch-4 DDP on
    # several fields, so this is the 4-GPU mode that can actually match it.
    if world > 1 and int(args.localsgd_every) > 0:
        cfg.bg_localsgd_every = int(args.localsgd_every)
    cfg.bg_log_prefix = f"r{rank}"

    infer_s = []

    def ev(m, _c=cfg):
        t = time.perf_counter()
        rec = run_bg_inference(unwrap_bg_model(m), Xs_full, Xps_full, _c, rel)
        infer_s.append(time.perf_counter() - t)
        return psnr_of(Xs_full[0], rec), 0.0

    model, hist = train_bg_only(Xs=Xs_c, Xps=Xps_c, device=device, cfg=cfg,
                                evaluator=ev if rank == 0 else None)

    if rank == 0 and args.export_enhanced_dir and spec["nyx"]:
        # One extra inference with the trained model: this volume is what a
        # later cascade stage consumes as an ENHANCED sibling.
        rec_vol = run_bg_inference(unwrap_bg_model(model), Xs_full, Xps_full, cfg, rel)
        export_enhanced(args.export_enhanced_dir, spec["stem"], cr_base, rec_vol,
                        codec=args.codec, log=log)
        del rec_vol

    if rank == 0:
        ps = [p[1] if isinstance(p, (tuple, list)) else p for p in hist["psnr"]]
        # steady-state inference cost (first call pays warm-up / autotune)
        t_infer = float(np.median(infer_s[1:] or infer_s))
        t_train = float(hist["time"][-1])
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        rec = dict(
            tag=args.tag, dataset=args.dataset, label=spec["label"],
            shape=list(SHAPE), gpu=torch.cuda.get_device_name(local_rank),
            world_size=world, batch=batch, total_batch=tot_batch,
            steps_per_rank=steps,
            epochs=args.epochs, lr=float(cfg.lr), axis=int(axis),
            bo_gain=bo_gain, bo_summary=bo_summ, train_depth=int(train_depth),
            codec=args.codec, sperr_q=(sperr_q or None),
            rel=rel, cr_sz3=cr_base, cr_eff=cr_eff, sz_bytes=int(base_bytes),
            bg_h=bg_h, params=int(n_params), n_fields=n_fields,
            train_budget_s=(None if train_budget > 1e8 else float(train_budget)),
            aux_mode=args.aux_mode, aux_names=aux_names,
            aux_enhanced_dir=(args.aux_enhanced_dir or None),
            t_sz_compress=t_base_comp, t_sz_decompress=t_base_decomp,
            t_train=t_train, t_infer=t_infer,
            t_compress=t_base_comp + t_train, t_decompress=t_base_decomp + t_infer,
            base_psnr=psnr_of(Xs_full[0], Xps_full[0]),
            time=[float(t) for t in hist["time"]], psnr=ps,
            # train_bg_only restores the best-PSNR weights before returning, so
            # the model the pipeline hands over is the peak of the curve, not
            # its last point. On SPERR baryon density the two differ by 2.3 dB,
            # and the last epoch even lands below the base reconstruction.
            final_psnr=float(max(ps)),
            last_psnr=float(ps[-1]),
            best_epoch=int(int(np.argmax(ps)) + 1))
        (out / f"{args.tag}.json").write_text(json.dumps(rec, indent=2))
        log(f"[done] {spec['label']} x{world}gpu | final {ps[-1]:.2f} dB | "
            f"compress {rec['t_compress']:.2f}s ({args.codec} {t_base_comp:.2f} + train {t_train:.2f}) | "
            f"decompress {rec['t_decompress']:.2f}s ({args.codec} {t_base_decomp:.2f} + infer {t_infer:.2f})")

    if world > 1:
        dist.destroy_process_group()


def t_to_reach(r, q):
    tt, pp = r["time"], np.maximum.accumulate(r["psnr"])
    for i in range(1, len(pp)):
        if pp[i] >= q:
            if pp[i] == pp[i - 1]:
                return float(tt[i])
            f = (q - pp[i - 1]) / (pp[i] - pp[i - 1])
            return float(tt[i - 1] + f * (tt[i] - tt[i - 1]))
    return None


# PSNR varies by ~0.1-0.3 dB run to run, so "matched the single-GPU quality"
# cannot be a bit-exact >= test; a 4-GPU run 0.02 dB short has matched it.
MATCH_TOL = 0.05


CODEC_LABEL = {"sz3": "SZ3 + Ours", "sperr": "SPERR + Ours"}


def _group(paths):
    """{codec: {dataset: {world_size: record}}}, codecs in CODEC_LABEL order.

    Records written before --codec existed have no key and are SZ3, which is
    what they were.
    """
    by = {}
    for p in paths:
        r = json.loads(Path(p).read_text())
        by.setdefault(r.get("codec", "sz3"), {}).setdefault(r["dataset"], {})[r["world_size"]] = r
    return {c: by[c] for c in CODEC_LABEL if c in by}


def _iso_psnr_row(a, b):
    """(target, t1_reach, t4_reach) for the iso-PSNR framing of one row.

    The target is the LOWER of the two final PSNRs -- the highest quality both
    runs actually attain -- so a time-to-reach exists on both curves without
    extrapolation (a 4-GPU run ending 0.02 dB under the 1-GPU run would never
    "reach" the 1-GPU final). For the run that owns the minimum, t_reach is
    its full training time by construction.
    """
    tgt = min(a["final_psnr"], b["final_psnr"])
    return tgt, t_to_reach(a, tgt), t_to_reach(b, tgt)


# Per-dataset single-GPU training budget (s), matching the wall-clock budget
# the main results table gives the method. Used by --iso budget.
TRAIN_BUDGET = {"nyx_b": 10.0, "nyx_t": 10.0, "nyx_d": 10.0,
                "miranda": 80.0, "qmc": 60.0, "mag": 10.0}


def _iso_budget_row(a, b):
    """(target, t1_eff, t4_reach) for the budget-anchored framing.

    The target is the best-so-far PSNR the single-GPU run holds when its
    training budget expires, so the anchor is the same operating point the
    main results table reports, and it sits on the climbing part of the curve
    rather than its flat top. The single-GPU run must cover the budget (rerun
    it with more epochs if it does not -- interpolating a target is fine,
    extrapolating one is not); if it still falls short, the target degrades to
    its final and t1_eff to when that was reached.
    """
    T = TRAIN_BUDGET[a["dataset"]]
    tt = np.asarray(a["time"], dtype=float)
    pp = np.maximum.accumulate(np.asarray(a["psnr"], dtype=float))
    if T >= tt[-1]:
        tgt = float(pp[-1])
        return tgt, t_to_reach(a, tgt), t_to_reach(b, tgt)
    tgt = float(np.interp(T, tt, pp))
    return tgt, T, t_to_reach(b, tgt)


def table(paths, iso="psnr"):
    groups = _group(paths)
    iso_row = {"psnr": _iso_psnr_row, "budget": _iso_budget_row}.get(iso)
    if iso_row:
        print(f"\n{'dataset':<26}{'CR':>6}{'PSNR':>8}{'1GPU comp':>11}{'dec':>7}"
              f"{'4GPU comp':>11}{'dec':>7}{'speed-up':>10}{'stable?':>10}")
    else:
        print(f"\n{'dataset':<26}{'CR':>6}{'1GPU dB':>9}{'comp':>8}{'dec':>7}"
              f"{'4GPU dB':>9}{'comp':>8}{'dec':>7}{'train':>7}{'stable?':>10}")
    for codec, by in groups.items():
        print(f"\n-- {CODEC_LABEL[codec]} " + "-" * 84)
        for ds in DATASETS:
            if ds not in by or 1 not in by[ds]:
                continue
            a = by[ds][1]
            b = by[ds].get(4)
            # a single-GPU curve that dips makes its time-to-target optimistic
            wobble = float(np.max(np.maximum.accumulate(a["psnr"]) - np.asarray(a["psnr"])))
            flag = "ok" if wobble < 0.5 else f"WOBBLY {wobble:.1f}"
            if iso_row:
                if not b:
                    print(f"{a['label']:<26}{a['cr_eff']:>6.0f}{'-':>8}  (no 4-GPU run)")
                    continue
                tgt, t1, t4 = iso_row(a, b)
                if t4 is None:
                    print(f"{a['label']:<26}{a['cr_eff']:>6.0f}{tgt:>8.2f}"
                          f"{a['t_sz_compress']+t1:>10.2f}s{a['t_decompress']:>6.2f}s"
                          f"{'never':>11}{'-':>7}{'-':>9}{flag:>10}")
                    continue
                sp = t1 / t4
                mark = " >4x!" if sp > b["world_size"] else ""
                print(f"{a['label']:<26}{a['cr_eff']:>6.0f}{tgt:>8.2f}"
                      f"{a['t_sz_compress']+t1:>10.2f}s{a['t_decompress']:>6.2f}s"
                      f"{b['t_sz_compress']+t4:>10.2f}s{b['t_decompress']:>6.2f}s"
                      f"{sp:>8.2f}x{flag:>10}{mark}")
            else:
                row = (f"{a['label']:<26}{a['cr_eff']:>6.0f}{a['final_psnr']:>9.2f}"
                       f"{a['t_compress']:>7.2f}s{a['t_decompress']:>6.2f}s")
                if b:
                    row += (f"{b['final_psnr']:>9.2f}{b['t_compress']:>7.2f}s"
                            f"{b['t_decompress']:>6.2f}s{a['t_train']/b['t_train']:>6.2f}x")
                else:
                    row += f"{'-':>9}{'-':>8}{'-':>7}{'-':>7}"
                print(row + f"{flag:>10}")
    if iso == "budget":
        print(f"\niso-budget: the target is the 1-GPU best-so-far PSNR when its "
              f"per-dataset training budget expires\n(TRAIN_BUDGET, matching the main "
              f"results table); both columns report codec encode + training to that\n"
              f"target, and 'speed-up' is the training-time ratio.")
    elif iso == "psnr":
        print(f"\niso-PSNR: both columns report codec encode + training stopped the "
              f"moment the run reaches the row's\ncommon target quality (the lower of "
              f"the two 10-epoch finals); 'speed-up' is the training-time ratio.\n"
              f"'>4x!' exceeds the 4-GPU hardware bound: there the 1-GPU baseline "
              f"stalls (see WOBBLY / the sampler\nnote), so the ratio reflects the "
              f"baseline's optimization trajectory, not parallel efficiency.")
    else:
        print(f"\niso-work: compression = codec encode + 10-epoch training; 'train' is "
              f"the pure training speed-up at equal work.")
    print(f"decompression = codec decode + full-volume inference. WOBBLY marks a 1-GPU "
          f"curve that dips >0.5 dB\nbefore its final value. SZ3 is timed in-process "
          f"via pysz; SPERR through its CLI, so its codec share\nincludes process "
          f"start and file I/O. The codec blocks are each internally comparable.")


def latex(paths, iso="psnr"):
    """VLDB-style booktabs table on stdout."""
    groups = _group(paths)

    iso_row = {"psnr": _iso_psnr_row, "budget": _iso_budget_row}.get(iso)
    print(r"\begin{table}[t]")
    print(r"\centering")
    if iso == "budget":
        print(r"\caption{Wall-clock cost to reach equal quality at "
              r"$\mathrm{CR}\!\approx\!500$, for both base codecs. Each row's "
              r"target PSNR is what the single GPU attains within the training "
              r"budget the method is configured with (10\,s for the $512^3$ "
              r"fields, 60\,s for QMCPack, 80\,s for Miranda); compression is "
              r"codec encoding plus training to that target, decompression is "
              r"codec decoding plus one full-volume forward pass. Both "
              r"configurations perform the same number of slices per optimizer "
              r"update, so within a row they differ only in GPU count; the "
              r"speed-up is the ratio of training times. SZ3 is timed "
              r"in-process, SPERR through its command-line tool.}")
    elif iso == "psnr":
        print(r"\caption{Wall-clock cost to reach equal quality at "
              r"$\mathrm{CR}\!\approx\!500$, for both base codecs. Each row fixes a "
              r"target PSNR -- the highest quality both configurations reach within "
              r"the 10-epoch budget -- and compression is codec encoding plus training "
              r"stopped the moment that target is met; decompression is codec decoding "
              r"plus one full-volume forward pass. Both configurations perform the "
              r"same number of slices per optimizer update, so within a row they "
              r"differ only in GPU count; the speed-up is the ratio of training times. "
              r"$\dagger$: exceeds the 4-GPU bound because the single-GPU baseline "
              r"stalls mid-run (Section~\ref{sec:eval-parallel}), so the ratio "
              r"includes the baseline's optimization trajectory, not hardware alone. "
              r"SZ3 is timed in-process, SPERR through its command-line tool.}")
    else:
        print(r"\caption{Compression and decompression cost at "
              r"$\mathrm{CR}\!\approx\!500$, for both base codecs. Compression is "
              r"codec encoding plus residual-model training (10 epochs); decompression "
              r"is codec decoding plus one full-volume forward pass. The two "
              r"configurations perform the same number of slices per optimizer update "
              r"and the same number of updates, so they differ only in GPU count, and "
              r"the speed-up is over training alone. SZ3 is timed in-process, SPERR "
              r"through its command-line tool.}")
    print(r"\label{tab:compress-time}")

    if iso_row:
        ncol = 7
        print(r"\begin{tabular}{l" + "r" * (ncol - 1) + "}")
        print(r"\toprule")
        print(r"& & \multicolumn{2}{c}{Single GPU} & \multicolumn{2}{c}{4 GPUs} & \\")
        print(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
        print(r"Dataset & PSNR & Comp. & Dec. & Comp. & Dec. & Speed- \\")
        print(r"        & (dB) & (s)   & (s)  & (s)   & (s)  & up     \\")
        for codec, by in groups.items():
            print(r"\midrule")
            print(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (ncol, CODEC_LABEL[codec]))
            for ds in DATASETS:
                if ds not in by or 1 not in by[ds]:
                    continue
                a, b = by[ds][1], by[ds].get(4)
                cells = [a["label"].replace("—", "--")]
                if b and (tgt_t := iso_row(a, b))[2] is not None:
                    tgt, t1, t4 = tgt_t
                    sp = t1 / t4
                    dag = r"$^\dagger$" if sp > b["world_size"] else ""
                    cells += [f"{tgt:.2f}",
                              f"{a['t_sz_compress'] + t1:.2f}", f"{a['t_decompress']:.2f}",
                              f"{b['t_sz_compress'] + t4:.2f}", f"{b['t_decompress']:.2f}",
                              f"{sp:.2f}$\\times${dag}"]
                else:
                    cells += ["--"] * 6
                print(" & ".join(cells) + r" \\")
    else:
        ncol = 8
        print(r"\begin{tabular}{l" + "r" * (ncol - 1) + "}")
        print(r"\toprule")
        print(r"& \multicolumn{3}{c}{Single GPU} & \multicolumn{3}{c}{4 GPUs} & \\")
        print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
        print(r"Dataset & PSNR & Comp. & Dec. & PSNR & Comp. & Dec. & Train \\")
        print(r"        & (dB) & (s)   & (s)  & (dB) & (s)   & (s)  & speed-up \\")
        for codec, by in groups.items():
            print(r"\midrule")
            print(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (ncol, CODEC_LABEL[codec]))
            for ds in DATASETS:
                if ds not in by or 1 not in by[ds]:
                    continue
                a = by[ds][1]
                cells = [a["label"].replace("—", "--"), f"{a['final_psnr']:.2f}",
                         f"{a['t_compress']:.2f}", f"{a['t_decompress']:.2f}"]
                b = by[ds].get(4)
                cells += ([f"{b['final_psnr']:.2f}", f"{b['t_compress']:.2f}",
                           f"{b['t_decompress']:.2f}",
                           f"{a['t_train'] / b['t_train']:.2f}$\\times$"]
                          if b else ["--", "--", "--", "--"])
                print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="nyx_b")
    ap.add_argument("--codec", choices=("sz3", "sperr"), default="sz3",
                    help="base codec the residual model corrects")
    ap.add_argument("--cr", type=float, default=500.0)
    ap.add_argument("--aux-mode", dest="aux_mode", default="cr_matched",
                    choices=["cr_matched", "enhanced", "originals"],
                    help="how NYX sibling channels are obtained; cr_matched is "
                         "the paper protocol, originals is NOT reproducible")
    ap.add_argument("--aux-enhanced-dir", dest="aux_enhanced_dir", default="",
                    help="cascade export dir read when --aux-mode enhanced")
    ap.add_argument("--export-enhanced-dir", dest="export_enhanced_dir", default="",
                    help="write this run's reconstruction here so a later "
                         "cascade stage can use it as an enhanced sibling")
    ap.add_argument("--rel", type=float, default=0.0,
                    help="--codec sz3: skip the CR bisection and use this rel_err")
    ap.add_argument("--sperr_q", type=float, default=0.0,
                    help="--codec sperr: skip the CR bisection and use this --psnr target")
    ap.add_argument("--sperr_omp", type=int, default=1,
                    help="sperr3d OpenMP threads; fixed so 1-GPU and 4-GPU jobs "
                         "measure the codec at the same width")
    ap.add_argument("--tmp", default="/tmp",
                    help="node-local scratch for sperr3d bitstreams and decoded volumes")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--iso-time", dest="iso_time", type=int, default=0,
                    help="train for the dataset's paper wall-clock budget "
                         "(spec train_s) instead of a fixed epoch count")
    ap.add_argument("--train-s", dest="train_s", type=float, default=0.0,
                    help="explicit pure-training budget in seconds; overrides "
                         "--iso-time and the per-dataset default")
    ap.add_argument("--epoch-cap", dest="epoch_cap", type=int, default=200,
                    help="epoch ceiling when a wall-clock budget is in force")
    ap.add_argument("--localsgd-every", dest="localsgd_every", type=int, default=0,
                    help="multi-GPU only: average weights every K steps instead "
                         "of DDP gradient averaging (0 = plain DDP)")
    ap.add_argument("--tot-batch", dest="tot_batch", type=int, default=0,
                    help="explicit TOTAL batch across ranks; overrides the "
                         "dataset spec (weak-scaling runs)")
    ap.add_argument("--batch", type=int, default=4, help="single-GPU batch (iso-batch)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_mult", type=float, default=8.0,
                    help="only used when --bo is off and no --axis given")
    ap.add_argument("--bo", type=int, default=1,
                    help="run the Phase-1 (axis, lr) TPE search")
    ap.add_argument("--axis", type=int, default=-1,
                    help="skip Phase 1 and use this axis (with --lr_abs)")
    ap.add_argument("--lr_abs", type=float, default=1e-3,
                    help="absolute lr to pair with --axis")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="bench_out/table")
    ap.add_argument("--table", nargs="*", default=None)
    ap.add_argument("--latex", nargs="*", default=None)
    ap.add_argument("--iso", choices=("budget", "psnr", "work"), default="budget",
                    help="table framing: 'budget' anchors the target to the 1-GPU "
                         "training budget (TRAIN_BUDGET); 'psnr' to the lower of "
                         "the two finals; 'work' reports full 10 epochs")
    a = ap.parse_args()
    if a.latex:
        latex(a.latex, a.iso)
    elif a.table:
        table(a.table, a.iso)
    else:
        run(a)
