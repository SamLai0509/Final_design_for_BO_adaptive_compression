"""FAST-style end-to-end breakdown at CR ~= 300 for NYX baryon density (512^3, 512 MB).

Reports, for each pipeline, the wall time of every stage a producer/consumer actually
pays: compress, write to SSD, read from SSD, decompress, plus (for the learned methods)
train and inference. I/O goes to the NVMe SSD (/home/sam, nvme0n1p2); every write is
fsync'ed and every file is evicted from the page cache with posix_fadvise(DONTNEED)
before it is read back, so the read column is real device I/O, not RAM.

Pipelines
  raw            : write/read the original 512 MB f32 (the no-compression baseline)
  SZ3            : compress -> write stream -> read stream -> decompress
  SZ3 + NeurLZ   : SZ3 + a 100-epoch NeurLZ model (weights stored fp32, as shipped)
  SZ3 + Ours     : SZ3 + a 1-epoch AdaMit model (weights stored bf16, as in the paper)
Learned methods store the model next to the stream; both are written/read together and
the effective CR counts both.
"""
import os, sys, time, io, contextlib, random, json
import numpy as np
import torch

sys.path.append("/home/sam/Halo_Finder/Final_design/base_script")
sys.path.append("/home/sam/Data_Compression/SZ3/tools/pysz")
from pysz import SZ
from experiment import build_bg_only_cfg
from bg_stage import train_bg_only, run_bg_inference, unwrap_bg_model
from bg_shard import pick_bg_h_under_budget
from monai.networks.nets import BasicUNet

SSD_DIR = "/home/sam/fast_io_bench_tmp"      # on nvme0n1p2 (ROTA=0)
NB      = "/home/sam/Halo_Finder/halo_finder_v1/SDRBENCH-EXASKY-NYX-512x512x512/origin/"
SHAPE   = (512, 512, 512)
TARGET_CR = 300.0
FIELDS  = ["baryon_density.f32", "temperature.f32", "dark_matter_density.f32",
           "velocity_z.f32", "velocity_x.f32", "velocity_y.f32"]
SEED, REPS = 17, 3
dev = torch.device("cuda:0")
os.makedirs(SSD_DIR, exist_ok=True)

def set_seed(s=SEED):
    torch.manual_seed(s); np.random.seed(s); random.seed(s); torch.cuda.manual_seed_all(s)

def psnr(a, b):
    a = a.astype(np.float64); dr = float(a.max() - a.min())
    return 20*np.log10(dr) - 10*np.log10(float(np.mean((a - b.astype(np.float64))**2)))

# ── I/O primitives: fsync'ed write, cache-evicted read ────────────────────────
def write_bytes(path, buf):
    t0 = time.perf_counter()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    with os.fdopen(fd, "wb", closefd=False) as f:
        f.write(buf); f.flush()
    os.fsync(fd); os.close(fd)
    return time.perf_counter() - t0

def evict(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)

def read_bytes(path):
    evict(path)
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        buf = f.read()
    return time.perf_counter() - t0, buf

def med(f, reps=REPS):
    return float(np.median([f() for _ in range(reps)]))

# ── data + SZ3 operating point ────────────────────────────────────────────────
print("loading NYX ...", flush=True)
vols = [np.fromfile(NB + f, np.float32).reshape(SHAPE) for f in FIELDS]
gt = vols[0]; orig_bytes = gt.nbytes
sz = SZ("/home/sam/Data_Compression/SZ3/build/lib64/libSZ3c.so")

lo, hi = 1e-7, 1e-3
for _ in range(18):
    mid = (lo*hi)**0.5
    b, _ = sz.compress(gt, 1, 0, float(mid), 0)
    cr = orig_bytes/len(b)
    if cr < TARGET_CR: lo = mid
    else: hi = mid
REL = (lo*hi)**0.5
b, _ = sz.compress(gt, 1, 0, REL, 0); sz_bytes = len(b); del b
print(f"SZ3 rel={REL:.4e} -> {sz_bytes/1e6:.2f} MB, CR {orig_bytes/sz_bytes:.1f}", flush=True)

rows = []

# ── 0) raw: no compression ────────────────────────────────────────────────────
raw_path = os.path.join(SSD_DIR, "raw.f32")
raw_buf = gt.tobytes()
w = med(lambda: write_bytes(raw_path, raw_buf))
r = med(lambda: read_bytes(raw_path)[0])
rows.append(dict(name="Raw (no compression)", comp=0.0, write=w, read=r, decomp=0.0,
                 train=0.0, infer=0.0, bytes=orig_bytes, cr=1.0, psnr=float("inf")))
del raw_buf
print(f"raw: write {w:.2f}s read {r:.2f}s", flush=True)

# ── 1) SZ3 ────────────────────────────────────────────────────────────────────
def sz_compress():
    t0 = time.perf_counter(); bb, _ = sz.compress(gt, 1, 0, REL, 0); dt = time.perf_counter()-t0
    return dt, bb
c_times, bb = [], None
for _ in range(REPS):
    dt, bb = sz_compress(); c_times.append(dt)
sz_arr = np.asarray(bb)          # pysz needs the ndarray for decompress
sz_buf = sz_arr.tobytes()        # ... and raw bytes for the file I/O
del bb
stream_path = os.path.join(SSD_DIR, "sz3.bin")
w = med(lambda: write_bytes(stream_path, sz_buf))
r = med(lambda: read_bytes(stream_path)[0])
d_times = []
for _ in range(REPS):
    t0 = time.perf_counter(); lq = sz.decompress(sz_arr, SHAPE, np.float32); d_times.append(time.perf_counter()-t0)
lq = np.ascontiguousarray(lq, np.float32)
rows.append(dict(name="SZ3", comp=float(np.median(c_times)), write=w, read=r,
                 decomp=float(np.median(d_times)), train=0.0, infer=0.0,
                 bytes=sz_bytes, cr=orig_bytes/sz_bytes, psnr=psnr(gt, lq)))
print(f"SZ3: comp {np.median(c_times):.2f}s write {w:.2f}s read {r:.2f}s decomp {np.median(d_times):.2f}s", flush=True)

# ── 2) SZ3 + NeurLZ, 100 epochs (recipe of SPERR_fft.py's run_neurlz) ─────────
def mm(x, eps=1e-8):
    loo, hii = float(x.min()), float(x.max())
    return ((x - loo)/(hii - loo + eps)).astype(np.float32), (loo, hii - loo + eps)
EPOCHS_NLZ, BATCH_NLZ = 100, 10
fields = [lq] + vols[1:]
lq_n = np.stack([mm(f)[0] for f in fields], axis=1)
err_n, (e_off, e_scale) = mm(gt - lq)
X = torch.from_numpy(lq_n); Y = torch.from_numpy(err_n[:, None]); del lq_n, err_n
set_seed()
with contextlib.redirect_stdout(io.StringIO()):
    nlz = BasicUNet(spatial_dims=2, features=(4,)*6, act="gelu", in_channels=6, out_channels=1).to(dev)
nlz_params = sum(p.numel() for p in nlz.parameters())
opt = torch.optim.Adam(nlz.parameters(), lr=1e-2)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
mse = torch.nn.MSELoss(); idx = np.arange(SHAPE[0]); nlz.train()
torch.cuda.synchronize(); t0 = time.perf_counter()
for ep in range(EPOCHS_NLZ):
    np.random.shuffle(idx)
    for st in range(0, SHAPE[0], BATCH_NLZ):
        bi = idx[st:st+BATCH_NLZ]
        loss = mse(nlz(X[bi].to(dev)), Y[bi].to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    if (ep+1) % 20 == 0:
        torch.cuda.synchronize(); print(f"  [neurlz] {ep+1}/{EPOCHS_NLZ} ep, {time.perf_counter()-t0:.0f}s", flush=True)
torch.cuda.synchronize(); nlz_train = time.perf_counter() - t0
nlz_path = os.path.join(SSD_DIR, "neurlz.pt")
torch.save(nlz.state_dict(), nlz_path); nlz_bytes = os.path.getsize(nlz_path)
nlz_blob = open(nlz_path, "rb").read()
w = med(lambda: write_bytes(stream_path, sz_buf) + write_bytes(nlz_path, nlz_blob))
r = med(lambda: read_bytes(stream_path)[0] + read_bytes(nlz_path)[0])
nlz.eval(); out = lq.copy()
torch.cuda.synchronize(); t0 = time.perf_counter()
with torch.no_grad():
    for st in range(0, SHAPE[0], BATCH_NLZ):
        bi = list(range(st, min(st+BATCH_NLZ, SHAPE[0])))
        pr = nlz(X[bi].to(dev)).cpu().numpy()[:, 0]
        out[bi] = lq[bi] + (pr*e_scale + e_off)
torch.cuda.synchronize(); nlz_infer = time.perf_counter() - t0
rows.append(dict(name=f"SZ3 + NeurLZ ({EPOCHS_NLZ} ep)", comp=float(np.median(c_times)), write=w, read=r,
                 decomp=float(np.median(d_times)), train=nlz_train, infer=nlz_infer,
                 bytes=sz_bytes+nlz_bytes, cr=orig_bytes/(sz_bytes+nlz_bytes), psnr=psnr(gt, out),
                 params=nlz_params, model_bytes=nlz_bytes))
print(f"NeurLZ: train {nlz_train:.1f}s infer {nlz_infer:.2f}s psnr {psnr(gt,out):.2f} params {nlz_params:,} model {nlz_bytes/1e3:.1f}KB", flush=True)
del nlz, X, Y, out; torch.cuda.empty_cache()

# ── 3) SZ3 + Ours, 1 epoch ───────────────────────────────────────────────────
with contextlib.redirect_stdout(io.StringIO()):
    h, npar = pick_bg_h_under_budget(30000, shape=SHAPE, n_fields=6, bg_arch="spatial",
                                     h_candidates=list(range(3, 256)))
Xs = [gt] + vols[1:]; Xps = [lq] + vols[1:]
cfg = build_bg_only_cfg(X_target=gt, Xps=Xps, max_train_time=1e9, bg_h=int(h), roi_h=4,
                        epochs=1, steps_per_epoch=SHAPE[0], bg_patch_size=SHAPE[2], bg_batch=1,
                        lr=1e-3, bg_freq_weight=0.5, bg_fft_phase_weight=1.0,
                        bg_freq_warmup_epochs=1, bg_field_norm="zscore")
cfg.bg_arch="spatial"; cfg.bg_split_mode="three"; cfg.bg_split_bands=True
cfg.bg_split_sigma=0.12; cfg.bg_sigma_low=0.08; cfg.bg_sigma_mid=0.18
cfg.bg_low_weight=0.2; cfg.bg_mid_weight=0.5; cfg.bg_high_weight=1.0
cfg.bg_cr_rel_err=float(REL); cfg.bg_gpu_sampling=True; cfg.seed=SEED
cfg.bg_sample_mode="random"; cfg.bg_early_stop=False
set_seed()
torch.cuda.synchronize(); t0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    model, _hist = train_bg_only(Xs=Xs, Xps=Xps, device=dev, cfg=cfg, evaluator=None)
torch.cuda.synchronize(); ours_train = time.perf_counter() - t0
core = unwrap_bg_model(model)
ours_path = os.path.join(SSD_DIR, "adamit.pt")
torch.save({k: v.to(torch.bfloat16) for k, v in core.state_dict().items()}, ours_path)
ours_bytes = os.path.getsize(ours_path); ours_blob = open(ours_path, "rb").read()
w = med(lambda: write_bytes(stream_path, sz_buf) + write_bytes(ours_path, ours_blob))
r = med(lambda: read_bytes(stream_path)[0] + read_bytes(ours_path)[0])
torch.cuda.synchronize(); t0 = time.perf_counter()
xh = run_bg_inference(core, Xs, Xps, cfg, float(REL))
torch.cuda.synchronize(); ours_infer = time.perf_counter() - t0
rows.append(dict(name="SZ3 + Ours (1 ep)", comp=float(np.median(c_times)), write=w, read=r,
                 decomp=float(np.median(d_times)), train=ours_train, infer=ours_infer,
                 bytes=sz_bytes+ours_bytes, cr=orig_bytes/(sz_bytes+ours_bytes), psnr=psnr(gt, xh),
                 params=int(npar), model_bytes=ours_bytes))
print(f"Ours: train {ours_train:.1f}s infer {ours_infer:.2f}s psnr {psnr(gt,xh):.2f} params {npar:,} model {ours_bytes/1e3:.1f}KB", flush=True)

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*118)
print(f"NYX baryon density {SHAPE}, {orig_bytes/1e6:.0f} MB, SZ3 rel={REL:.3e}, NVMe SSD, {REPS}-run median for I/O")
print(f"{'pipeline':24s} {'comp':>7s} {'write':>7s} {'read':>7s} {'decomp':>7s} {'train':>8s} {'infer':>7s} | "
      f"{'W-path':>8s} {'R-path':>8s} {'MB':>7s} {'CR':>7s} {'PSNR':>7s}")
print("="*118)
for r_ in rows:
    wpath = r_["comp"] + r_["write"] + r_["train"]
    rpath = r_["read"] + r_["decomp"] + r_["infer"]
    print(f"{r_['name']:24s} {r_['comp']:7.2f} {r_['write']:7.2f} {r_['read']:7.2f} {r_['decomp']:7.2f} "
          f"{r_['train']:8.1f} {r_['infer']:7.2f} | {wpath:8.2f} {rpath:8.2f} {r_['bytes']/1e6:7.2f} "
          f"{r_['cr']:7.1f} {r_['psnr']:7.2f}")
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fast_io_bench.json"), "w"), indent=1)
print("\nsaved fast_io_bench.json")
