"""Effective CR = 300 exactly (SZ3 stream + bf16 model), RTX PRO 6000:
SZ3 compress/decompress time, and AdaMit's 1-epoch train + full-volume inference time.
Every stage is repeated and reported as the median; NYX baryon density 512^3."""
import os, sys, time, io, contextlib, random
import numpy as np, torch

sys.path.append("/home/sam/Halo_Finder/Final_design/base_script")
sys.path.append("/home/sam/Data_Compression/SZ3/tools/pysz")
from pysz import SZ
from experiment import build_bg_only_cfg
from bg_stage import train_bg_only, run_bg_inference, unwrap_bg_model
from bg_shard import pick_bg_h_under_budget

NB="/home/sam/Halo_Finder/halo_finder_v1/SDRBENCH-EXASKY-NYX-512x512x512/origin/"
FIELDS=["baryon_density.f32","temperature.f32","dark_matter_density.f32",
        "velocity_z.f32","velocity_x.f32","velocity_y.f32"]
SHAPE=(512,512,512); TARGET_EFF_CR=300.0; PARAM_BUDGET=30000; BYTES_PER_PARAM=2
SEED, REPS = 17, 3
dev=torch.device("cuda:0")
print("device:", torch.cuda.get_device_name(0), flush=True)

def set_seed(s=SEED):
    torch.manual_seed(s); np.random.seed(s); random.seed(s); torch.cuda.manual_seed_all(s)
def psnr(a,b):
    a=a.astype(np.float64); dr=float(a.max()-a.min())
    return 20*np.log10(dr)-10*np.log10(float(np.mean((a-b.astype(np.float64))**2)))

vols=[np.fromfile(NB+f,np.float32).reshape(SHAPE) for f in FIELDS]
gt=vols[0]; orig=gt.nbytes
sz=SZ("/home/sam/Data_Compression/SZ3/build/lib64/libSZ3c.so")
with contextlib.redirect_stdout(io.StringIO()):
    h,npar=pick_bg_h_under_budget(PARAM_BUDGET, shape=SHAPE, n_fields=6, bg_arch="spatial",
                                  h_candidates=list(range(3,256)))
model_bytes=int(npar)*BYTES_PER_PARAM
print(f"model: bg_h={h}  {npar:,} params  {model_bytes/1e3:.1f} KB (bf16)", flush=True)

# SZ3 rel such that orig/(stream + model) == 300
lo,hi=1e-7,1e-3
for _ in range(20):
    mid=(lo*hi)**0.5
    b,_=sz.compress(gt,1,0,float(mid),0)
    eff=orig/(len(b)+model_bytes)
    if eff<TARGET_EFF_CR: lo=mid
    else: hi=mid
REL=(lo*hi)**0.5
b,_=sz.compress(gt,1,0,REL,0); sz_bytes=len(b); arr=np.asarray(b); del b
print(f"SZ3 rel={REL:.4e} -> stream {sz_bytes/1e6:.3f} MB | effective CR "
      f"{orig/(sz_bytes+model_bytes):.1f} (SZ3 alone {orig/sz_bytes:.1f})", flush=True)

ct=[]
for _ in range(REPS):
    t0=time.perf_counter(); bb,_=sz.compress(gt,1,0,REL,0); ct.append(time.perf_counter()-t0); arr=np.asarray(bb)
dt=[]
for _ in range(REPS):
    t0=time.perf_counter(); lq=sz.decompress(arr,SHAPE,np.float32); dt.append(time.perf_counter()-t0)
lq=np.ascontiguousarray(lq,np.float32)
print(f"SZ3 compress {np.median(ct):.2f}s | decompress {np.median(dt):.2f}s | base PSNR {psnr(gt,lq):.2f} dB", flush=True)

Xs=[gt]+vols[1:]; Xps=[lq]+vols[1:]
def mk_cfg():
    cfg=build_bg_only_cfg(X_target=gt, Xps=Xps, max_train_time=1e9, bg_h=int(h), roi_h=4, epochs=1,
                          steps_per_epoch=SHAPE[0], bg_patch_size=SHAPE[2], bg_batch=1, lr=1e-3,
                          bg_freq_weight=0.5, bg_fft_phase_weight=1.0, bg_freq_warmup_epochs=1,
                          bg_field_norm="zscore")
    cfg.bg_arch="spatial"; cfg.bg_split_mode="three"; cfg.bg_split_bands=True
    cfg.bg_split_sigma=0.12; cfg.bg_sigma_low=0.08; cfg.bg_sigma_mid=0.18
    cfg.bg_low_weight=0.2; cfg.bg_mid_weight=0.5; cfg.bg_high_weight=1.0
    cfg.bg_cr_rel_err=float(REL); cfg.bg_gpu_sampling=True; cfg.seed=SEED
    cfg.bg_sample_mode="random"; cfg.bg_early_stop=False
    return cfg

tr, inf, ps = [], [], []
for rep in range(REPS+1):                       # rep 0 = warm-up (cuDNN autotune), discarded
    cfg=mk_cfg(); set_seed()
    torch.cuda.synchronize(); t0=time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model,_=train_bg_only(Xs=Xs, Xps=Xps, device=dev, cfg=cfg, evaluator=None)
    torch.cuda.synchronize(); t_tr=time.perf_counter()-t0
    core=unwrap_bg_model(model)
    torch.cuda.synchronize(); t0=time.perf_counter()
    xh=run_bg_inference(core, Xs, Xps, cfg, float(REL))
    torch.cuda.synchronize(); t_in=time.perf_counter()-t0
    if rep: tr.append(t_tr); inf.append(t_in); ps.append(psnr(gt,xh))
    print(f"  rep{rep}{' (warm-up)' if rep==0 else ''}: train {t_tr:.2f}s  inference {t_in:.2f}s  PSNR {psnr(gt,xh):.2f}", flush=True)
    del model, core, xh; torch.cuda.empty_cache()

print("\n"+"="*78)
print(f"NYX baryon density 512^3 ({orig/1e6:.0f} MB) | effective CR {orig/(sz_bytes+model_bytes):.1f} | {torch.cuda.get_device_name(0)}")
print("="*78)
print(f"  SZ3 compress          {np.median(ct):6.2f} s")
print(f"  SZ3 decompress        {np.median(dt):6.2f} s")
print(f"  AdaMit train (1 ep)   {np.median(tr):6.2f} s   (runs: {['%.2f'%v for v in tr]})")
print(f"  AdaMit inference      {np.median(inf):6.2f} s   (runs: {['%.2f'%v for v in inf]})")
print(f"  ---------------------------------")
print(f"  PSNR  SZ3 {psnr(gt,lq):.2f} dB -> AdaMit {np.median(ps):.2f} dB  (+{np.median(ps)-psnr(gt,lq):.2f} dB)")
print(f"  compress+train = {np.median(ct)+np.median(tr):.2f} s | decompress+inference = {np.median(dt)+np.median(inf):.2f} s")
