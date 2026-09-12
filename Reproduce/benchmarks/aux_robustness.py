"""Does inference survive DECOMPRESSED auxiliary inputs?

Train exactly as the paper-final pipeline does on NYX baryon density at the CR~500
band (SZ3 rel=1.1708e-5; Phase-2 config = the one Phase 1 picked in the final run:
axis Y, lr 9.3e-3, 9 s budget, sequential, lambda_phase=1), with ORIGINAL aux fields.
Then run inference three times with the same weights:
  (a) aux = original sibling fields (the paper's protocol)
  (b) aux = each sibling field SZ3-compressed at rel 1e-5 and decompressed (tight)
  (c) aux = same at rel 1e-3 (loose -- a deliberately pessimistic archive setting)
GPU: cuda:1 (5090); the six-field chain owns cuda:0."""
import os, sys, time, io, contextlib, random
import numpy as np, torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "base_script"))
from local_paths import P
sys.path.append(P("ADAMIT_PYSZ"))
from pysz import SZ
from experiment import build_bg_only_cfg
from bg_stage import train_bg_only, run_bg_inference, unwrap_bg_model
from bg_shard import pick_bg_h_under_budget

NB=P("ADAMIT_NYX_DIR")
FIELDS=["baryon_density.f32","temperature.f32","dark_matter_density.f32",
        "velocity_z.f32","velocity_x.f32","velocity_y.f32"]
SHAPE=(512,512,512); REL=1.1708e-5; LR=9.3e-3; AXIS=1; BUDGET=9.0; SEED=17
dev=torch.device("cuda:1")
print("device:", torch.cuda.get_device_name(1), flush=True)

def set_seed(s=SEED):
    torch.manual_seed(s); np.random.seed(s); random.seed(s); torch.cuda.manual_seed_all(s)
def psnr(a,b):
    a=a.astype(np.float64); dr=float(a.max()-a.min())
    return 20*np.log10(dr)-10*np.log10(float(np.mean((a-b.astype(np.float64))**2)))

vols=[np.fromfile(NB+f,np.float32).reshape(SHAPE) for f in FIELDS]
gt=vols[0]; aux=vols[1:]
sz=SZ(P("ADAMIT_SZ3_LIB"))
b,cr=sz.compress(gt,1,0,REL,0); lq=np.ascontiguousarray(sz.decompress(b,SHAPE,np.float32),np.float32); del b
print(f"target: SZ3 rel={REL:.3e} CR={cr:.0f} base PSNR {psnr(gt,lq):.2f}", flush=True)

def dec_aux(rel):
    out=[]
    for i,a in enumerate(aux):
        bb,_=sz.compress(a,1,0,float(rel),0)
        d=np.ascontiguousarray(sz.decompress(bb,SHAPE,np.float32),np.float32); del bb
        out.append(d)
        print(f"  aux[{i}] {FIELDS[i+1]:24s} rel={rel:.0e} PSNR {psnr(a,d):.1f} dB", flush=True)
    return out
aux_tight=dec_aux(1e-5); aux_loose=dec_aux(1e-3)

FWD={0:(0,1,2),1:(1,0,2),2:(2,0,1)}; INV={0:(0,1,2),1:(1,0,2),2:(1,2,0)}
P=lambda x: np.ascontiguousarray(np.transpose(x,FWD[AXIS]))
Xs=[P(f) for f in vols]; Xps=[P(lq)]+[P(a) for a in aux]
with contextlib.redirect_stdout(io.StringIO()):
    h,npar=pick_bg_h_under_budget(30000, shape=SHAPE, n_fields=6, bg_arch="spatial",
                                  h_candidates=list(range(3,256)))
cfg=build_bg_only_cfg(X_target=Xs[0], Xps=Xps, max_train_time=BUDGET, bg_h=int(h), roi_h=4,
                      epochs=100000, steps_per_epoch=SHAPE[0], bg_patch_size=SHAPE[2], bg_batch=1,
                      lr=LR, bg_freq_weight=0.5, bg_fft_phase_weight=1.0, bg_freq_warmup_epochs=1,
                      bg_field_norm="zscore")
cfg.bg_arch="spatial"; cfg.bg_split_mode="three"; cfg.bg_split_bands=True
cfg.bg_split_sigma=0.12; cfg.bg_sigma_low=0.08; cfg.bg_sigma_mid=0.18
cfg.bg_low_weight=0.2; cfg.bg_mid_weight=0.5; cfg.bg_high_weight=1.0
cfg.bg_cr_rel_err=float(REL); cfg.bg_gpu_sampling=True; cfg.seed=SEED
cfg.bg_sample_mode="sequential"; cfg.bg_early_stop=False; cfg.bg_sched_time_calibrate=True
set_seed()
with contextlib.redirect_stdout(io.StringIO()):
    model,_=train_bg_only(Xs=Xs, Xps=Xps, device=dev, cfg=cfg, evaluator=None)
core=unwrap_bg_model(model)

def infer(aux_set, tag):
    Xp=[P(lq)]+[P(a) for a in aux_set]
    xh=run_bg_inference(core, Xs, Xp, cfg, float(REL))
    p=psnr(gt, np.transpose(xh, INV[AXIS]))
    print(f"  [{tag:28s}] PSNR {p:.3f} dB", flush=True)
    return p
print("inference with the SAME trained weights:", flush=True)
p0=infer(aux, "original aux (paper)")
p1=infer(aux_tight, "decompressed aux, rel 1e-5")
p2=infer(aux_loose, "decompressed aux, rel 1e-3")
print(f"\ndelta tight: {p1-p0:+.3f} dB | delta loose: {p2-p0:+.3f} dB | base {psnr(gt,lq):.2f}", flush=True)
