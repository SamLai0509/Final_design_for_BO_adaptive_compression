"""Read-path inference timing, 1 vs 4 GPUs, for every dataset in the table.

Inference is slice-independent, so the volume splits into contiguous z-slabs
(run_bg_inference takes z_start/z_stop); each rank corrects its own slab and
the result is bit-identical to the single-GPU pass. Timing depends only on the
model architecture and the volume shape -- not on the trained weights -- so a
randomly initialised model of the table's exact size is timed instead of
retraining. Median of 3, pure compute, synchronized across ranks.

    srun --gres=gpu:1 python bench_infer_mgpu.py --dataset nyx_b
    srun --gres=gpu:4 python -m torch.distributed.run --nproc_per_node=4 \
        bench_infer_mgpu.py --dataset nyx_b
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT, "base_script"))
from experiment import build_bg_only_cfg
from bg_stage import run_bg_inference
from bg_shard import pick_bg_h_under_budget
from bench_compress_table import DATASETS, NYX_DIR, load_target

NYX_ALL = ["baryon_density", "dark_matter_density", "temperature",
           "velocity_x", "velocity_y", "velocity_z"]

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()
spec = DATASETS[args.dataset]

world = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
local = int(os.environ.get("LOCAL_RANK", 0))
if world > 1:
    import torch.distributed as dist
    dist.init_process_group("nccl")
torch.cuda.set_device(local)
device = torch.device(f"cuda:{local}")

gt = load_target(spec)
if spec.get("nyx"):
    aux = [np.fromfile(NYX_DIR + f + ".f32", np.float32).reshape(spec["shape"])
           for f in NYX_ALL if f != spec["stem"]]
else:
    aux = []
Xs = [gt] + aux
Xps = [gt] + aux              # weights are random; content does not matter
REL = 1e-5

bg_h, n_params = pick_bg_h_under_budget(spec["budget"], shape=gt.shape,
                                        n_fields=len(Xs), bg_arch="spatial",
                                        h_candidates=tuple(range(3, 256)))
torch.manual_seed(17); np.random.seed(17)
cfg = build_bg_only_cfg(X_target=gt, Xps=Xps, max_train_time=1e9, bg_h=int(bg_h),
                        roi_h=4, epochs=1, steps_per_epoch=1, bg_patch_size=512,
                        bg_batch=1, lr=1e-3, bg_field_norm="zscore")
cfg.bg_arch = "spatial"; cfg.bg_split_mode = "three"; cfg.bg_split_bands = True
cfg.bg_split_sigma = 0.12; cfg.bg_sigma_low = 0.08; cfg.bg_sigma_mid = 0.18
cfg.bg_cr_rel_err = REL; cfg.rel_err = REL
cfg.amp = True; cfg.amp_dtype = "bf16"

from siren_fft_backbone_model import UNET_Model  # noqa: E402
D, H, W = gt.shape
model = UNET_Model(n_fields=len(Xs), K=7, D=D, H=H, W=W, bg_hidden=cfg.bg_h,
                   bg_arch="spatial", bg_split_bands=True,
                   bg_split_mode="three").to(device).eval()

depth = gt.shape[0]
z0 = rank * depth // world
z1 = (rank + 1) * depth // world

def one_pass():
    if world > 1:
        dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_bg_inference(model, Xs, Xps, cfg, REL, z_start=z0, z_stop=z1)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if world > 1:
        t = torch.tensor([dt], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        dt = float(t.item())
    return dt

one_pass()                                     # warmup
ts = sorted(one_pass() for _ in range(args.reps))
if rank == 0:
    print(f"[infer] {args.dataset} params={int(n_params):,} world={world} "
          f"median={ts[len(ts)//2]:.3f}s  all={[round(t,3) for t in ts]}")
