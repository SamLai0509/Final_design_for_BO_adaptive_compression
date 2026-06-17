import random
import sys
import time
import os
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.distributed as dist

sys.path.append("/home/sam/Halo_Finder/Final_design/base_script")

from config_io import load_multifield_from_disk
from experiment import build_bg_only_cfg, estimate_bg_model_param_bytes
from bg_stage import run_bg_inference, train_bg_only, unwrap_bg_model
from bg_shard import (
    z_quad_shard_bounds,
    build_shard_plan,
    crop_multifield_zyx,
    infer_shard_ensemble_blend,
    pick_bg_h_under_budget,
    train_shards_parallel,
    reload_shard_model,
    _build_shard_cfg,
)

pysz_path = r"/home/sam/Data_Compression/SZ3/tools/pysz"
if pysz_path not in sys.path:
    sys.path.append(pysz_path)
from pysz import SZ

def set_seed(seed=17):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# import ctypes
# libfabric_path = '/opt/cray/libfabric/2.2.0rc1/lib64/libfabric.so.1'
# try:
#     ctypes.CDLL(libfabric_path, mode=ctypes.RTLD_GLOBAL)
#     if int(os.environ.get("LOCAL_RANK", 0)) == 0:
#         print(f"✅ 成功从 {libfabric_path} 强制加载 libfabric")
# except Exception as e:
#     if int(os.environ.get("LOCAL_RANK", 0)) == 0:
#         print(f"❌ 强制加载失败: {e}")

NUM_GPUS = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print(f"cuda devices: {NUM_GPUS} | device: {device}")

# ==========================================
# 基础路径与全局设置
# ==========================================
halo_finder_root = Path("/home/sam/Halo_Finder")
base_path = (halo_finder_root / "halo_finder_v1").as_posix() + "/"
sz_lib_path = r"/home/sam/Data_Compression/SZ3/build/lib64/libSZ3c.so"
pysz_path = r"/home/sam/Data_Compression/SZ3/tools/pysz"
data_shape = (1024, 1024, 1024)

FIELD_FILES = []
TARGET_STEMS = ["miranda_1024x1024x1024_float32"]
FIELD_LABEL = {
    "miranda_1024x1024x1024_float32": "Miranda",
}

REL_PROBE = 5e-3
sz_engine = SZ(sz_lib_path)

def rel_sz_suffix(rel_err):
    return f"{float(rel_err):.0e}".replace("+", "")

def load_field_data(target_stem, rel_probe=REL_PROBE):
    fname = f"{target_stem}.raw"
    gt_path = base_path + fname
    aux_paths = []
    sz_bin = base_path + target_stem + "_rel" + rel_sz_suffix(rel_probe) + ".sz"
    
    if not Path(sz_bin).is_file():
        vol = np.fromfile(gt_path, dtype=np.float32).reshape(data_shape)
        Path(sz_bin).write_bytes(sz_engine.compress(vol, 1, 0, float(rel_probe), 0)[0])
        
    Xs, Xps = load_multifield_from_disk(
        gt_path=gt_path, aux_paths=aux_paths, sz_bin_path=sz_bin,
        data_shape=data_shape, pysz_path=pysz_path, sz_lib_path=sz_lib_path,
    )
    gt_target = np.asarray(Xs[0], np.float32)
    aux_fields = [np.asarray(f, np.float32) for f in Xs[1:]]

    def build_Xps_for_rel(rel_err):
        b, _cr = sz_engine.compress(gt_target, 1, 0, float(rel_err), 0)
        x_lq = sz_engine.decompress(b, gt_target.shape, np.float32)
        return [x_lq] + aux_fields, float(_cr), int(len(b))

    return {
        "target_stem": target_stem,
        "field_label": FIELD_LABEL.get(target_stem, target_stem[:8]),
        "Xs": Xs,
        "gt_target": gt_target,
        "build_Xps_for_rel": build_Xps_for_rel,
        "n_fields": len(Xs),
        "original_target_bytes": int(gt_target.nbytes),
    }

def build_cfg_global(ctx, rel_err, max_train_time, bg_h, steps_per_epoch, log_prefix=""):
    Xs = ctx["Xs"]
    Xps_list = ctx["Xps_list"]
    
    cfg = build_bg_only_cfg(
        X_target=Xs[0],
        Xps=Xps_list,
        max_train_time=max_train_time,
        epochs=200,
        steps_per_epoch=steps_per_epoch,
        bg_h=bg_h,
        bg_batch=1,
        bg_patch_size=1024,
        lr=1e-4,
    )
    
    cfg.bg_sample_mode = "sequential"
    cfg.bg_log_prefix = log_prefix
    cfg.bg_arch ="spatial" 
    cfg.bg_split_bands= 'three'
    cfg.amp = True
    cfg.amp_dtype = "bf16"
    cfg.bg_feat_attn = True
    cfg.bg_low_adapter = True
    cfg.bg_mid_adapter = True
    cfg.bg_high_adapter = True
    
    return cfg

def _train_single_lr(lr, Xs_chunk, Xps_chunk, device, base_cfg, evaluator_tune, epochs=10):
    """在当前 GPU 上独立训练一个 LR，返回 (final_psnr, hist)"""
    tune_cfg = copy.deepcopy(base_cfg)
    tune_cfg.lr = lr
    tune_cfg.epochs = epochs
    # 如果调用方没有限制，则给个默认值，这里尊重 base_cfg 里的时间限制
    if tune_cfg.max_train_time > 1000:
        tune_cfg.max_train_time = 5.0  # 给每个 LR 5 秒的时间，让曲线能分叉
    tune_cfg.bg_log_prefix = f"Tune-LR{lr:.1e}"
    # Use DDP for tuning so batch size exactly matches Phase 2
    tune_cfg.bg_ddp = True  
    tune_cfg.bg_data_parallel = False
    
    set_seed(42)
    m, hist = train_bg_only(
        Xs=Xs_chunk, Xps=Xps_chunk, device=device,
        cfg=tune_cfg, evaluator=evaluator_tune
    )
    
    final_psnr = 0.0
    if len(hist.get("psnr", [])) > 0:
        final_psnr = hist["psnr"][-1][1] if isinstance(hist["psnr"][-1], tuple) else hist["psnr"][-1]
        
    sd_cpu = {k: v.cpu() for k, v in m.state_dict().items()}
    return final_psnr, hist, sd_cpu

def autotune_binary_search_single_gpu(Xs_chunk, Xps_chunk, device, base_cfg, evaluator_tune, candidate_lrs):
    """
    Pure Ternary Search for single GPU
    - steps_per_epoch = 64 (approx 0.8s per epoch)
    - exactly 3 epochs per evaluated LR
    - max 5 evaluations to fit within 12s budget
    """
    candidates = list(candidate_lrs)
    N = len(candidates)
    print(f"\n🔍 [Pure Ternary Search - Single GPU] ({N} candidate LRs)")
    
    all_tune_histories = {}
    tested_cache = {}
    local_models = {}
    
    def _evaluate_lr(lr):
        if lr in tested_cache:
            return tested_cache[lr]
            
        tune_cfg = copy.deepcopy(base_cfg)
        tune_cfg.epochs = 3
        tune_cfg.max_train_time = 1e9  # bound purely by epochs
        
        # Wrapped evaluator
        def wrapped_evaluator(m, current_cfg=tune_cfg):
            psnr, loss = evaluator_tune(m, current_cfg)
            return psnr, loss
            
        _, hist, sd_cpu = _train_single_lr(
            lr, Xs_chunk, Xps_chunk, device, tune_cfg, wrapped_evaluator, epochs=3
        )
        
        # extract max psnr (using history to capture initial state and all 3 epochs)
        max_psnr = -1.0
        if "psnr" in hist:
            for p in hist["psnr"]:
                val = p[1] if isinstance(p, tuple) else p
                if val > max_psnr:
                    max_psnr = val
        
        tested_cache[lr] = max_psnr
        all_tune_histories[lr] = hist
        local_models[lr] = sd_cpu
        
        print(f"  🎯 LR={lr:.1e} → Best PSNR (over 3 epochs): {max_psnr:.2f} dB")
        return max_psnr

    lo = 0
    hi = N - 1
    round_idx = 0
    
    while hi - lo > 2:
        round_idx += 1
        third = (hi - lo) // 3
        m1 = lo + third
        m2 = hi - third
        
        print(f"\n  📐 Round {round_idx}: Search interval [{candidates[lo]:.1e}, {candidates[hi]:.1e}] | Probing = {candidates[m1]:.1e}, {candidates[m2]:.1e}")
        
        psnr1 = _evaluate_lr(candidates[m1])
        psnr2 = _evaluate_lr(candidates[m2])
        
        if psnr1 > psnr2:
            hi = m2
            print(f"    → Left probe is better, moving interval left")
        else:
            lo = m1
            print(f"    → Right probe is better, moving interval right")
            
    print(f"\n  🔍 Finalizing remaining candidates in interval [{candidates[lo]:.1e}, {candidates[hi]:.1e}]")
    for i in range(lo, hi + 1):
        _evaluate_lr(candidates[i])
        
    best_lr = max(tested_cache, key=tested_cache.get)
    best_psnr = tested_cache[best_lr]
    best_sd = local_models[best_lr]
    
    print(f"\n🏆 [Pure Ternary Search] Search Complete!")
    print(f"   Best LR: {best_lr:.1e} (Highest PSNR: {best_psnr:.2f} dB)\n")
    
    return best_lr, all_tune_histories, best_sd


def autotune_successive_halving(Xs_chunk, Xps_chunk, device, base_cfg, evaluator_tune,
                                candidate_lrs):
    """
    Successive Halving AutoTune:
    1. Initially evaluate all candidate LRs.
    2. Round 1: Assign a small time budget to each LR, evaluate PSNR, and discard the bottom 50%.
    3. Round 2: Double the time budget for surviving LRs, evaluate again, discard another 50%.
    4. Repeat until 1~2 LRs remain and give them the maximum budget to decide the winner.
    This avoids local optima while concentrating compute resources on the most promising LRs.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    candidates = list(candidate_lrs)
    N = len(candidates)
    
    # Define 3 rounds of budgets (using base_cfg.max_train_time as a base unit)
    # round 1: 1x time (e.g., 2s)
    # round 2: 2x time (e.g., 4s)
    # round 3: 4x time (e.g., 8s)
    # Note: Since _train_single_lr trains from scratch every time, time budgets are independent.
    base_time = base_cfg.max_train_time if base_cfg.max_train_time < 1000 else 2.0
    budgets = [base_time, base_time * 2, base_time * 4]
    
    if local_rank == 0:
        print(f"\n🔍 [Successive Halving AutoTune] {N} candidate LRs")
        print(f"   LR Range: [{candidates[0]:.1e}, {candidates[-1]:.1e}]")
        print(f"   Budgets (seconds per LR): {budgets}\n")
    
    all_tune_histories = {}
    current_candidates = list(candidates)
    
    for round_idx, budget in enumerate(budgets):
        if local_rank == 0:
            print(f"  📐 Round {round_idx+1}: Surviving {len(current_candidates)} LRs, Budget = {budget:.1f}s")
            
        round_psnrs = {}
        local_models = {}
        
        # All GPUs sequentially train the same LR in DDP mode
        for lr in current_candidates:
            # Modify budget
            tune_cfg = copy.deepcopy(base_cfg)
            tune_cfg.max_train_time = budget
            
            psnr_val, hist, sd_cpu = _train_single_lr(
                lr, Xs_chunk, Xps_chunk, device, tune_cfg, evaluator_tune, epochs=100
            )
            
            round_psnrs[lr] = psnr_val
            local_models[lr] = sd_cpu
            all_tune_histories[lr] = hist
            
            if local_rank == 0:
                print(f"  🎯 [DDP] LR={lr:.1e} finished in {budget:.1f}s → PSNR: {psnr_val:.2f} dB")
            
        dist.barrier()
        
        # Sort by PSNR descending
        sorted_lrs = sorted(round_psnrs.keys(), key=lambda x: round_psnrs[x], reverse=True)
        
        if local_rank == 0:
            print(f"    → Round Rankings:")
            for rank, lr in enumerate(sorted_lrs):
                print(f"      {rank+1}. LR={lr:.1e} (PSNR: {round_psnrs[lr]:.2f} dB)")
                
        # Eliminate bottom 50%, except on the last round
        if round_idx < len(budgets) - 1 and len(current_candidates) > 1:
            keep_k = max(1, len(current_candidates) // 2)
            current_candidates = sorted_lrs[:keep_k]
            if local_rank == 0:
                print(f"    → Eliminating bottom half. Advancing LRs: {[f'{x:.1e}' for x in current_candidates]}")
        else:
            current_candidates = sorted_lrs
    
    best_lr = current_candidates[0]
    best_psnr = round_psnrs[best_lr]
    
    # Since all GPUs trained the model using DDP, they all already have the identical state_dict!
    best_sd = local_models[best_lr]
    
    if local_rank == 0:
        print(f"\n🏆 [Successive Halving AutoTune] Elimination Complete!")
        print(f"   Best LR: {best_lr:.1e} (Highest PSNR: {best_psnr:.2f} dB)\n")
    
    dist.barrier()
    return best_lr, all_tune_histories, best_sd


def main():
    # 支持单卡直接运行 (python lr_Miranda.py) 和多卡 DDP 运行
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    TARGET_STEM = "miranda_1024x1024x1024_float32"
    TEST_REL_ERR = 1e-3
    
    if local_rank == 0:
        print(f"\n{'#'*60}\n# 🚀 开始 DDP 自适应优化测试: {TARGET_STEM}\n{'#'*60}")
    
        
    ctx = load_field_data(TARGET_STEM)
    Xs_list, sz_cr, sz_bytes = ctx["build_Xps_for_rel"](TEST_REL_ERR)
    if local_rank == 0:
        original_bytes = ctx["original_target_bytes"]
        print(f"📊 SZ3 压缩: rel={TEST_REL_ERR:.0e} | CR = {sz_cr:.2f}x | "
              f"原始 {original_bytes/1e9:.2f} GB → 压缩后 {sz_bytes/1e6:.2f} MB")
    Xs_list = [np.asarray(f, np.float32) for f in Xs_list]
    
    Xs = ctx["Xs"]
    Xps_list = Xs_list
    ctx["Xps_list"] = Xps_list
    
    dist.barrier()
    
    LR_LIST = [float(x) for x in np.logspace(-4, np.log10(2e-2), 10)]
    
    # ==========================================
    # Phase 1: Pyramid Downsample (1/32 Volume) + Pure Ternary Search
    # ==========================================
    if local_rank == 0:
        print(f"\n{'='*50}\n▶ [Phase 1] 1/32 Pyramid Downsample (32 slices) + Pure Ternary Search\n{'='*50}")
    
    # 1/32 Volume downsample: 32 strided slices across the ENTIRE volume.
    # This guarantees identical global spatial distribution and gradient landscape.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    start_z_tune = local_rank
    step_tune = 32 * world_size
    Xs_tune_chunk = [x[start_z_tune::step_tune] for x in Xs]
    Xps_tune_chunk = [x[start_z_tune::step_tune] for x in Xps_list]
    
    # We evaluate on the SAME tiny chunk to keep PSNR check under 0.1s
    Xs_tune = Xs_tune_chunk
    Xps_list_tune = Xps_tune_chunk
    
    try:
        # Estimate bg_h based on FULL dataset shape, so Phase 1 and Phase 2 match exactly!
        h_30k_tune, _ = pick_bg_h_under_budget(
            30000, shape=Xs[0].shape, n_fields=len(Xps_tune_chunk),
            bg_arch="spatial",
            bg_split_bands=True,
            bg_split_mode="three",
            bg_feat_attn=True,
            bg_low_adapter=True,
            bg_mid_adapter=True,
            bg_high_adapter=True
        )
        h_30k_tune = int(h_30k_tune)
    except Exception:
        h_30k_tune = 5
        
    cfg_ddp_tune = build_cfg_global(
        ctx, TEST_REL_ERR, max_train_time=1e9, bg_h=h_30k_tune, 
        steps_per_epoch=64, log_prefix="DDP-Tune"  # 改为 64，3个epoch刚好占用约2.4秒
    )
    cfg_ddp_tune.bg_batch = 1
    # IDENTICAL TO PHASE 2: 1024 patch, sequential mode!
    cfg_ddp_tune.bg_patch_size = 1024
    cfg_ddp_tune.bg_sample_mode = "sequential"
    cfg_ddp_tune.bg_ddp = True
    cfg_ddp_tune.bg_data_parallel = False
    
    def evaluator_tune(m, current_cfg=cfg_ddp_tune):
        m_core = unwrap_bg_model(m)
        xh = run_bg_inference(m_core, Xs_tune, Xps_list_tune, current_cfg, TEST_REL_ERR)
        
        data_range = float(Xs_tune[0].max() - Xs_tune[0].min())
        if data_range <= 0:
            data_range = 1.0
            
        mse = float(np.mean((Xs_tune[0] - xh)**2))
        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse + 1e-12) if mse > 0 else 100.0
        return psnr, 0.0
    
    base_tune_cfg = copy.deepcopy(cfg_ddp_tune)
    best_ddp_lr, tune_histories, best_sd = autotune_binary_search_single_gpu(
        Xs_tune_chunk, Xps_tune_chunk, device, base_tune_cfg, evaluator_tune,
        candidate_lrs=LR_LIST
    )
    
    dist.barrier()
    
    # ==========================================
    # Phase 2: Run Full Dataset with Best LR (and Control Group)
    # ==========================================
    if local_rank == 0:
        print(f"\n{'='*50}\n▶ [Phase 2] Full Dataset Training with Best LR = {best_ddp_lr:.1e} ({TARGET_STEM})\n{'='*50}")
        
    # Phase 2 uses the FULL 1024 dataset (no striding)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    slices_per_gpu_p2 = max(1, 1024 // world_size)
    start_z_p2 = local_rank * slices_per_gpu_p2
    end_z_p2 = (local_rank + 1) * slices_per_gpu_p2
    Xs_chunk_p2 = [x[start_z_p2:end_z_p2] for x in Xs]
    Xps_chunk_p2 = [x[start_z_p2:end_z_p2] for x in Xps_list]
        
    try:
        # Estimate bg_h based on chunk shape, matching the EXACT architecture
        h_30k, _ = pick_bg_h_under_budget(
            30000, shape=Xs_chunk_p2[0].shape, n_fields=len(Xps_chunk_p2),
            bg_arch="spatial", #resunet_small
            bg_split_bands=True,
            bg_split_mode="three",
            bg_feat_attn=True,
            bg_low_adapter=True,
            bg_mid_adapter=True,
            bg_high_adapter=True
        )
        h_30k = int(h_30k)
    except Exception:
        h_30k = 5
        
    full_histories = {}
    for test_lr in LR_LIST:
        if local_rank == 0:
            if test_lr == best_ddp_lr:
                print(f"\n▶ Validating Full DDP | LR = {test_lr:.1e} [Fair Ablation: From Scratch]")
            else:
                print(f"\n▶ Validating Full DDP | LR = {test_lr:.1e}")
            
        cfg_ddp_real = build_cfg_global(
            ctx, TEST_REL_ERR, max_train_time=108.0, bg_h=h_30k, 
            steps_per_epoch=1024, log_prefix=f"Full-LR{test_lr:.1e}"
        )
        cfg_ddp_real.bg_batch = 1
        cfg_ddp_real.lr = test_lr
        cfg_ddp_real.epochs = 7  # 核心改动：强制让 T_max = 7 epochs，使得 Cosine Scheduler 在 70s 内充分衰减
        cfg_ddp_real.bg_ddp = True
        cfg_ddp_real.bg_data_parallel = False
        
        def evaluator_ddp(m, current_cfg=cfg_ddp_real):
            m_core = unwrap_bg_model(m)
            xh = run_bg_inference(m_core, Xs, Xps_list, current_cfg, TEST_REL_ERR)
            data_range = float(Xs[0].max() - Xs[0].min())
            if data_range <= 0: data_range = 1.0
            mse = float(np.mean((Xs[0] - xh)**2))
            psnr = 20 * np.log10(data_range) - 10 * np.log10(mse + 1e-12) if mse > 0 else 100.0
            max_err = float(np.max(np.abs(Xs[0] - xh)))
            return psnr, max_err
            
        set_seed(42)
        
        # Fair Ablation: Ensure ALL LRs start from scratch in Phase 2
        init_sd = None
        
        model_ddp, hist_ddp = train_bg_only(
            Xs=Xs_chunk_p2, Xps=Xps_chunk_p2, device=device, 
            cfg=cfg_ddp_real, evaluator=evaluator_ddp,
            init_state_dict=init_sd
        )
        full_histories[test_lr] = hist_ddp
        dist.barrier()
    
    if local_rank == 0:
        print("\n--- Full DDP Training Finished ---")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # ==================================
        # 左图: 1/64 数据 Binary Search Tune 阶段
        # ==================================
        tested_lrs = sorted(tune_histories.keys())
        for lr in tested_lrs:
            hist = tune_histories.get(lr, {})
            t = hist.get("time", [])
            p_raw = hist.get("psnr", [])
            p = [val[1] if isinstance(val, tuple) else val for val in p_raw]
            if len(t) > 0 and len(p) > 0:
                style = 's--' if lr == best_ddp_lr else 'x--'
                lw = 2 if lr == best_ddp_lr else 1
                alpha = 0.9 if lr == best_ddp_lr else 0.4
                
                label_str = f"Best: LR={lr:.1e}" if lr == best_ddp_lr else f"LR={lr:.1e}"
                
                axes[0].plot(t, p, style, linewidth=lw, alpha=alpha, label=label_str)
                
        axes[0].set_title(f"Phase 1: Pure Ternary Search on Global Random Sampling\n"
            f"Evaluated {len(tested_lrs)}/10 LRs | Best LR = {best_ddp_lr:.1e}")
        axes[0].set_xlabel("Wall Time (s)")
        axes[0].set_ylabel("Global Validation PSNR (dB)")
        axes[0].grid(True, alpha=0.35)
        axes[0].legend()
        
        # ==================================
        # 右图: 完整数据集跑满 (60s limit)
        # ==================================
        for lr in LR_LIST:
            hist = full_histories.get(lr, {})
            t = hist.get("time", [])
            p_raw = hist.get("psnr", [])
            p = [val[1] if isinstance(val, tuple) else val for val in p_raw]
            if len(t) > 0 and len(p) > 0:
                style = 'o-' if lr == best_ddp_lr else '^-'
                lw = 2.5 if lr == best_ddp_lr else 1.0
                alpha = 1.0 if lr == best_ddp_lr else 0.4
                
                label_str = f"Best: LR={lr:.1e}" if lr == best_ddp_lr else f"LR={lr:.1e}"
                
                axes[1].plot(t, p, style, linewidth=lw, alpha=alpha, label=label_str)
                
        axes[1].set_title("Phase 2: Full Data Training (108s limit) - Fair Ablation")
        axes[1].set_xlabel("Wall Time (s)")
        axes[1].set_ylabel("Global Validation PSNR (dB)")
        axes[1].grid(True, alpha=0.35)
        axes[1].legend()
        
        plt.suptitle(f"Autotune Results: {TARGET_STEM} (rel={TEST_REL_ERR:.0e})", fontsize=14)
        plt.tight_layout()
        
        out_png = f"autotune_validation_Miranda_spatial_rel_{TEST_REL_ERR:.0e}_CR_{sz_cr:.2f}.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"\n✅ 验证图表已保存至: {out_png}")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
