import random
import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.distributed as dist
import copy

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

import ctypes
libfabric_path = '/opt/cray/libfabric/2.2.0rc1/lib64/libfabric.so.1'
try:
    ctypes.CDLL(libfabric_path, mode=ctypes.RTLD_GLOBAL)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"✅ 成功从 {libfabric_path} 强制加载 libfabric")
except Exception as e:
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"❌ 强制加载失败: {e}")
    
NUM_GPUS = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print(f"cuda devices: {NUM_GPUS} | device: {device}")

# ==========================================
# 基础路径与全局设置
# ==========================================
halo_finder_root = Path("/home/sam/Halo_Finder")
base_path = (halo_finder_root / "halo_finder_v1/SDRBENCH-EXASKY-NYX-512x512x512/origin").as_posix() + "/"
sz_lib_path = r"/home/sam/Data_Compression/SZ3/build/lib64/libSZ3c.so"
pysz_path = r"/home/sam/Data_Compression/SZ3/tools/pysz"
data_shape = (512, 512, 512)

FIELD_FILES = [
    "dark_matter_density.f32", "velocity_z.f32", "baryon_density.f32",
    "temperature.f32", "velocity_x.f32", "velocity_y.f32",
]
TARGET_STEMS = ["dark_matter_density", "baryon_density", "temperature"]
FIELD_LABEL = {
    "dark_matter_density": "DM",
    "baryon_density": "BD",
    "temperature": "T",
}

REL_PROBE = 1e-4
sz_engine = SZ(sz_lib_path)

def rel_sz_suffix(rel_err):
    return f"{float(rel_err):.0e}".replace("+", "")

def load_field_data(target_stem, rel_probe=REL_PROBE):
    fname = f"{target_stem}.f32"
    gt_path = base_path + fname
    aux_paths = [base_path + f for f in FIELD_FILES if f != fname]
    sz_bin = base_path + Path(fname).stem + "_rel" + rel_sz_suffix(rel_probe) + ".sz"
    
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
        bg_patch_size=512,
        lr=1e-4,
    )
    
    cfg.bg_sample_mode = "sequential"
    cfg.bg_log_prefix = log_prefix
    cfg.bg_arch = "spatial" 
    cfg.amp = True
    cfg.amp_dtype = "bf16"
    
    return cfg

def autotune_fast_parallel(Xs_tune, Xps_list_tune, device, base_cfg, evaluator_tune, candidate_lrs):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 轮询分配 LR 给各卡
    local_lrs = [candidate_lrs[i] for i in range(len(candidate_lrs)) if i % world_size == local_rank]
    
    if local_rank == 0:
        print(f"\n🔍 [Parallel AutoTune] 采用 {world_size} 卡独立并行测试 {len(candidate_lrs)} 个 LRs...")
        print(f"   (每张卡将使用完整的 1/64 数据独立测试分配给它的 LR，互相不干扰)")
        
    local_histories = {}
    local_best_lr = candidate_lrs[0] if len(candidate_lrs)>0 else 1e-4
    local_best_psnr = -1.0
    
    for lr in local_lrs:
        tune_cfg = copy.deepcopy(base_cfg)
        tune_cfg.lr = lr
        tune_cfg.epochs = 10
        tune_cfg.max_train_time = 9999.0  # 不按时间早停，严格按 epoch 跑
        tune_cfg.bg_log_prefix = f"Tune-LR{lr:.1e}"
        tune_cfg.bg_ddp = False  # 关闭 DDP，让这块卡自己独立跑！
        
        set_seed(42)
        _, hist = train_bg_only(
            Xs=Xs_tune, Xps=Xps_list_tune, device=device, 
            cfg=tune_cfg, evaluator=evaluator_tune
        )
        
        local_histories[lr] = hist
        
        final_psnr = 0.0
        if len(hist.get("psnr", [])) > 0:
            final_psnr = hist["psnr"][-1][1] if isinstance(hist["psnr"][-1], tuple) else hist["psnr"][-1]
            
        print(f" 🎯 [Rank {local_rank}] LR={lr:.1e} 测试结束 | PSNR: {final_psnr:.2f} dB")
        
    # 同步所有 GPU 的测试结果字典
    dist.barrier()
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, local_histories)
    
    tune_histories = {}
    for d in gather_list:
        if d is not None:
            tune_histories.update(d)
            
    best_lr = candidate_lrs[0]
    best_psnr = -1.0
    for lr, hist in tune_histories.items():
        if len(hist.get("psnr", [])) > 0:
            final_psnr = hist["psnr"][-1][1] if isinstance(hist["psnr"][-1], tuple) else hist["psnr"][-1]
            if final_psnr > best_psnr:
                best_psnr = final_psnr
                best_lr = lr
                
    if local_rank == 0:
        print(f"🏆 [Parallel AutoTune] 并行搜索汇总完成！全局最佳 LR: {best_lr:.1e} (早期 PSNR: {best_psnr:.2f} dB)\n")
        
    dist.barrier()
    return best_lr, tune_histories

def main():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    TARGET_STEM = "temperature"
    TEST_REL_ERR = 5e-4
    
    if local_rank == 0:
        print(f"\n{'#'*60}\n# 🚀 开始 DDP 自适应优化测试: {TARGET_STEM}\n{'#'*60}")
    
    
    ctx = load_field_data(TARGET_STEM)
    Xs_list, _, _ = ctx["build_Xps_for_rel"](TEST_REL_ERR)
    Xs_list = [np.asarray(f, np.float32) for f in Xs_list]
    
    Xs = ctx["Xs"]
    Xps_list = Xs_list
    ctx["Xps_list"] = Xps_list
    
    dist.barrier()
    
    LR_LIST = [float(x) for x in np.logspace(-5, -2, 10)]
    
    # ==========================================
    # 阶段 1: 抽取 1/64 极小数据集进行快速 AutoTune
    # ==========================================
    if local_rank == 0:
        print(f"\n{'='*50}\n▶ [阶段 1] 截取 1/64 数据 (128x128x128) 进行快速 AutoTune\n{'='*50}")
    
    # 抽取 8 级全分辨率切片 -> 同样是 1/64 体积 (8 * 512 * 512 = 128^3)
    Xs_tune = [x[:8, :, :] for x in Xs]
    Xps_list_tune = [x[:8, :, :] for x in Xps_list]
    
    # DDP 对 Z 轴均分 4 份 (每张卡分 2 片)
    start_z_tune = local_rank * 2
    end_z_tune = (local_rank + 1) * 2
    Xs_chunk_tune = [x[start_z_tune:end_z_tune] for x in Xs_tune]
    Xps_chunk_tune = [x[start_z_tune:end_z_tune] for x in Xps_list_tune]
    
    try:
        h_30k_tune, _ = pick_bg_h_under_budget(30000, shape=Xs_chunk_tune[0].shape, n_fields=len(Xps_chunk_tune))
        h_30k_tune = int(h_30k_tune)
    except Exception:
        h_30k_tune = 10
        
    cfg_ddp_tune = build_cfg_global(
        ctx, TEST_REL_ERR, max_train_time=1e9, bg_h=h_30k_tune, 
        steps_per_epoch=8, log_prefix="DDP-Tune"
    )
    cfg_ddp_tune.bg_batch = 1
    cfg_ddp_tune.bg_patch_size = 512  # 适配 512x512 的切片，与 Phase 2 对齐
    cfg_ddp_tune.bg_ddp = True
    cfg_ddp_tune.bg_data_parallel = False
    
    def evaluator_tune(m, current_cfg=cfg_ddp_tune):
        m_core = unwrap_bg_model(m)
        xh = run_bg_inference(m_core, Xs_tune, Xps_list_tune, current_cfg, TEST_REL_ERR)
        
        # 为了和 full PSNR 在图上对齐，必须计算真实 Data Range
        data_range = float(Xs_tune[0].max() - Xs_tune[0].min())
        if data_range <= 0:
            data_range = 1.0
            
        mse = float(np.mean((Xs_tune[0] - xh)**2))
        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse + 1e-12) if mse > 0 else 100.0
        return psnr, 0.0
    
    best_ddp_lr, tune_histories = autotune_fast_parallel(
        Xs_tune, Xps_list_tune, device, cfg_ddp_tune, evaluator_tune, candidate_lrs=LR_LIST
    )
    
    dist.barrier()
    
    # ==========================================
    # 阶段 2: 跑完整数据集对比所有 LR，验证 AutoTune 的有效性
    # ==========================================
    if local_rank == 0:
        print(f"\n{'='*50}\n▶ [阶段 2] 跑完整数据集对比所有 LR，验证 AutoTune 的有效性\n{'='*50}")
        
    start_z = local_rank * 128
    end_z = (local_rank + 1) * 128
    Xs_chunk = [x[start_z:end_z] for x in Xs]
    Xps_chunk = [x[start_z:end_z] for x in Xps_list]
    
    try:
        h_30k, _ = pick_bg_h_under_budget(30000, shape=Xs_chunk[0].shape, n_fields=len(Xps_chunk))
        h_30k = int(h_30k)
    except Exception:
        h_30k = 10
        
    full_histories = {}
    for test_lr in LR_LIST:
        if local_rank == 0:
            print(f"\n▶ 正在验证 Full DDP | LR = {test_lr:.1e}")
            
        cfg_ddp_real = build_cfg_global(
            ctx, TEST_REL_ERR, max_train_time=60.0, bg_h=h_30k, 
            steps_per_epoch=128, log_prefix=f"Full-LR{test_lr:.1e}"
        )
        cfg_ddp_real.bg_batch = 1
        cfg_ddp_real.lr = test_lr
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
        model_ddp, hist_ddp = train_bg_only(
            Xs=Xs_chunk, Xps=Xps_chunk, device=device, 
            cfg=cfg_ddp_real, evaluator=evaluator_ddp
        )
        full_histories[test_lr] = hist_ddp
        dist.barrier()
    
    if local_rank == 0:
        print("\n--- Full DDP Training Finished ---")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # ==================================
        # 左图: 1/64 数据 Tune 阶段 (前 3 epoch)
        # ==================================
        for lr in LR_LIST:
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
                
        axes[0].set_title(f"Phase 1: Tune on 1/64 Data (3 Epochs)\nBest Predicted LR = {best_ddp_lr:.1e}")
        axes[0].set_xlabel("Wall Time (s)")
        axes[0].set_ylabel("Global Validation PSNR (dB)")
        axes[0].grid(True, alpha=0.35)
        axes[0].legend()
        
        # ==================================
        # 右图: 完整数据集 60s 跑满
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
                
        axes[1].set_title("Phase 2: Full Data Training (60s)")
        axes[1].set_xlabel("Wall Time (s)")
        axes[1].set_ylabel("Global Validation PSNR (dB)")
        axes[1].grid(True, alpha=0.35)
        axes[1].legend()
        
        plt.suptitle(f"Autotune Results: {TARGET_STEM} (rel={TEST_REL_ERR:.0e})", fontsize=14)
        plt.tight_layout()
        
        out_png = "autotune_validation_NYX.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"\n✅ 验证图表已保存至: {out_png}")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
