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


def _global_diag(x_true, x_hat):
    """Global reconstruction metrics (replaces the old ROI diagnostics)."""
    x_true = np.asarray(x_true); x_hat = np.asarray(x_hat)
    dr = float(x_true.max() - x_true.min()) or 1.0
    mse = float(np.mean((x_true - x_hat) ** 2))
    psnr = 20 * np.log10(dr) - 10 * np.log10(mse + 1e-12) if mse > 0 else 100.0
    max_err = float(np.max(np.abs(x_true - x_hat)))
    return {"psnr": psnr, "max_err": max_err}

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


NUM_GPUS = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"cuda devices: {NUM_GPUS} | device: {device}")

# ==========================================
# 基础路径与全局设置
# ==========================================
halo_finder_root = Path(__file__).resolve().parent.parent.parent.parent
base_path = (halo_finder_root / "halo_finder_v1").as_posix() + "/"
sz_lib_path = r"/home/sam/Data_Compression/SZ3/build/lib64/libSZ3c.so"
pysz_path = r"/home/sam/Data_Compression/SZ3/tools/pysz"
data_shape = (1024, 1024, 1024)

FIELD_FILES = []
TARGET_STEMS = ["miranda_1024x1024x1024_float32"]
FIELD_LABEL = {
    "miranda_1024x1024x1024_float32": "Miranda",
}

REL_PROBE = 1e-4
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
    
    # 初始化标准参数
    cfg = build_bg_only_cfg(
        X_target=Xs[0],
        Xps=Xps_list,
        max_train_time=max_train_time,
        epochs=200,
        steps_per_epoch=steps_per_epoch,
        bg_h=bg_h,
        bg_batch=1,
        bg_patch_size=512,
        lr=1e-4,  # 默认 LR
    )
    
    # 手动挂载额外的控制参数
    cfg.bg_sample_mode = "sequential"
    cfg.bg_log_prefix = log_prefix
    cfg.bg_arch = "spatial" 
    cfg.amp = True
    cfg.amp_dtype = "bf16"
    
    return cfg

def main():
    # 1. 初始化 DDP 环境
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # 2. 准备数据
    TARGET_STEM = "miranda_1024x1024x1024_float32"
    TEST_REL_ERR = 5e-4
    
    if local_rank == 0:
        print(f"\n{'#'*60}\n# 🚀 开始 DDP 测试: {TARGET_STEM}\n{'#'*60}")
    
    # 对于 Miranda 数据，我们没有 ROI 信息，所以直接给 None 和空列表
    
    ctx = load_field_data(TARGET_STEM)
    
    if local_rank == 0:
        print(f"📊 测试不同 rel_err 对应的 CR (Compression Ratio):")
        for test_err in [1e-3, 5e-4, 1e-4, 5e-5]:
            _, cr_val, _ = ctx["build_Xps_for_rel"](test_err)
            print(f"   rel_err = {test_err:.0e} -> CR = {cr_val:.2f}")
            
    Xs_list, cr_used, _ = ctx["build_Xps_for_rel"](TEST_REL_ERR)
    if local_rank == 0:
        print(f"\n=> 当前实验使用的 TEST_REL_ERR = {TEST_REL_ERR:.0e}, 对应 CR = {cr_used:.2f}\n")
        
    Xs_list = [np.asarray(f, np.float32) for f in Xs_list]
    
    # 这里的 Xs_list 就是 Xps_list，真正的高精度数据在 ctx["Xs"] 里
    Xs = ctx["Xs"]
    Xps_list = Xs_list
    ctx["Xps_list"] = Xps_list
    
    dist.barrier() # 同步各进程
    
    # 3. 截断切片，各跑各的 256 层 (1024/4 = 256)
    start_z = local_rank * 256
    end_z = (local_rank + 1) * 256
    
    Xs_chunk = [x[start_z:end_z] for x in Xs]
    Xps_chunk = [x[start_z:end_z] for x in Xps_list]
    
    if local_rank == 0:
        print(f"[Rank {local_rank}] Data sliced: Z-range [{start_z}, {end_z})")
        print(f"[Rank {local_rank}] Shape of chunk: {Xs_chunk[0].shape}")
    
    # 动态匹配参数
    try:
        h_30k, _ = pick_bg_h_under_budget(30000, shape=Xs_chunk[0].shape, n_fields=len(Xps_chunk))
        h_30k = int(h_30k)
    except Exception:
        h_30k = 20
        
    LR_LIST = [1e-4, 4e-4, 1e-3, 4e-3]
    
    ddp_histories = {}
    for test_lr in LR_LIST:
        if local_rank == 0:
            print(f"\n{'='*50}\n▶ DDP Training | LR = {test_lr}\n{'='*50}")
            
        cfg_ddp = build_cfg_global(
            ctx, TEST_REL_ERR, max_train_time=60.0, bg_h=h_30k, 
            steps_per_epoch=256, log_prefix=f"DDP-LR{test_lr}"
        )
        
        cfg_ddp.bg_batch = 1 # DDP里每卡 batch=1
        cfg_ddp.lr = test_lr
        cfg_ddp.bg_ddp = True
        cfg_ddp.bg_data_parallel = False
        
        def evaluator_ddp(m, current_cfg=cfg_ddp):
            m_core = unwrap_bg_model(m)
            xh = run_bg_inference(m_core, Xs, Xps_list, current_cfg, TEST_REL_ERR)
            m2 = _global_diag(Xs[0], xh)
            return m2["psnr"], m2["max_err"]
            
        set_seed(42)
        model_ddp, hist_ddp = train_bg_only(
            Xs=Xs_chunk, Xps=Xps_chunk, device=device, 
            cfg=cfg_ddp, evaluator=evaluator_ddp
        )
        ddp_histories[test_lr] = hist_ddp
    
    if local_rank == 0:
        print("\n--- DDP LR Sweep Finished ---")
        
    dist.destroy_process_group()

    # 4. 跑一个 Single GPU 的 LR Sweep 基准
    if local_rank == 0:
        print(f"\n{'#'*60}\n# 🚀 开始单卡基准测试 LR Sweep: {TARGET_STEM}\n{'#'*60}")
        try:
            h_single, _ = pick_bg_h_under_budget(30000 * 4, shape=Xs[0].shape, n_fields=len(Xps_list))
            h_single = int(h_single)
        except Exception:
            h_single = 20
            
        single_histories = {}
        for test_lr in LR_LIST:
            print(f"\n{'='*50}\n▶ Single GPU Training | LR = {test_lr}\n{'='*50}")
            
            cfg_single = build_cfg_global(
                ctx, TEST_REL_ERR, max_train_time=60.0, bg_h=h_single, 
                steps_per_epoch=1024, log_prefix=f"Single-LR{test_lr}"
            )
            cfg_single.bg_batch = 4
            cfg_single.bg_patch_size = 512
            cfg_single.lr = test_lr
            cfg_single.bg_ddp = False
            
            def evaluator_single(m, current_cfg=cfg_single):
                xh = run_bg_inference(m, Xs, Xps_list, current_cfg, TEST_REL_ERR)
                m2 = _global_diag(Xs[0], xh)
                return m2["psnr"], m2["max_err"]
                
            set_seed(42)
            model_single, hist_single = train_bg_only(
                Xs=Xs, Xps=Xps_list, device=device, 
                cfg=cfg_single, evaluator=evaluator_single
            )
            single_histories[test_lr] = hist_single
        
        # 5. 画图对比: 单卡和多卡分开 (Subplots)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # DDP 曲线
        for test_lr in LR_LIST:
            hist = ddp_histories.get(test_lr, {})
            t = hist.get("time", [])
            p_raw = hist.get("psnr", [])
            p = [val[1] if isinstance(val, tuple) else val for val in p_raw]
            if len(t) > 0 and len(p) > 0:
                axes[0].plot(t, p, 'o-', label=f"LR = {test_lr:.1e}")
                
        axes[0].set_title(f"4-GPU DDP ({TARGET_STEM}, rel={TEST_REL_ERR:.0e})")
        axes[0].set_xlabel("Wall Time (s)")
        axes[0].set_ylabel("Global Validation PSNR (dB)")
        axes[0].grid(True, alpha=0.35)
        axes[0].legend()
        
        # Single GPU 曲线
        for test_lr in LR_LIST:
            hist = single_histories.get(test_lr, {})
            t = hist.get("time", [])
            p_raw = hist.get("psnr", [])
            p = [val[1] if isinstance(val, tuple) else val for val in p_raw]
            if len(t) > 0 and len(p) > 0:
                axes[1].plot(t, p, 's-', label=f"LR = {test_lr:.1e}")
                
        axes[1].set_title(f"Single GPU ({TARGET_STEM}, rel={TEST_REL_ERR:.0e})")
        axes[1].set_xlabel("Wall Time (s)")
        axes[1].set_ylabel("Global Validation PSNR (dB)")
        axes[1].grid(True, alpha=0.35)
        axes[1].legend()
        
        plt.tight_layout()
        out_png = "lr_sweep_ddp_vs_single.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"\n✅ 画图已保存至: {out_png}")

if __name__ == "__main__":
    main()
