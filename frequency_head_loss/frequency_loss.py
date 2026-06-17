import os
import re
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.append("/home/sam/Halo_Finder/Final_design/base_script")
from config_io import load_multifield_from_disk
from experiment import build_bg_only_cfg
from bg_stage import run_bg_inference, train_bg_only


def _global_fft_err(x_true, x_hat, n_slices=32):
    """Global FFT magnitude / phase L1 error over a strided set of z-slices."""
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


def _global_diag(x_true, x_hat):
    """Global reconstruction metrics (replaces the old ROI diagnostics)."""
    x_true = np.asarray(x_true); x_hat = np.asarray(x_hat)
    dr = float(x_true.max() - x_true.min()) or 1.0
    mse = float(np.mean((x_true - x_hat) ** 2))
    psnr = 20 * np.log10(dr) - 10 * np.log10(mse + 1e-12) if mse > 0 else 100.0
    max_err = float(np.max(np.abs(x_true - x_hat)))
    fm, fp = _global_fft_err(x_true, x_hat)
    return {"psnr": psnr, "max_err": max_err, "fft_mag_err": fm, "fft_phase_err": fp}

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



# ==== 路径与数据（可按需改 TARGET_STEM / REL_SETTINGS）====
import sys
from pathlib import Path

base_path = Path(r"/home/sam/Halo_Finder/halo_finder_v1/SDRBENCH-EXASKY-NYX-512x512x512/origin").resolve()
base_path = base_path.as_posix() + "/"
sz_lib_path = r"/home/sam/Data_Compression/SZ3/build/lib64/libSZ3c.so"
data_shape = (512, 512, 512)

TARGET_STEM = "dark_matter_density"
FIELD_FILES = [
    "dark_matter_density.f32",
    "velocity_z.f32",
    "baryon_density.f32",
    "temperature.f32",
    "velocity_x.f32",
    "velocity_y.f32",
]

REL_SETTINGS = [("r0", 1e-4), ("r1", 2e-4), ("r2", 3e-4),("r3", 4e-4),("r4", 5e-4),("r5", 6e-4),("r6", 1e-5)]
REL_ACTIVE_IDX = 0
REL_ERR_SZ_BITSTREAM = REL_SETTINGS[REL_ACTIVE_IDX][1]


def rel_sz_suffix(rel_err: float) -> str:
    return f"{rel_err:.0e}".replace("+", "")


def sz_bin_for_target(fname: str, rel_err: float) -> str:
    stem = Path(fname).stem
    return base_path + stem + "_rel" + rel_sz_suffix(rel_err) + ".sz"

fname = TARGET_STEM + ".f32"
gt_path = base_path + fname
aux_paths = [base_path + f for f in FIELD_FILES if f != fname]
sz_bin_path = sz_bin_for_target(fname, REL_ERR_SZ_BITSTREAM)

_sz_path = Path(sz_bin_path)
if not _sz_path.is_file():
    from pysz import SZ
    print("[save .sz] 压缩:", _sz_path)
    eng = SZ(sz_lib_path)
    vol = np.fromfile(gt_path, dtype=np.float32).reshape(data_shape)
    blob, cr = eng.compress(vol, 1, 0, REL_ERR_SZ_BITSTREAM, 0)
    del vol
    _sz_path.parent.mkdir(parents=True, exist_ok=True)
    _sz_path.write_bytes(blob)
    print("CR ≈", float(cr))

Xs, Xps = load_multifield_from_disk(
    gt_path=gt_path,
    aux_paths=aux_paths,
    sz_bin_path=sz_bin_path,
    data_shape=data_shape,
    pysz_path=pysz_path,
    sz_lib_path=sz_lib_path,
)
print("Loaded", TARGET_STEM, "| fields", len(Xs))

# (ROI removed -- global metrics only)

sz = SZ(sz_lib_path)
gt_target = np.asarray(Xs[0], np.float32)
aux_fields = [np.asarray(f, np.float32) for f in Xs[1:]]


def build_Xps_for_rel(rel_err: float):
    b, cr = sz.compress(gt_target, 1, 0, float(rel_err), 0)
    x_lq = sz.decompress(b, gt_target.shape, np.float32)
    return [x_lq] + aux_fields, float(cr), b

print("sanity CR:", build_Xps_for_rel(REL_SETTINGS[0][1])[1])



from pathlib import Path
import re

NEURLZ_CSV_ROOT = Path(
    "/home/sam/Halo_Finder/halo_finder_v1/SDRBENCH-EXASKY-NYX-512x512x512"
).resolve()
FIELD_ABBR = {"DMD": "dark_matter_density", "T": "temperature", "BD": "baryon_density"}


def discover_neurlz_csvs(root: Path):
    seen, out = set(), []
    for pattern in ("sz_decompress/*/decompressed_f32/nyx_SF_*.csv", "**/nyx_SF_*.csv"):
        for p in root.glob(pattern):
            if p.is_file() and p.suffix.lower() == ".csv":
                r = p.resolve()
                if r not in seen:
                    seen.add(r)
                    out.append(p)
    return sorted(out, key=lambda p: (str(p.parent), p.name))


def parse_nyx_sf_csv_name(path: Path):
    m = re.match(r"nyx_SF_([A-Za-z0-9]+)_(.+)\.csv$", path.name, re.I)
    return None if not m else (m.group(1).upper(), m.group(2))


def load_neurlz_official_csv(csv_path: Path, *, field_name: str, rel_slug: str, rel_err_float: float) -> dict:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    for c in ["Epoch", "Average Loss", "PSNR"]:
        if c not in df.columns:
            raise KeyError(c, list(df.columns))
    epochs = df["Epoch"].astype(int).tolist()
    hist = {
        "epoch": epochs,
        "loss": df["Average Loss"].astype(float).tolist(),
        "psnr": list(zip(epochs, df["PSNR"].astype(float).tolist())),
        "psnr_roi": list(zip(epochs, [float("nan")] * len(df))),
        "time": [float("nan")] * len(df),
    }
    metric_log = []
    for _, row in df.iterrows():
        ep = int(row["Epoch"])
        m = {"psnr": float(row["PSNR"]), "epoch": ep}
        if "absDiffMax" in df.columns:
            m["max_err"] = float(row["absDiffMax"])
        metric_log.append(m)
    return {
        "name": f"NeurLZ [{field_name}] rel={rel_slug} [CSV]",
        "cfg": {"bg_arch": "neurlz_unet_official_train_py", "field": field_name, "rel_err": float(rel_err_float)},
        "hist": hist,
        "hint_pretrain_hist": None,
        "metric_log": metric_log,
        "color": "tab:cyan",
        "marker": "X",
        "model": None,
        "rel_err": float(rel_err_float),
        "sz_ratio": float("nan"),
        "sz3_bytes": -1,
        "x_hat": None,
        "ablation_panel": None,
    }

if "results_compare" not in globals():
    results_compare = []

for p in discover_neurlz_csvs(NEURLZ_CSV_ROOT):
    pr = parse_nyx_sf_csv_name(p)
    if not pr:
        continue
    abbr, rel_slug = pr
    field_name = FIELD_ABBR.get(abbr, abbr.lower())
    if field_name != TARGET_STEM:
        continue
    try:
        rf = float(rel_slug)
    except ValueError:
        continue
    results_compare.append(load_neurlz_official_csv(p, field_name=field_name, rel_slug=rel_slug, rel_err_float=rf))

print("NeurLZ CSV entries:", sum(1 for r in results_compare if "[CSV]" in str(r.get("name", ""))))



import time
import re
from pathlib import Path

set_seed(17)

# 每次 train_bg_variant 结束后保存 checkpoint；改为 None 可关闭落盘
BG_CKPT_ROOT = Path("/home/sam/Halo_Finder/halo_finder_v1/scripts/ablation_ckpts")

BG_TRAIN_TIME = 60.0
BG_LR = 1e-3
BG_PATCH = 512
BG_BATCH = 1
MODEL_DTYPE_BYTES = 2

DEFAULT_BG_ARCH = "spatial"
DEFAULT_BG_H = 7
DEFAULT_BG_FIELD_NORM = "zscore"
DEFAULT_BG_FREQ_WEIGHT = 1
DEFAULT_BG_FFT_PHASE_WEIGHT = 1
DEFAULT_BG_FREQ_WARMUP = 1
DEFAULT_BG_SPLIT_MODE = "three"


def model_param_bytes(model, dtype_bytes=None):
    if model is None:
        return 0
    tot = 0
    for p in model.parameters():
        tot += p.numel() * (p.element_size() if dtype_bytes is None else int(dtype_bytes))
    return int(tot)

def purge_ablation_panel(panel: str) -> None:
    global results_compare
    results_compare = [r for r in results_compare if r.get("ablation_panel") != panel]

# 例：purge_ablation_panel("norm")

def psnr_np(x_true, x_hat):
    x_true = np.asarray(x_true, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    mse = float(np.mean((x_true - x_hat) ** 2))
    dr = float(np.max(x_true) - np.min(x_true))
    if mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(max(dr, 1e-12) / np.sqrt(mse)))


def get_run_psnr(r, mode="best"):
    logs = r.get("metric_log", []) or []
    vals = [float(m["psnr"]) for m in logs if m.get("psnr") is not None and np.isfinite(float(m["psnr"]))]
    if vals:
        return float(max(vals) if mode == "best" else vals[-1])
    if r.get("x_hat") is not None:
        return psnr_np(Xs[0], r["x_hat"])
    return float("nan")


def patch_sz3_bytes(r):
    rel = float(r.get("rel_err", float("nan")))
    if not np.isfinite(rel):
        return
    sb = r.get("sz3_bytes", None)
    if sb is None or sb <= 0:
        _, _, b = build_Xps_for_rel(rel)
        r["sz3_bytes"] = int(len(b))
        r["sz_ratio"] = float(gt_target.nbytes / max(len(b), 1))


def _ablation_ckpt_slug(s, max_len=72):
    t = str(s).strip().replace(" ", "_")
    t = re.sub(r"[^\w.\-]+", "_", t)
    return t[:max_len] or "run"


def train_bg_variant(
    tag,
    rel_err,
    *,
    name,
    ablation_panel,
    ablation_tag,
    bg_field_norm=DEFAULT_BG_FIELD_NORM,
    bg_freq_weight=DEFAULT_BG_FREQ_WEIGHT,
    bg_fft_phase_weight=DEFAULT_BG_FFT_PHASE_WEIGHT,
    bg_freq_warmup_epochs=DEFAULT_BG_FREQ_WARMUP,
    bg_split_mode=DEFAULT_BG_SPLIT_MODE,
):
    Xps_rel, sz_ratio, sz_bytes = build_Xps_for_rel(float(rel_err))
    cfg = build_bg_only_cfg(
        X_target=Xs[0],
        Xps=Xps_rel,
        max_train_time=float(BG_TRAIN_TIME),
        bg_h=int(DEFAULT_BG_H),
        roi_h=4,
        epochs=200,
        steps_per_epoch=512,
        bg_patch_size=int(BG_PATCH),
        bg_batch=int(BG_BATCH),
        lr=float(BG_LR),
        bg_field_norm=str(bg_field_norm),
        bg_freq_weight=float(bg_freq_weight),
        bg_fft_phase_weight=float(bg_fft_phase_weight),
        bg_freq_warmup_epochs=int(bg_freq_warmup_epochs),
    )
    cfg.bg_arch = DEFAULT_BG_ARCH
    cfg.bg_split_mode = bg_split_mode
    cfg.bg_split_bands = bg_split_mode in {"two", "three"}
    cfg.bg_split_sigma = 0.12
    cfg.bg_sigma_low = 0.08
    cfg.bg_sigma_mid = 0.18
    cfg.bg_cr_rel_err = float(rel_err)
    cfg.bg_low_weight = 0.25
    cfg.bg_mid_weight = 0.55
    cfg.bg_high_weight = 1.10
    cfg.bg_dyn_band_weight = False
    cfg.bg_band_curriculum = False
    cfg.bg_hard_patch_reweight = False
    cfg.bg_preserve_band_weight_sum = True
    cfg.bg_roi_weight = 0.0
    cfg.bg_random_rel_err = False
    cfg.bg_rel_err_choices = []
    cfg.bg_use_se = bool(globals().get("BG_USE_SE", False))
    cfg.bg_se_reduction = int(globals().get("BG_SE_REDUCTION", 4))

    metric_log = []
    first = [True]

    def evaluator(model):
        if first[0]:
            first[0] = False
            x0 = np.asarray(Xps_rel[0], np.float32)
            m = _global_diag(Xs[0], x0)
            metric_log.append(m)
            return m["psnr"], m["max_err"]
        x_hat = run_bg_inference(model, Xs, Xps_rel, cfg, rel_err)
        m = _global_diag(Xs[0], x_hat)
        metric_log.append(m)
        return m["psnr"], m["max_err"]

    model, hist = train_bg_only(Xs=Xs, Xps=Xps_rel, device=device, cfg=cfg, evaluator=evaluator)

    ckpt_path = None
    if BG_CKPT_ROOT is not None:
        root = Path(BG_CKPT_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        rel_slug = f"{float(rel_err):.0e}".replace("+", "")
        sm = bg_split_mode if bg_split_mode is not None else "none"
        base = (
            f"{_ablation_ckpt_slug(ablation_panel)}__{_ablation_ckpt_slug(ablation_tag)}__"
            f"{_ablation_ckpt_slug(tag)}__rel{rel_slug}__split{_ablation_ckpt_slug(sm)}"
        )
        ckpt_path = root / f"{base}.pt"
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "hist": hist,
                "metric_log": metric_log,
                "meta": {
                    "name": name,
                    "tag": tag,
                    "rel_err": float(rel_err),
                    "ablation_panel": ablation_panel,
                    "ablation_tag": ablation_tag,
                    "bg_arch": getattr(cfg, "bg_arch", None),
                    "bg_field_norm": str(bg_field_norm),
                    "bg_freq_weight": float(bg_freq_weight),
                    "bg_fft_phase_weight": float(bg_fft_phase_weight),
                    "bg_freq_warmup_epochs": int(bg_freq_warmup_epochs),
                    "bg_split_mode": sm,
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "TARGET_STEM": str(globals().get("TARGET_STEM", "")),
                },
            },
            ckpt_path,
        )
        print(f"[ckpt] saved {ckpt_path}")

    return {
        "name": name,
        "cfg": cfg,
        "hist": hist,
        "hint_pretrain_hist": None,
        "metric_log": metric_log,
        "color": "tab:orange",
        "marker": "o",
        "model": model,
        "rel_err": float(rel_err),
        "sz_ratio": float(sz_ratio),
        "sz3_bytes": int(len(sz_bytes)),
        "ablation_panel": ablation_panel,
        "ablation_tag": ablation_tag,
        "ckpt_path": str(ckpt_path) if ckpt_path is not None else None,
    }



FREQ_VARIANTS = [
    ("freq_off", {"bg_freq_weight": 0.0, "bg_fft_phase_weight": 0.0, "bg_freq_warmup_epochs": 1}),
    ("freq_mid", {"bg_freq_weight": 0.5, "bg_fft_phase_weight": 0.55, "bg_freq_warmup_epochs": 1}),
    ("freq_strong", {"bg_freq_weight": 1, "bg_fft_phase_weight": 1, "bg_freq_warmup_epochs": 1}),
]

for tag, rel_err in REL_SETTINGS:
    for vn, kw in FREQ_VARIANTS:
        print(f"\n==== [freq] {vn} | {tag} rel={rel_err} ====")
        results_compare.append(
            train_bg_variant(
                tag, rel_err, name=f"Abl|freq|{vn}|{tag}", ablation_panel="freq", ablation_tag=vn,
                bg_freq_weight=kw["bg_freq_weight"],
                bg_fft_phase_weight=kw["bg_fft_phase_weight"],
                bg_freq_warmup_epochs=kw["bg_freq_warmup_epochs"],
            )
        )

print("results_compare:", len(results_compare))



PSNR_MODE = "best"
INCLUDE_AUX_BYTES = False
NEURLZ_DTYPE_BYTES = 4

x_true = np.asarray(Xs[0], np.float32)
original_target_bytes = int(x_true.nbytes)
aux_bytes = 0
if INCLUDE_AUX_BYTES and len(Xs) > 1:
    aux_bytes = int(sum(np.asarray(f, np.float32).nbytes for f in Xs[1:]))

for r in results_compare:
    patch_sz3_bytes(r)


def rows_for_panel(panel: str):
    rows = []
    for tag, rel in REL_SETTINGS:
        Xps_rel, _, b = build_Xps_for_rel(float(rel))
        xp = np.asarray(Xps_rel[0], np.float32)
        sz3_b = float(len(b))
        tot = sz3_b + aux_bytes
        rows.append({
            "label": "SZ3 only",
            "rel_err": float(rel),
            "effective_cr": float(original_target_bytes / tot),
            "psnr": float(psnr_np(x_true, xp)),
            "ls": "--",
            "c": "black",
        })
    for r in results_compare:
        nm = str(r.get("name", ""))
        if "[CSV]" in nm and TARGET_STEM in nm and "NeurLZ" in nm:
            rel = float(r.get("rel_err", float("nan")))
            if not np.isfinite(rel):
                continue
            sz3_b = float(r["sz3_bytes"])
            nn_b = float(model_param_bytes(r.get("model"), dtype_bytes=NEURLZ_DTYPE_BYTES))
            tot = sz3_b + nn_b + aux_bytes
            rows.append({
                "label": "NeurLZ (CSV)",
                "rel_err": rel,
                "effective_cr": float(original_target_bytes / tot),
                "psnr": float(get_run_psnr(r, mode=PSNR_MODE)),
                "ls": "-",
                "c": "tab:cyan",
            })
    for r in results_compare:
        if r.get("ablation_panel") != panel:
            continue
        rel = float(r.get("rel_err", float("nan")))
        if not np.isfinite(rel):
            continue
        sz3_b = float(r["sz3_bytes"])
        nn_b = float(model_param_bytes(r.get("model"), dtype_bytes=MODEL_DTYPE_BYTES))
        tot = sz3_b + nn_b + aux_bytes
        tag = str(r.get("ablation_tag", "?"))
        rows.append({
            "label": tag,
            "rel_err": rel,
            "effective_cr": float(original_target_bytes / tot),
            "psnr": float(get_run_psnr(r, mode=PSNR_MODE)),
            "ls": "-",
            "c": None,
        })
    return pd.DataFrame(rows)


def plot_panel(panel: str, title: str):
    df = rows_for_panel(panel)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["effective_cr", "psnr"])
    best = df.groupby(["label", "rel_err"], as_index=False).agg({"effective_cr": "mean", "psnr": "max"})
    plt.figure(figsize=(8.5, 5.2), dpi=140)
    cmap = plt.get_cmap("tab10")
    fixed_color = {"SZ3 only": "black", "NeurLZ (CSV)": "tab:cyan"}
    ladder_line_colors = {
        "M0_baseline": "tab:green",
        "M1_plus_freq": "#d62728",
        "M2_plus_freq_split3": "#1f77b4",
    }
    fixed_ls = {"SZ3 only": "--", "NeurLZ (CSV)": "-"}
    labels_order = []
    for pref in ("SZ3 only", "NeurLZ (CSV)"):
        if pref in set(best["label"]):
            labels_order.append(pref)
    rest = sorted([l for l in best["label"].unique() if l not in labels_order])
    labels_order += rest
    for i, lab in enumerate(labels_order):
        g = best[best["label"] == lab].sort_values("effective_cr")
        if g.empty:
            continue
        sub = df[df["label"] == lab].iloc[0]
        ls = fixed_ls.get(lab, str(sub["ls"]))
        co = sub["c"]
        if lab in fixed_color:
            c = fixed_color[lab]
        elif lab in ladder_line_colors:
            c = ladder_line_colors[lab]
        elif co is not None and not pd.isna(co):
            c = co
        else:
            c = cmap(i % 10)
        plt.plot(g["effective_cr"], g["psnr"], marker="o", linestyle=ls, color=c, label=lab, linewidth=2)
    plt.xlabel("Effective CR = original / (SZ3 + NN params)")
    plt.ylabel(f"{PSNR_MODE.capitalize()} global PSNR (dB)")
    plt.title(title + f" | {TARGET_STEM}")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

plot_panel("freq", "Ablation: frequency loss")