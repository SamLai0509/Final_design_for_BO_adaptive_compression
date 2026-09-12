"""NYX: PSNR / FFT-magnitude / FFT-phase vs effective CR, siblings WITH vs WITHOUT the three velocity fields.
'with' = paper-final run (CR-matched siblings: 5 fields), 'without' = run_novel_6000.sh (2 siblings).
    python Reproduce/experiment/plot_velocity_ablation.py -> Reproduce/experiment/nyx_with_vs_without_velocity.{pdf,png}
"""
import os, json, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); E = f"{REPO}/Reproduce/experiment"
pins = json.load(open(f"{E}/EXPERIMENT_PINS.json"))
WITH, WITHOUT = pins["reference (CR-matched siblings)"], pins["no velocity siblings"]
def load(f):
    _p = next((q for q in (f"{REPO}/sec_4_evaluation/sperr_fft_cache/{f}", f"{E}/pkl/{f}") if os.path.exists(q)), None)
    if _p is None: raise FileNotFoundError(f"{f}: not in sec_4_evaluation/sperr_fft_cache/ or Reproduce/experiment/pkl/")
    r = pickle.load(open(_p, "rb")); return r.get("results", r) if "results" in r else r
FIELDS = [("Baryon", "Baryon density"), ("Temp", "Temperature"), ("DMD", "Dark matter density")]
METRICS = [("PSNR", "PSNR (dB)"), ("fft_mag", "FFT magnitude error"), ("fft_phase", "FFT phase error")]
FAM = {"sz3": ("SZ3", "#08519c", "#4355b9", "sz3", "pipe"), "sperr": ("SPERR", "#a50f15", "#e6550d", "sperr", "sperr_pipe")}
FS = 15
fig, axes = plt.subplots(3, 3, figsize=(17, 12.5))
for r, (key, title) in enumerate(FIELDS):
    rw, rn = load(WITH[key]), load(WITHOUT[key])
    for c, (mk, ylabel) in enumerate(METRICS):
        ax = axes[r, c]
        for fam, (lab, cb, co, b, p) in FAM.items():
            xy = lambda R, s: (np.asarray(R[s]["CR"], float), np.asarray(R[s][mk], float))
            x, y = xy(rw, b); ax.plot(x, y, "-o", color=cb, lw=2.0, ms=8, mfc="none", mew=1.8, label=lab)
            x, y = xy(rw, p); ax.plot(x, y, "--s", color=co, lw=2.6, ms=9, mfc=co, mew=1.8, label=f"{lab} + Ours, 5 siblings (with velocity)")
            x, y = xy(rn, p); ax.plot(x, y, ":D", color=co, lw=2.0, ms=9, mfc="none", mew=2.0, label=f"{lab} + Ours, 2 siblings (no velocity)")
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS - 2); ax.set_xticks([100, 200, 300, 400, 500])
        if c > 0: ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
        if r == 0: ax.set_title(ylabel, fontsize=FS + 2)
        if c == 0: ax.set_ylabel(title, fontsize=FS + 2, fontweight="bold")
        if r == 2: ax.set_xlabel("Effective Compression Ratio", fontsize=FS)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.965), ncol=3, fontsize=FS - 1, frameon=False)
fig.suptitle("NYX: siblings with vs. without the velocity fields (paper protocol otherwise)", fontsize=FS + 3, y=1.04)
plt.tight_layout(rect=(0, 0, 1, 0.96))
for ext in ("pdf", "png"):
    fig.savefig(f"{E}/nyx_with_vs_without_velocity.{ext}", dpi=140, bbox_inches="tight"); print("Saved:", f"{E}/nyx_with_vs_without_velocity.{ext}")
