"""NYX: PSNR / FFT-magnitude / FFT-phase vs effective CR for the three sibling orders:
   reference = every sibling merely decompressed (paper protocol),
   cascade A = DMD -> baryon -> temperature (later stages get ENHANCED siblings),
   cascade B = DMD -> temperature -> baryon.   DMD is stage 1 in both cascades (identical run).
    python Reproduce/experiment/plot_cascade_orders.py -> Reproduce/experiment/nyx_cascade_orders.{pdf,png}
"""
import os, json, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); E = f"{REPO}/Reproduce/experiment"
pins = json.load(open(f"{E}/EXPERIMENT_PINS.json"))
REF = pins["reference (CR-matched siblings)"]
CA = pins["cascade A: DMD -> baryon -> temperature (enhanced siblings)"]
CB = pins["cascade B: DMD -> temperature -> baryon (enhanced siblings)"]
def load(f):
    r = pickle.load(open(f"{E}/pkl/{f}", "rb")); return r.get("results", r) if "results" in r else r
FIELDS = [("Baryon", "Baryon density"), ("Temp", "Temperature"), ("DMD", "Dark matter density")]
METRICS = [("PSNR", "PSNR (dB)"), ("fft_mag", "FFT magnitude error"), ("fft_phase", "FFT phase error")]
FAM = {"sz3": ("SZ3", "#08519c", "#4355b9", "sz3", "pipe"), "sperr": ("SPERR", "#a50f15", "#e6550d", "sperr", "sperr_pipe")}
FS = 15
fig, axes = plt.subplots(3, 3, figsize=(17, 12.5))
for r, (key, title) in enumerate(FIELDS):
    runs = [("reference (decompressed siblings)", load(REF[key]), "--s", True)]
    if key == "DMD":
        runs.append(("cascade stage 1 (same in A and B)", load(CA[key]), "-.^", False))
    else:
        runs.append(("cascade A: DMD → baryon → temperature", load(CA[key]), "-.^", False))
        runs.append(("cascade B: DMD → temperature → baryon", load(CB[key]), ":D", False))
    for c, (mk, ylabel) in enumerate(METRICS):
        ax = axes[r, c]
        for fam, (lab, cb, co, b, p) in FAM.items():
            x, y = np.asarray(runs[0][1][b]["CR"], float), np.asarray(runs[0][1][b][mk], float)
            ax.plot(x, y, "-o", color=cb, lw=2.0, ms=8, mfc="none", mew=1.8, label=lab)
            for name, R, style, filled in runs:
                x, y = np.asarray(R[p]["CR"], float), np.asarray(R[p][mk], float)
                ax.plot(x, y, style, color=co, lw=2.4 if filled else 2.0, ms=9, mfc=co if filled else "none", mew=1.9, label=f"{lab} + Ours — {name}")
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS - 2); ax.set_xticks([100, 200, 300, 400, 500])
        if c > 0: ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
        if r == 0: ax.set_title(ylabel, fontsize=FS + 2)
        if c == 0: ax.set_ylabel(title, fontsize=FS + 2, fontweight="bold")
        if r == 2: ax.set_xlabel("Effective Compression Ratio", fontsize=FS)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.955), ncol=2, fontsize=FS - 2, frameon=False)
fig.suptitle("NYX: cross-field cascade orders vs. the paper protocol (DMD row: stage 1 is shared)", fontsize=FS + 3, y=1.075)
plt.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ("pdf", "png"):
    fig.savefig(f"{E}/nyx_cascade_orders.{ext}", dpi=140, bbox_inches="tight"); print("Saved:", f"{E}/nyx_cascade_orders.{ext}")
