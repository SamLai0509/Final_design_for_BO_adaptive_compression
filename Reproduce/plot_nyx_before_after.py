"""NYX before/after figure: PSNR, FFT-magnitude and FFT-phase error vs effective CR for the three NYX
fields -- 'before' = the draft's v3 run (lossless siblings, random sampling, lr [1e-4,3e-3], gates on),
'now' = the paper-final run (CR-matched decompressed siblings, shuffled, lr [1e-3,1e-2], no gates).
    python Reproduce/plot_nyx_before_after.py            -> Reproduce/figures/NYX_before_vs_now_psnr_fft.{pdf,png}
"""
import os, glob, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO  = "/home/sam/Halo_Finder/Final_design"
CACHE = f"{REPO}/SPERR/sperr_fft_cache"
OUT   = f"{REPO}/Reproduce/figures"; os.makedirs(OUT, exist_ok=True)
since = float(open(f"{REPO}/Reproduce/logs/RUN_START").read())

FIELDS = [("Baryon", "NYX_baryon_density", "Baryon density"), ("Temp", "NYX_temperature", "Temperature"),
          ("DMD", "NYX_dark_matter_density", "Dark matter density")]
v3 = json.load(open(f"{CACHE}/PAPER_V3_CACHES.json"))


def load(path):
    r = pickle.load(open(path, "rb")); return r.get("results", r) if "results" in r else r


def newest(prefix):
    fs = [f for f in glob.glob(f"{CACHE}/{prefix}__*.pkl") if os.path.getmtime(f) >= since]
    return max(fs, key=os.path.getmtime)


METRICS = [("PSNR", "PSNR (dB)"), ("fft_mag", "FFT magnitude error"), ("fft_phase", "FFT phase error")]
FAM = {"sz3": dict(color="#08519c", base="sz3", ours="pipe", nlz="neurlz", label="SZ3"),
       "sperr": dict(color="#a50f15", base="sperr", ours="sperr_pipe", nlz="sperr_neurlz", label="SPERR")}
OURS_COL = {"sz3": "#4355b9", "sperr": "#e6550d"}
NLZ_COL = {"sz3": "#4292c6", "sperr": "#fdae6b"}
FS = 15

fig, axes = plt.subplots(3, 3, figsize=(17, 12.5))
for r, (key, prefix, title) in enumerate(FIELDS):
    ro, rn = load(f"{CACHE}/{v3[key]}"), load(newest(prefix))
    for c, (mk, ylabel) in enumerate(METRICS):
        ax = axes[r, c]
        for fam, F in FAM.items():
            xy = lambda R, s: (np.asarray(R[s]["CR"], float), np.asarray(R[s][mk], float))
            x, y = xy(rn, F["base"]); ax.plot(x, y, "-o", color=F["color"], lw=2.0, ms=8, mfc="none", mew=1.8, label=f"{F['label']}")
            x, y = xy(rn, F["nlz"]);  ax.plot(x, y, ":^", color=NLZ_COL[fam], lw=1.6, ms=8, mfc="none", mew=1.5, label=f"{F['label']} + NeurLZ")
            x, y = xy(ro, F["ours"]); ax.plot(x, y, "--s", color=OURS_COL[fam], lw=1.6, ms=8, mfc="none", mew=1.5, alpha=0.55, label=f"{F['label']} + Ours (before: lossless siblings)")
            x, y = xy(rn, F["ours"]); ax.plot(x, y, "--s", color=OURS_COL[fam], lw=2.6, ms=9, mfc=OURS_COL[fam], mew=1.8, label=f"{F['label']} + Ours (now: CR-matched siblings)")
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS - 2); ax.set_xticks([100, 200, 300, 400, 500])
        if c > 0:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
        if r == 0:
            ax.set_title(ylabel, fontsize=FS + 2)
        if c == 0:
            ax.set_ylabel(title, fontsize=FS + 2, fontweight="bold")
        if r == 2:
            ax.set_xlabel("Effective Compression Ratio", fontsize=FS)
handles, labels = axes[0, 0].get_legend_handles_labels()
order = [0, 4, 1, 5, 2, 6, 3, 7]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, 0.965),
           ncol=4, fontsize=FS - 1, frameon=False)
fig.suptitle("NYX: draft (v3) vs paper-final run", fontsize=FS + 3, y=1.045)
plt.tight_layout(rect=(0, 0, 1, 0.96))
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/NYX_before_vs_now_psnr_fft.{ext}", dpi=140, bbox_inches="tight"); print("Saved:", f"{OUT}/NYX_before_vs_now_psnr_fft.{ext}")
