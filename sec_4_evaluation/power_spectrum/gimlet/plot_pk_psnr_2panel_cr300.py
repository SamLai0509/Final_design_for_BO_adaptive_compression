import os

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT_DIR = "ps3d_out"
ori = np.loadtxt(f"{OUT_DIR}/ori_rhob_ps3d.txt")
k, po = ori[:, 2], ori[:, 3]
m = k < 10.0
def relabs(tag):
    d = np.loadtxt(f"{OUT_DIR}/{tag}_rhob_ps3d.txt")
    return np.abs((d[:, 3] - po) / po) * 100

BASE = 116.34
ours_t = [0.0, 2.83, 5.35, 7.87, 10.39]
ours_p = [BASE, 124.64, 124.99, 126.75, 127.02]
h = json.load(open(os.environ.get("ADAMIT_RECONS_ROOT", "/path/to/recons") + "/"
                   "recons_exp/NYX_baryon_density__cr300_sz3_neurlz_ep100_results_ckrun.json"))[0]["history"]
nlz_t = [0.0] + [t for _, t, _ in h]
nlz_p = [BASE] + [p for _, _, p in h]
NLZ_REACH = (106.8, 119.936)

FS = 19
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.2, 5.6))

axL.axhline(1.0, color="tab:orange", ls="--", lw=1.5, zorder=1,
            label="Nyx 1% requirement")
axL.plot(k[m], relabs("cr300_sz3")[m], color="tab:gray", marker="^", ls="--",
         lw=1.8, markerfacecolor="none", markeredgewidth=1.4, markersize=12,
         label="SZ3", zorder=3)
axL.plot(k[m], relabs("cr300_nlz_ck70")[m], color="tab:red", marker="v", ls=":",
         lw=1.8, markerfacecolor="none", markeredgecolor="tab:red",
         markeredgewidth=1.4, markersize=12, label="SZ3+NeurLZ (107 s)", zorder=4)
axL.plot(k[m], relabs("cr300_ours_b1ep4")[m], color="tab:blue", marker="o",
         ls="-", lw=1.8, markerfacecolor="none", markeredgecolor="tab:blue",
         markeredgewidth=1.4, markersize=12, label="SZ3+Ours (11 s)", zorder=5)
axL.set_xlabel("k", fontsize=FS, fontweight="bold")
axL.set_ylabel("|P(k) relative error| (%)", fontsize=FS, fontweight="bold")
axL.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
axL.tick_params(axis="both", which="major", labelsize=FS - 2)
axL.grid(True, which="major", alpha=0.3)
axL.set_title("PSD relative error vs k", fontsize=FS + 1, pad=10, fontweight="bold")
axL.legend(loc="upper right", fontsize=FS - 6, frameon=False, handletextpad=0.4)

axR.axhline(BASE, color="tab:gray", ls="--", lw=1.6, zorder=1)
axR.text(158, BASE - 0.35, "SZ3 base", fontsize=FS - 6, color="tab:gray",
         ha="right", va="top")
axR.plot(nlz_t, nlz_p, color="tab:red", ls=":", lw=1.8, marker="v",
         markersize=8, markerfacecolor="none", markeredgecolor="tab:red",
         markeredgewidth=1.2, label="SZ3+NeurLZ: 107 s", zorder=3)
axR.plot([NLZ_REACH[0]], [NLZ_REACH[1]], marker="v", color="tab:red",
         markersize=14, markerfacecolor="tab:red", ls="none", zorder=5)
axR.plot(ours_t, ours_p, color="tab:blue", ls="-", lw=2.0, marker="o",
         markersize=10, markerfacecolor="none", markeredgecolor="tab:blue",
         markeredgewidth=1.5, label="SZ3+Ours: 11 s", zorder=5)
axR.plot([ours_t[-1]], [ours_p[-1]], marker="o", color="tab:blue",
         markersize=12, markerfacecolor="tab:blue", ls="none", zorder=6)
axR.set_xlim(-4, 162)
axR.set_xlabel("Pure training wall time (s)", fontsize=FS, fontweight="bold")
axR.set_ylabel("Global PSNR (dB)", fontsize=FS, fontweight="bold")
axR.tick_params(axis="both", which="major", labelsize=FS - 2)
axR.grid(True, which="major", alpha=0.3)
axR.set_title("PSNR vs training time", fontsize=FS + 1, pad=10, fontweight="bold")
axR.set_ylim(114.8, 128.8)
axR.legend(loc="upper right", bbox_to_anchor=(0.99, 0.90), fontsize=FS - 4,
           frameon=False, handletextpad=0.4)

fig.suptitle("NYX Baryon Density @ CR 300", fontsize=FS + 2, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("pk_psnr_2panel_cr300.pdf", dpi=150, bbox_inches="tight")
plt.savefig("pk_psnr_2panel_cr300.png", dpi=150, bbox_inches="tight")
print("wrote")
