"""CR-400 version of the P(k) figure: does an easier base rescue NeurLZ?

Same Figure-13 styling. NeurLZ@100ep is drawn as its default-seed (17) line
with a light band over the three seeds (17/18/19) behind it; SZ3 and Ours are
the CR-400 operating point's own rows.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT_DIR = "ps3d_out"
K_MAX = 10.0
NLZ_RUNS = ["cr400_nlz_ep100", "cr400_nlz_ep100_s18", "cr400_nlz_ep100_s19"]

ori = np.loadtxt(f"{OUT_DIR}/ori_rhob_ps3d.txt")
k, po = ori[:, 2], ori[:, 3]
m = k < K_MAX

def relabs(tag):
    d = np.loadtxt(f"{OUT_DIR}/{tag}_rhob_ps3d.txt")
    assert np.array_equal(ori[:, 1], d[:, 1]), tag
    return np.abs((d[:, 3] - po) / po) * 100

FONTSIZE = 20
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.axhline(1.0, color="tab:orange", ls="--", lw=1.5, zorder=1,
           label="Nyx 1% requirement")

ax.plot(k[m], relabs("cr400_sz3")[m], color="tab:gray", marker="^", ls="--",
        lw=1.8, markerfacecolor="none", markeredgecolor="tab:gray",
        markeredgewidth=1.4, markersize=14, label="SZ3", zorder=3)

nlz = np.vstack([relabs(t) for t in NLZ_RUNS])
ax.fill_between(k[m], nlz.min(0)[m], nlz.max(0)[m], color="tab:red",
                alpha=0.15, lw=0, zorder=2,
                label="SZ3+NeurLZ 100 ep (3 seeds)")
ax.plot(k[m], relabs("cr400_nlz_ep100")[m], color="tab:red", marker="v",
        ls=":", lw=1.8, markerfacecolor="none", markeredgecolor="tab:red",
        markeredgewidth=1.4, markersize=14,
        label="SZ3+NeurLZ 100 ep (seed 17)", zorder=4)

ax.plot(k[m], relabs("cr400_sz3_ours")[m], color="tab:blue", marker="o",
        ls="-", lw=1.8, markerfacecolor="none", markeredgecolor="tab:blue",
        markeredgewidth=1.4, markersize=14, label="SZ3+Ours", zorder=5)

ax.set_xlabel("k", fontsize=FONTSIZE, fontweight="bold")
ax.set_ylabel("|P(k) relative error| (%)", fontsize=FONTSIZE, fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
ax.grid(True, which="major", alpha=0.3)
ax.set_title("NYX Baryon Density PSD vs k (CR 400)", fontsize=FONTSIZE,
             pad=12, fontweight="bold")
leg = ax.legend(loc="upper right", fontsize=14.5, frameon=False,
                handletextpad=0.4, handlelength=2.4)
plt.tight_layout()
plt.savefig("pk_relerr_cr400.pdf", dpi=150, bbox_inches="tight")
plt.savefig("pk_relerr_cr400.png", dpi=150, bbox_inches="tight")
print("wrote pk_relerr_cr400.pdf / .png")
