"""Per-CR version (pass --cr)  of the P(k) figure: does an easier base rescue NeurLZ?

Same Figure-13 styling. NeurLZ@100ep is drawn as its default-seed (17) line
with a light band over the three seeds (17/18/19) behind it; SZ3 and Ours are
the CR-400 operating point's own rows.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ap = argparse.ArgumentParser(); ap.add_argument("--cr", type=int, required=True)
CR = ap.parse_args().cr
P = f"cr{CR}_"
OUT_DIR = "ps3d_out"
K_MAX = 10.0
NLZ_RUNS = [f"{P}nlz_ep100", f"{P}nlz_ep100_s18", f"{P}nlz_ep100_s19"]

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

ax.plot(k[m], relabs(f"{P}sz3")[m], color="tab:gray", marker="^", ls="--",
        lw=1.8, markerfacecolor="none", markeredgecolor="tab:gray",
        markeredgewidth=1.4, markersize=14, label="SZ3", zorder=3)

ax.plot(k[m], relabs(f"{P}nlz_ep100")[m], color="tab:red", marker="v",
        ls=":", lw=1.8, markerfacecolor="none", markeredgecolor="tab:red",
        markeredgewidth=1.4, markersize=14,
        label="SZ3+NeurLZ (100 ep)", zorder=4)

ax.plot(k[m], relabs(f"{P}sz3_ours")[m], color="tab:blue", marker="o",
        ls="-", lw=1.8, markerfacecolor="none", markeredgecolor="tab:blue",
        markeredgewidth=1.4, markersize=14, label="SZ3+Ours", zorder=5)

ax.set_xlabel("k", fontsize=FONTSIZE, fontweight="bold")
ax.set_ylabel("|P(k) relative error| (%)", fontsize=FONTSIZE, fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
ax.grid(True, which="major", alpha=0.3)
ax.set_title(f"NYX Baryon Density PSD vs k (CR {CR})", fontsize=FONTSIZE,
             pad=12, fontweight="bold")
leg = ax.legend(loc="upper right", fontsize=14.5, frameon=False,
                handletextpad=0.4, handlelength=2.4)
plt.tight_layout()
plt.savefig(f"pk_relerr_cr{CR}.pdf", dpi=150, bbox_inches="tight")
plt.savefig(f"pk_relerr_cr{CR}.png", dpi=150, bbox_inches="tight")
print(f"wrote pk_relerr_cr{CR}.pdf / .png")
