"""|P(k) relative error| with the NeurLZ@100ep run spread, Figure-13 styling.

Same look as pk_relerr_cr500_k_lt_10.pdf (bold 20pt, hollow markers, single
orange 1% requirement line, |relerr| on y). The NeurLZ 100-epoch recipe is
shown as a min-max band over its four nondeterministic runs (seeds 17, 17
repeated, 18, 19) plus the bit-reproducible deterministic run as its own
line; Ours is the median of three independent runs (eff. CR 481/498/509)
with a min-max band that is too tight to see at this scale.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT_DIR = "ps3d_out"
K_MAX = 10.0

import os
# every nondeterministic recipe run whose spectrum exists on disk
_CAND = (["neurlz_ep100", "neurlz_ep100_r2", "neurlz_ep100_r3"]
         + [f"neurlz_ep100_s{i}" for i in range(18, 30)])
NEURLZ_RUNS = [t for t in _CAND if os.path.isfile(f"{OUT_DIR}/{t}_rhob_ps3d.txt")]
NEURLZ_DET = "neurlz_ep100_det"
# True  -> exploration figure: min-max band over every recipe run + the
#          deterministic run, written to pk_relerr_spread.pdf
# False -> paper figure: NeurLZ shown as ONE line, the latest seed-17 run
#          (r3), in Figure-13's own NeurLZ style; pk_relerr_spread_nodet.pdf
SHOW_DET = True
OUT_STEM = "pk_relerr_spread" if SHOW_DET else "pk_relerr_spread_nodet"
NEURLZ_LATEST = "neurlz_ep100_r3"
OURS_RUNS = ["sz3_ours", "sz3_ours_530", "sz3_ours_540"]

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

ax.plot(k[m], relabs("sz3")[m], color="tab:gray", marker="^", ls="--", lw=1.8,
        markerfacecolor="none", markeredgecolor="tab:gray", markeredgewidth=1.4,
        markersize=14, label="SZ3", zorder=3)

if SHOW_DET:
    nlz = np.vstack([relabs(t) for t in NEURLZ_RUNS])
    ax.fill_between(k[m], nlz.min(0)[m], nlz.max(0)[m], color="tab:red",
                    alpha=0.18, lw=0, zorder=2,
                    label=f"SZ3+NeurLZ 100 ep ({len(NEURLZ_RUNS)} runs)")
    ax.plot(k[m], relabs(NEURLZ_DET)[m], color="#8b0000", marker="D",
            ls=(0, (3, 1, 1, 1)), lw=1.8, markerfacecolor="none",
            markeredgecolor="#8b0000", markeredgewidth=1.4, markersize=10,
            label="SZ3+NeurLZ 100 ep (determ.)", zorder=4)
else:
    ax.plot(k[m], relabs(NEURLZ_LATEST)[m], color="tab:red", marker="v",
            ls=":", lw=1.8, markerfacecolor="none", markeredgecolor="tab:red",
            markeredgewidth=1.4, markersize=14,
            label="SZ3+NeurLZ (100 ep, 153 s)", zorder=4)

ours = np.vstack([relabs(t) for t in OURS_RUNS])
ax.fill_between(k[m], ours.min(0)[m], ours.max(0)[m], color="tab:blue",
                alpha=0.25, lw=0, zorder=3)
ax.plot(k[m], np.median(ours, 0)[m], color="tab:blue", marker="o", ls="-",
        lw=1.8, markerfacecolor="none", markeredgecolor="tab:blue",
        markeredgewidth=1.4, markersize=14,
        label="SZ3+Ours (26 s)", zorder=5)

ax.set_xlabel("k", fontsize=FONTSIZE, fontweight="bold")
ax.set_ylabel("|P(k) relative error| (%)", fontsize=FONTSIZE, fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
ax.grid(True, which="major", alpha=0.3)
ax.set_title("NYX Baryon Density PSD vs k", fontsize=FONTSIZE, pad=12,
             fontweight="bold")
leg = ax.legend(loc="upper right", fontsize=15, frameon=False,
                handletextpad=0.4, handlelength=2.4)
plt.tight_layout()
plt.savefig(f"{OUT_STEM}.pdf", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_STEM}.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT_STEM}.pdf / .png")
