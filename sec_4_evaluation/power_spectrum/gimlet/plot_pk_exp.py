"""Figure-13-style P(k) plot extended with the two experiments:

    NeurLZ trained 100 epochs (its own recipe)  -- still over the 1% line at low k
    Ours on an SZ3 base bisected to CR 530      -- effective CR ~498, still under

Same styling as psd.ipynb's Figure 13 (bold 20pt, hollow markers, orange 1%
requirement line). New-series colors are same-family dark variants, validated
(OKLab dE, Machado CVD sim): dark blue #08306b vs tab:blue 22.9 (min CVD 22.2),
dark red #8b0000 vs tab:red 17.3 (min CVD 14.7). The pre-existing tab:gray /
tab:blue pair sits at 13.3 -- the paper's own Figure-13 choice, kept for
consistency; marker shape and line style carry the difference.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT_DIR = "ps3d_out"
K_MAX = 10.0

STYLE = {
    "sz3":          dict(color="#7f7f7f", marker="^", ls="--", ms=13,
                         label="SZ3"),
    "sz3_neurlz":   dict(color="#d62728", marker="v", ls=":", ms=13,
                         label="SZ3+NeurLZ (26 s)"),
    "neurlz_ep100": dict(color="#8b0000", marker="D", ls=(0, (3, 1, 1, 1)), ms=10,
                         label="SZ3+NeurLZ (100 ep, 153 s)"),
    "sz3_ours":     dict(color="#1f77b4", marker="o", ls="-", ms=13,
                         label="SZ3+Ours"),
    "sz3_ours_530": dict(color="#08306b", marker="s", ls="-.", ms=10,
                         label="SZ3+Ours (eff. CR 498)"),
}

ori = np.loadtxt(f"{OUT_DIR}/ori_rhob_ps3d.txt")
k, po = ori[:, 2], ori[:, 3]
mask = k < K_MAX

rel = {}
for tag in STYLE:
    d = np.loadtxt(f"{OUT_DIR}/{tag}_rhob_ps3d.txt")
    assert np.array_equal(ori[:, 1], d[:, 1]), f"{tag}: bins not aligned"
    rel[tag] = np.abs((d[:, 3] - po) / po)

FONTSIZE = 20
fig, ax = plt.subplots(figsize=(8.6, 6.0))
ax.axhline(1.0, color="tab:orange", ls="--", lw=1.8, zorder=1,
           label="Nyx 1% requirement")
for tag, sty in STYLE.items():
    ax.plot(k[mask], rel[tag][mask] * 100, color=sty["color"], marker=sty["marker"],
            ls=sty["ls"], lw=1.8, markerfacecolor="none",
            markeredgecolor=sty["color"], markeredgewidth=1.5,
            markersize=sty["ms"], label=sty["label"], zorder=3)

ax.set_xlabel("k", fontsize=FONTSIZE, fontweight="bold")
ax.set_ylabel("|P(k) relative error| (%)", fontsize=FONTSIZE, fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
ax.grid(True, which="major", alpha=0.3)
ax.set_title("NYX Baryon Density PSD vs k", fontsize=FONTSIZE, pad=12,
             fontweight="bold")
leg = ax.legend(loc="upper right", fontsize=14.5, frameon=False,
                handletextpad=0.4, handlelength=2.6)
plt.tight_layout()
plt.savefig("pk_relerr_exp.pdf", dpi=150, bbox_inches="tight")
plt.savefig("pk_relerr_exp.png", dpi=150, bbox_inches="tight")
print("wrote pk_relerr_exp.pdf / .png")
