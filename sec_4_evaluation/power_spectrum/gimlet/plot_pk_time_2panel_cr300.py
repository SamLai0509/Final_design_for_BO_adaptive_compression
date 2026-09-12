import os
"""CR-300 two-panel figure, spectral-compliance framing.

Left : |P(k) relerr| vs k for each method's FIRST compliant model
       (NeurLZ at 107 s = epoch 70 of its run; Ours at 11 s = 4 epochs).
Right: the requirement metric itself -- max_k |P(k) relerr| -- against pure
       training time, both methods from the SZ3 base, with the 1 % line.
       PSNR is deliberately NOT an axis here: compliance is a spectral
       property and is decoupled from PSNR.
"""
import json, os
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
def mx(tag): return float(relabs(tag)[m].max())

BASE_MAX = mx("cr300_sz3")
# Ours, batch-1 budget-matched runs (seed 17): epochs -> wall time
# wall times = train= field of each run's DONE line
OURS = [(1, 3.12, "cr300_ours_b1ep1"), (2, 5.61, "cr300_ours_b1ep2"),
        (3, 8.13, "cr300_ours_b1ep3"), (4, 10.66, "cr300_ours_b1ep4"),
        (5, 13.1, "cr300_ours_b1ep5"), (6, 15.6, "cr300_ours_b1ep6"),
        (8, 20.6, "cr300_ours_b1ep8"), (10, 26.1, "cr300_sz3_ours")]
OURS = [(e, t, tag) for e, t, tag in OURS if os.path.isfile(f"{OUT_DIR}/{tag}_rhob_ps3d.txt")]
ours_t = [0.0] + [t for _, t, _ in OURS]
ours_y = [BASE_MAX] + [mx(tag) for _, _, tag in OURS]
# NeurLZ checkpoint run (seed 17): every 10 epochs
h = json.load(open(os.environ.get("ADAMIT_RECONS_ROOT", "/path/to/recons") + "/"
                   "recons_exp/NYX_baryon_density__cr300_sz3_neurlz_ep100_results_ckrun.json"))[0]["history"]
t_of = {ep: t for ep, t, _ in h}
nlz_t = [0.0] + [t_of[ep] for ep in range(10, 101, 10)]
nlz_y = [BASE_MAX] + [mx(f"cr300_nlz_ck{ep}") for ep in range(10, 101, 10)]
# best-so-far on the compliance metric (best checkpoint within the budget),
# the same convention as the time-to-PSNR figure: monotone in budget
ours_y = list(np.minimum.accumulate(ours_y))
nlz_y = list(np.minimum.accumulate(nlz_y))
# first compliant points
o_first = next((t, y) for t, y in zip(ours_t[1:], ours_y[1:]) if y <= 1.0)
n_first = next((t, y) for t, y in zip(nlz_t[1:], nlz_y[1:]) if y <= 1.0)
print("Ours first <=1%:", o_first, "| NeurLZ first <=1%:", n_first)

FS = 19
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.2, 5.6))

axL.axhline(1.0, color="tab:orange", ls="--", lw=1.5, zorder=1, label="Nyx 1% requirement")
axL.plot(k[m], relabs("cr300_sz3")[m], color="tab:gray", marker="^", ls="--", lw=1.8,
         markerfacecolor="none", markeredgewidth=1.4, markersize=12, label="SZ3", zorder=3)
axL.plot(k[m], relabs("cr300_nlz_ck70")[m], color="tab:red", marker="v", ls=":", lw=1.8,
         markerfacecolor="none", markeredgecolor="tab:red", markeredgewidth=1.4,
         markersize=12, label="SZ3+NeurLZ (107 s)", zorder=4)
o_tag = next(tag for _, t, tag in OURS if mx(tag) <= 1.0)
axL.plot(k[m], relabs(o_tag)[m], color="tab:blue", marker="o", ls="-", lw=1.8,
         markerfacecolor="none", markeredgecolor="tab:blue", markeredgewidth=1.4,
         markersize=12, label=f"SZ3+Ours ({o_first[0]:.0f} s)", zorder=5)
axL.set_xlabel("k", fontsize=FS, fontweight="bold")
axL.set_ylabel(r"$\varepsilon(k)$ (%)", fontsize=FS, fontweight="bold")
axL.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
axL.tick_params(axis="both", which="major", labelsize=FS - 2)
axL.grid(True, which="major", alpha=0.3)
axL.set_title(r"Power-spectrum error $\varepsilon(k)$", fontsize=FS + 1, pad=10, fontweight="bold")
axL.legend(loc="upper right", fontsize=FS - 6, frameon=False, handletextpad=0.4)

axR.axhline(1.0, color="tab:orange", ls="--", lw=1.5, zorder=1, label="Nyx 1% requirement")
axR.plot(nlz_t, nlz_y, color="tab:red", ls=":", lw=1.8, marker="v", markersize=10,
         markerfacecolor="none", markeredgecolor="tab:red", markeredgewidth=1.3,
         label=f"SZ3+NeurLZ: {n_first[0]:.0f} s", zorder=3)
axR.plot([n_first[0]], [n_first[1]], marker="v", color="tab:red", markersize=15,
         markerfacecolor="tab:red", ls="none", zorder=5)
axR.plot(ours_t, ours_y, color="tab:blue", ls="-", lw=2.0, marker="o", markersize=10,
         markerfacecolor="none", markeredgecolor="tab:blue", markeredgewidth=1.5,
         label=f"SZ3+Ours: {o_first[0]:.0f} s", zorder=5)
axR.plot([o_first[0]], [o_first[1]], marker="o", color="tab:blue", markersize=13,
         markerfacecolor="tab:blue", ls="none", zorder=6)
axR.set_xlim(-4, 162)
axR.set_ylim(0, 2.9)
axR.set_xlabel("Pure training wall time (s)", fontsize=FS, fontweight="bold")
axR.set_ylabel(r"best $\max_k \varepsilon(k)$ (%)", fontsize=FS, fontweight="bold")
axR.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
axR.tick_params(axis="both", which="major", labelsize=FS - 2)
axR.grid(True, which="major", alpha=0.3)
axR.set_title("Time to meet the 1% requirement", fontsize=FS + 1, pad=10, fontweight="bold")
axR.legend(loc="upper right", fontsize=FS - 4, frameon=False, handletextpad=0.4)

fig.suptitle("NYX Baryon Density @ CR 300", fontsize=FS + 2, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("pk_time_2panel_cr300.pdf", dpi=150, bbox_inches="tight")
plt.savefig("pk_time_2panel_cr300.png", dpi=150, bbox_inches="tight")
print("wrote pk_time_2panel_cr300.pdf / .png")
