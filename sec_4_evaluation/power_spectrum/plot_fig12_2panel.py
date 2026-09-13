"""Figure 12, two panels, from a power_spectrum_fig12.py data directory.

Left : eps(k) of SZ3, SZ3+NeurLZ and SZ3+Ours at the point each first meets
       the Nyx 1% requirement.
Right: the requirement metric itself -- best max_k eps(k) so far -- against
       pure training wall time, both curves starting from the SZ3 base.

Data directories: figures/fig12_b1_s17 holds the AdaMit run behind the
figure (batch 1, enhanced siblings, base CR 317.4 -> effective 306.7);
figures/fig12_neurlz holds the NeurLZ branch (its own recipe, 100 epochs,
CR-matched siblings -- NeurLZ has no cascade), passed via --neurlz-data.

Knobs below are kept identical to plot_pk_time_2panel_cr300.ipynb so the two
figures render the same; only YLIM_R differs (that data reaches 2.9 %, this
tops out at the 1.6 % base, so a fixed 2.9 would leave half the panel empty).
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="figures/fig12_b1_s17")
ap.add_argument("--neurlz-data", dest="nl_data", default=None,
                help="take the NeurLZ branch from a different run, e.g. when "
                     "Ours was rerun at its own base CR to align the effective "
                     "ratios. Each method then sits at its own base, which is "
                     "the same convention the results table uses.")
ap.add_argument("--out", default=None)
a = ap.parse_args()
d = json.load(open(f"{a.data}/fig12_data.json"))
if a.nl_data:
    dn = json.load(open(f"{a.nl_data}/fig12_data.json"))
    d["neurlz"], d["neurlz_aux_mode"] = dn["neurlz"], dn["neurlz_aux_mode"]
    # the NeurLZ curve must start from ITS OWN base, not this run's
    NL_BASE = max(dn["eps_base"])
else:
    NL_BASE = None
OUT = a.out or f"{a.data}/fig12_2panel.pdf"

REQ = 1.0
k = np.array(d["k"], dtype=float)
BASE_MAX = max(d["eps_base"])

def first_compliant(o, base=None):
    """(times, best-so-far max eps, index of first compliant checkpoint).

    The accumulation starts from the untrained state, so the curve begins at
    the SZ3 base and is monotone in budget: a checkpoint worse than what the
    run already had does not raise the reported error.
    """
    b = BASE_MAX if base is None else base
    t = np.r_[0.0, np.array(o["time"], dtype=float)]
    y = np.minimum.accumulate(
        np.r_[b, np.array(o["max_eps_trace"], dtype=float)])
    i = int(np.argmax(y <= REQ)) if (y <= REQ).any() else len(y) - 1
    return t, y, i

t_o, y_o, i_o = first_compliant(d["ours"])
t_n, y_n, i_n = first_compliant(d["neurlz"], NL_BASE)
eps_o = np.array(d["ours"]["eps_trace"][i_o - 1], dtype=float)
eps_n = np.array(d["neurlz"]["eps_trace"][i_n - 1], dtype=float)

print(f"CR_base {d['cr_base']:.1f} | siblings: ours={d['aux_mode']}, "
      f"neurlz={d['neurlz_aux_mode']}")
print(f"SZ3    max eps {BASE_MAX:.3f}%")
print(f"Ours   first <=1% at {t_o[i_o]:6.2f}s  max {y_o[i_o]:.3f}%  "
      f"PSNR {d['ours']['psnr_trace'][i_o - 1]:.2f}")
print(f"NeurLZ first <=1% at {t_n[i_n]:6.2f}s  max {y_n[i_n]:.3f}%  "
      f"PSNR {d['neurlz']['psnr_trace'][i_n - 1]:.2f}")

# ── TEXT ────────────────────────────────────────────────────────────────────
# effective CR: the corrector weights are part of the payload, and each
# method sits at its own base so the effective ratios line up across methods
_RAW, _W = 512 ** 3 * 4, 29803 * 2
EFF_CR = _RAW / (_RAW / d["cr_base"] + _W)
SUPTITLE  = f"NYX Baryon Density @ CR {EFF_CR:.0f}"
TITLE_L   = r"Power-spectrum error $\varepsilon(k)$"
TITLE_R   = "Time to meet the 1% requirement"
XLABEL_L  = "k"
YLABEL_L  = r"$\varepsilon(k)$ (%)"
XLABEL_R  = "Pure training wall time (s)"
YLABEL_R  = r"best $\max_k \varepsilon(k)$ (%)"
LAB_REQ   = "Nyx 1% requirement"
LAB_SZ3   = "SZ3"
LAB_NLZ   = "SZ3+NeurLZ"      # " ({t:.0f} s)" is appended from the data
LAB_OURS  = "SZ3+Ours"        # ditto

# ── LAYOUT / FONTS (the knobs) ──────────────────────────────────────────────
FIGSIZE    = (9.5, 4)
FS         = 15          # base; everything else is FS +/- n
FS_TITLE   = FS + 1
FS_SUP     = FS + 2
FS_TICK    = FS
FS_LEG_L   = FS - 3
FS_LEG_R   = FS - 2
BOLD_LABELS = True
XLIM_R     = None        # auto: curves end at their compliance points
YLIM_R     = (0, 2.0)    # headroom so the legend clears the curves
LEG_LOC_L, LEG_LOC_R = "upper right", "upper right"

# ── STYLE ───────────────────────────────────────────────────────────────────
C_SZ3, C_NLZ, C_OURS, C_REQ = "tab:gray", "tab:red", "tab:blue", "tab:orange"
LW, MS = 1.8, 12
MS_R      = 10
MS_FIRST  = 14
MEW       = 1.4
GRID_ALPHA = 0.3

plt.rcdefaults()
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.weight": "normal", "axes.labelweight": "normal",
    "axes.titleweight": "normal", "figure.dpi": 100,
    "savefig.facecolor": "white",
})
BW = "bold" if BOLD_LABELS else "normal"
fig, (axL, axR) = plt.subplots(1, 2, figsize=FIGSIZE)

axL.axhline(REQ, color=C_REQ, ls="--", lw=1.5, zorder=1, label=LAB_REQ)
axL.plot(k, d["eps_base"], color=C_SZ3, marker="^", ls="--", lw=LW,
         markerfacecolor="none", markeredgewidth=MEW, markersize=MS,
         label=LAB_SZ3, zorder=3)
axL.plot(k, eps_n, color=C_NLZ, marker="v", ls=":", lw=LW, markerfacecolor="none",
         markeredgecolor=C_NLZ, markeredgewidth=MEW, markersize=MS,
         label=f"{LAB_NLZ} ({t_n[i_n]:.0f} s)", zorder=4)
axL.plot(k, eps_o, color=C_OURS, marker="o", ls="-", lw=LW, markerfacecolor="none",
         markeredgecolor=C_OURS, markeredgewidth=MEW, markersize=MS,
         label=f"{LAB_OURS} ({t_o[i_o]:.0f} s)", zorder=5)
axL.set_xlabel(XLABEL_L, fontsize=FS, fontweight=BW)
axL.set_ylabel(YLABEL_L, fontsize=FS, fontweight=BW)
axL.set_title(TITLE_L, fontsize=FS_TITLE, pad=10, fontweight=BW)
axL.legend(loc=LEG_LOC_L, fontsize=FS_LEG_L, frameon=False, handletextpad=0.4)

axR.axhline(REQ, color=C_REQ, ls="--", lw=1.5, zorder=1, label=LAB_REQ)
axR.plot(t_n[:i_n + 1], y_n[:i_n + 1], color=C_NLZ, ls=":", lw=LW, marker="v",
         markersize=MS_R - 3, markerfacecolor="none", markeredgecolor=C_NLZ,
         markeredgewidth=1.2, markevery=5,
         label=f"{LAB_NLZ}: {t_n[i_n]:.0f} s", zorder=3)
axR.plot([t_n[i_n]], [y_n[i_n]], marker="v", color=C_NLZ, ls="none",
         markersize=MS_FIRST, markerfacecolor=C_NLZ, zorder=5)
axR.plot(t_o[:i_o + 1], y_o[:i_o + 1], color=C_OURS, ls="-", lw=LW + 0.2,
         marker="o", markersize=MS_R - 2, markerfacecolor="none",
         markeredgecolor=C_OURS, markeredgewidth=1.5,
         label=f"{LAB_OURS}: {t_o[i_o]:.0f} s", zorder=5)
axR.plot([t_o[i_o]], [y_o[i_o]], marker="o", color=C_OURS, ls="none",
         markersize=MS_FIRST - 2, markerfacecolor=C_OURS, zorder=6)
if XLIM_R: axR.set_xlim(*XLIM_R)
if YLIM_R: axR.set_ylim(*YLIM_R)
axR.set_xlabel(XLABEL_R, fontsize=FS, fontweight=BW)
axR.set_ylabel(YLABEL_R, fontsize=FS, fontweight=BW)
axR.set_title(TITLE_R, fontsize=FS_TITLE, pad=10, fontweight=BW)
axR.legend(loc=LEG_LOC_R, fontsize=FS_LEG_R, frameon=False, handletextpad=0.4)

for ax in (axL, axR):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
    ax.grid(True, which="major", alpha=GRID_ALPHA)

fig.suptitle(SUPTITLE, fontsize=FS_SUP, fontweight=BW, y=0.99)
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
