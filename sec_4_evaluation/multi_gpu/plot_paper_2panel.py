"""The paper's two-panel PSNR-vs-time figure, from the corrected runs.

Left  SZ3 + Ours on NYX baryon density -- redone: the five auxiliary channels
      are the archived decompressed siblings, not the originals.
Right SPERR + Ours on Miranda -- single-field, so no siblings and nothing to
      correct; re-measured only so both panels come from one code path.

The dashed line is the single-GPU delivered PSNR, i.e. the peak of its curve,
because training returns the best-PSNR weights rather than the last epoch's.
Legend times are pure training: the single GPU's budget, and the point where
four GPUs first clear the same PSNR.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcdefaults()
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "font.weight": "normal", "axes.labelweight": "normal",
                     "axes.titleweight": "normal"})

OUT = Path("psnr_vs_walltime_2panel.pdf")
C1, C4 = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d9d6"
FS = 15
# No tolerance: the results table's iso-budget rows interpolate the crossing
# exactly, and a 0.05 dB slack moves it by seconds wherever the four-GPU curve
# is flat near the target (NYX baryon: 5.0 s with slack, 6.9 s without).
TOL = 0.0

# (directory, tag stem, title). NYX comes from the enhanced-sibling runs, the
# protocol the results table reports; Miranda is single-field -- no siblings to
# get wrong -- and is read from the same runs the table's Miranda row uses, so
# figure and table cannot drift apart.
# Final protocol, same runs as the table: single GPU = native batch-1 config,
# 4 GPUs = local SGD (independent batch-1 replicas, weights averaged every 16
# steps). Panels are the table's SZ3 NYX-baryon and SPERR Miranda rows.
# Confirmed protocol: single GPU batch-1; 4 GPUs DDP (1/rank, gradient
# averaging), the paper's Section-4.4 scheme, budget-fixed runs.
# FINAL protocol (= the paper's Section 4.4 and Table 4): iso-batch, single
# GPU batch-4, four GPUs DDP with batch 1 per rank, budget-fixed runs.
PANELS = [("bench_out/table_nyx_enh_r2/nyx_b_sz3_n1", "bench_out/table_n4_fixed/nyx_b_sz3_n4",
           10.0, "SZ3 + Ours — NYX Baryon Density"),
          ("bench_out/table_mir_fixlr_n1/miranda_sperr_n1", "bench_out/table_mir_fixlr_n4/miranda_sperr_n4",
           80.0, "SPERR + Ours — Miranda")]


def reach(r, q):
    t, p = r["time"], np.maximum.accumulate(r["psnr"])
    for i in range(1, len(p)):
        if p[i] >= q - TOL:
            if p[i] == p[i - 1]:
                return float(t[i])
            f = (q - TOL - p[i - 1]) / (p[i] - p[i - 1])
            return float(t[i - 1] + f * (t[i] - t[i - 1]))
    return None


fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
for ax, (f1, f4, budget, title) in zip(axes, PANELS):
    a = json.loads(Path(f"{f1}.json").read_text())
    b = json.loads(Path(f"{f4}.json").read_text())
    # the target is what the single GPU delivers when its prescribed budget
    # expires -- the same anchor the results table uses, so the two agree
    tt1 = np.asarray(a["time"], dtype=float)
    pp1 = np.maximum.accumulate(np.asarray(a["psnr"], dtype=float))
    t1 = min(budget, float(tt1[-1]))
    target = float(np.interp(t1, tt1, pp1))
    t4 = reach(b, target)

    ax.axhline(target, color="#3a3a3a", ls=(0, (5, 4)), lw=1.0, zorder=2)
    for r, c, ls, mk, lab, tt in ((b, C4, "--", "s", "4 GPUs", t4),
                                  (a, C1, "-", "o", "Single GPU", t1)):
        # the curve ends where it meets the target: everything after the
        # crossing is plateau and only clutters the panel
        rt = np.asarray(r["time"], dtype=float)
        rp = np.asarray(r["psnr"], dtype=float)
        keep = rt <= tt
        xs = np.r_[rt[keep], tt]
        ys = np.r_[rp[keep], target]
        ax.plot(xs, ys, color=c, ls=ls, lw=1.8, marker=mk,
                markersize=8, markerfacecolor="none", markeredgewidth=1.6,
                markevery=list(range(len(xs) - 1)),
                zorder=4, label=f"{lab} — {tt:.1f} s")
        ax.plot([tt], [target], marker=mk, color=c, markersize=12, zorder=5)
        ax.axvline(tt, color=MUTED, lw=0.7, ls=":", alpha=0.55, zorder=1)

    lo = min(min(a["psnr"]), min(b["psnr"]))
    hi = max(max(a["psnr"]), max(b["psnr"]))
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.2 * pad)
    ax.set_xlim(0, max(t1, t4) * 1.08)
    ax.set_title(title, fontsize=FS, color=INK, pad=6)
    # shared x-label: set once on the figure, bold, instead of per panel
    leg = ax.legend(loc="lower right", frameon=False, fontsize=FS + 1,
                    title=f"training time to {target:.2f} dB",
                    handlelength=1.8, borderpad=0.2, labelspacing=0.3)
    leg.get_title().set_color(MUTED); leg.get_title().set_fontsize(FS - 1)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=FS - 1)
    print(f"{title}: 1GPU {t1:.1f}s -> {target:.2f} dB | "
          f"4GPU reaches it at {t4:.1f}s ({t1/t4:.2f}x)")

axes[0].set_ylabel("Global PSNR (dB)", fontsize=FS, color=INK, fontweight="bold")
fig.supxlabel("Pure training wall time (s)", fontsize=FS, color=INK,
              fontweight="bold", y=0.02)
plt.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=170, bbox_inches="tight")
print(f"wrote {OUT}")
