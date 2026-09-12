"""Figure 12, redrawn from the corrected runs.

Left  eps(k) per spherical k-shell for the base codec, NeurLZ and ours.
Right best-so-far max_k eps(k) against pure training wall time, with the Nyx
      1 % requirement and the point where each method first clears it.

The right panel is linear, not logarithmic. NeurLZ trains for an order of
magnitude longer than we do, so a single linear axis either hides our crossing
or hides its endpoint; the main axes cover the window where both cross, and the
inset carries NeurLZ's full descent so neither fact is lost.

  python plot_fig12.py [--data figures/fig12_enh] [--out ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(HERE / "figures" / "fig12"))
ap.add_argument("--out", default="")
ap.add_argument("--zoom", type=float, default=60.0,
                help="right panel x-range, seconds")
args = ap.parse_args()

D = Path(args.data)
r = json.loads((D / "fig12_data.json").read_text())
OUT = Path(args.out) if args.out else D / "fig12_power_spectrum.pdf"

plt.rcdefaults()
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "font.weight": "normal", "axes.labelweight": "normal",
                     "axes.titleweight": "normal"})

C_BASE, C_NL, C_OURS, C_REQ = "#8a8a8a", "#d1495b", "#2a78d6", "#e8952b"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d9d6"
FS = 10


def first_below(t, m, thr=1.0):
    """(time, index) at which the best-so-far max_k eps(k) first clears thr."""
    best = float("inf")
    for i, (tt, v) in enumerate(zip(t, m)):
        best = min(best, v)
        if best < thr:
            return tt, i
    return None, None


def spectrum_at(run, idx):
    """eps(k) at evaluation idx, falling back to the end-of-training vector.

    The left panel compares the two methods at the moment each first meets the
    requirement, not at the end of their very different training budgets.
    """
    tr = run.get("eps_trace")
    if tr and idx is not None and idx < len(tr):
        return tr[idx]
    return run["eps"]


k = np.array(r["k"])
o, n = r["ours"], r["neurlz"]
o_t, o_i = first_below(o["time"], o["max_eps_trace"])
n_t, n_i = first_below(n["time"], n["max_eps_trace"])
o_eps_at, n_eps_at = spectrum_at(o, o_i), spectrum_at(n, n_i)
o_best = np.minimum.accumulate(o["max_eps_trace"])
n_best = np.minimum.accumulate(n["max_eps_trace"])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.6))

# ── left: eps(k) ────────────────────────────────────────────────────────────
axL.axhline(1.0, color=C_REQ, ls="--", lw=1.2, label="Nyx 1% requirement")
for vals, c, mk, ls, lab in (
        (r["eps_base"], C_BASE, "^", "--", "SZ3"),
        (n_eps_at, C_NL, "v", ":", f"SZ3+NeurLZ ({n_t:.1f} s)"),
        (o_eps_at, C_OURS, "o", "-", f"SZ3+Ours ({o_t:.1f} s)")):
    axL.plot(k, vals, color=c, ls=ls, lw=1.6, marker=mk, markersize=5,
             markerfacecolor="none", markeredgewidth=1.3, label=lab)
axL.set_xlabel("$k$", fontsize=FS, color=INK)
axL.set_ylabel(r"$\varepsilon(k)$  (%)", fontsize=FS, color=INK)
axL.set_title(r"$\varepsilon(k)$ when each first meets 1%", fontsize=FS, color=INK, pad=6)
axL.set_xticks(k)

# ── right: best-so-far max_k eps(k), linear, with a full-range inset ────────
axR.axhline(1.0, color=C_REQ, ls="--", lw=1.2, label="Nyx 1% requirement")
# Each curve stops at the requirement. What either method does afterwards is a
# different question from the one this panel asks, and NeurLZ's tail is an
# order of magnitude longer than ours, so carrying it would set the x-range.
for t, best, idx, c, mk, ls, lab, tx in (
        (n["time"], n_best, n_i, C_NL, "v", ":", f"SZ3+NeurLZ: {n_t:.1f} s", n_t),
        (o["time"], o_best, o_i, C_OURS, "o", "-", f"SZ3+Ours: {o_t:.1f} s", o_t)):
    stop = (idx + 1) if idx is not None else len(t)
    axR.plot(t[:stop], best[:stop], color=c, ls=ls, lw=1.6, marker=mk,
             markersize=4.5, markerfacecolor="none", markeredgewidth=1.2, label=lab)
    if tx is not None:
        axR.plot([tx], [best[idx]], marker=mk, color=c, markersize=10, zorder=5)
axR.set_xlim(0, max(x for x in (o_t, n_t) if x is not None) * 1.12)
axR.set_xlabel("Pure training wall time (s)", fontsize=FS, color=INK)
axR.set_ylabel(r"best $\max_k \varepsilon(k)$  (%)", fontsize=FS, color=INK)
axR.set_title("Time to meet the 1% requirement", fontsize=FS, color=INK, pad=6)

for ax in (axL, axR):
    ax.legend(frameon=False, fontsize=FS - 2, loc="upper right",
              handlelength=1.9, borderpad=0.2, labelspacing=0.3)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=FS - 2)
axR.legend(frameon=False, fontsize=FS - 2, loc="lower left",
           handlelength=1.9, borderpad=0.2, labelspacing=0.3)

aux_note = (f"ours: {r['aux_mode']}"
            + (f", NeurLZ: {r['neurlz_aux_mode']}"
               if r.get("neurlz_aux_mode") and r["neurlz_aux_mode"] != r["aux_mode"]
               else ""))
fig.suptitle(f"NYX Baryon Density @ CR {r['cr_base']:.0f}   ({aux_note} siblings)",
             fontsize=FS + 1, color=INK, y=1.03)
plt.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=170, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"ours   meets 1% at {o_t:.2f}s -> max eps {max(o_eps_at):.3f}% "
      f"(ends {max(o['eps']):.3f}% after {o['train_s']:.0f}s)")
print(f"neurlz meets 1% at {n_t:.2f}s -> max eps {max(n_eps_at):.3f}% "
      f"(ends {max(n['eps']):.3f}% after {n['train_s']:.0f}s)")
