"""Paper figure: PSNR vs effective CR, six fields, 3 rows x 2 cols (single-column width).

Data come from the pickles listed in a cache-pin JSON (default PAPER_V3_CACHES.json);
pass another pin file as argv[1] (e.g. CACHES_V4_lrmax1e-2_qmc35k.json) to regenerate
from a different six-field run without touching the plotting code.
    python plot_paper_figs.py [pin.json]
"""
import os, sys, json, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE   = os.path.dirname(os.path.abspath(__file__))
CACHE  = os.environ.get("ADAMIT_CACHE_DIR", os.path.join(HERE, "sperr_fft_cache"))
OUTDIR = os.environ.get("ADAMIT_FIG_DIR", os.path.join(HERE, "overleaf_figures")).rstrip("/") + "/"
PIN    = sys.argv[1] if len(sys.argv) > 1 else "PAPER_V3_CACHES.json"
pin    = json.load(open(os.path.join(CACHE, PIN)))
PANELS = [("Baryon density", pin["Baryon"]), ("Temperature", pin["Temp"]),
          ("Dark matter density", pin["DMD"]), ("Miranda", pin["Miranda"]),
          ("QMCPack", pin["QMC"]), ("Magnetic Reconnection", pin["Magnetic"])]
print("pin:", PIN, {t: f for t, f in PANELS})

_SERIES_DEFS = [("sz3", "SZ3"), ("pipe", "SZ3 + Ours"), ("sperr", "SPERR"), ("sperr_pipe", "SPERR + Ours"),
                ("neurlz", "SZ3 + NeurLZ"), ("sperr_neurlz", "SPERR + NeurLZ")]
_SERIES_STYLE = {
    "sz3":          dict(color="#08519c", marker="o", ls="-"),
    "pipe":         dict(color="#4355b9", marker="s", ls="--"),
    "neurlz":       dict(color="#4292c6", marker="^", ls=":"),
    "sperr":        dict(color="#a50f15", marker="o", ls="-"),
    "sperr_pipe":   dict(color="#e6550d", marker="s", ls="--"),
    "sperr_neurlz": dict(color="#fdae6b", marker="^", ls=":"),
}
TITLE_FS = LABEL_FS = TICK_FS = LEGEND_FS = 20
MARKER_SCALE, LINE_SCALE, GRID_ALPHA, DPI = 2.0, 1.0, 0.30, 150
# legend displays as  row 1: SZ3, SZ3+NeurLZ, SZ3+Ours / row 2: SPERR, SPERR+NeurLZ, SPERR+Ours
_LEG_ORDER = ["SZ3", "SPERR", "SZ3 + NeurLZ", "SPERR + NeurLZ", "SZ3 + Ours", "SPERR + Ours"]

DATA = {t: pickle.load(open(os.path.join(CACHE, f), "rb")) for t, f in PANELS}

def plot_metric_3x2(metric_key, ylabel, save_name, ylog=False):
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 12.0))
    for ax, (title, _) in zip(axes.ravel(), PANELS):
        r = DATA[title]
        for key, lbl in _SERIES_DEFS:
            d = r.get(key, {})
            if not d.get("CR"):
                continue
            x, y = np.asarray(d["CR"], float), np.asarray(d[metric_key], float); o = np.argsort(x)
            st = _SERIES_STYLE[key]
            ax.plot(x[o], y[o], marker=st["marker"], linestyle=st["ls"], color=st["color"],
                    linewidth=1.8 * LINE_SCALE, markersize=7 * MARKER_SCALE,
                    markerfacecolor="none", markeredgewidth=2.0, label=lbl)
        if ylog:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=TITLE_FS); ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(True, alpha=GRID_ALPHA, which="both")
    fig.supxlabel("Effective Compression Ratio", fontsize=LABEL_FS, fontweight="bold")
    fig.supylabel(ylabel, fontsize=LABEL_FS, fontweight="bold")
    lax = max(axes.ravel(), key=lambda a: len(a.get_lines()))
    handles, labels = lax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: _LEG_ORDER.index(labels[i]) if labels[i] in _LEG_ORDER else 99)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    top = max(a.get_position().y1 for a in axes[0])    # top edge of the first row (incl. titles below)
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
               bbox_to_anchor=(0.53, top + 0.05), fontsize=LEGEND_FS, ncol=3,
               frameon=True, framealpha=0.0, borderaxespad=0.0, handletextpad=0.4,
               columnspacing=1.2)                      # transparent box, hugging the panels
    for out in (save_name + ".pdf", save_name + ".png"):
        plt.savefig(out, dpi=DPI, bbox_inches="tight"); print("Saved:", out)
    plt.close()

plot_metric_3x2("PSNR", "PSNR (dB)", OUTDIR + "sperr_cmp_all6_3x2")
