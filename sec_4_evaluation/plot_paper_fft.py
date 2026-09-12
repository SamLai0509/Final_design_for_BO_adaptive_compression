"""Paper figure: FFT magnitude + phase error vs effective CR, six fields, 3 rows x 4 cols
(each dataset owns an adjacent magnitude|phase pair). Same data pin / styles as
plot_paper_figs.py; this is the notebook's plot_fft_combined cell as a script.
    python plot_paper_fft.py [pin.json]
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
GRID_ALPHA = 0.30
FFT2_FIGSIZE    = (16.2, 8.8)
FFT2_DSTITLE_FS = 19     # dataset name over each panel pair
FFT2_METRIC_FS  = 15     # "Magnitude error" / "Phase error" per-panel title
FFT2_LABEL_FS   = 18     # supxlabel
FFT2_TICK_FS    = 16
FFT2_LEGEND_FS  = 16
FFT2_MS         = 10
FFT2_LW         = 1.7
FFT2_XTICKS     = [100, 300, 500]
_FFT2_METRICS = [("fft_mag", "Magnitude error"), ("fft_phase", "Phase error")]
_LEG_ORDER = ["SZ3", "SZ3 + NeurLZ", "SZ3 + Ours", "SPERR", "SPERR + NeurLZ", "SPERR + Ours"]

DATA = {t: pickle.load(open(os.path.join(CACHE, f), "rb")) for t, f in PANELS}


def plot_fft_combined(save_stem):
    fig, axes = plt.subplots(3, 4, figsize=FFT2_FIGSIZE, sharex=True)
    fig.subplots_adjust(left=0.048, right=0.995, bottom=0.10, top=0.87, wspace=0.32, hspace=0.35)
    pair_axes = []
    for i, (title, _) in enumerate(PANELS):
        row, col0 = divmod(i, 2)
        axM, axP = axes[row, 2 * col0], axes[row, 2 * col0 + 1]
        pair_axes.append((axM, axP, title))
        r = DATA[title]
        for ax, (metric_key, metric_lbl) in zip((axM, axP), _FFT2_METRICS):
            for key, lbl in _SERIES_DEFS:
                d = r.get(key, {})
                if not d.get("CR"):
                    continue
                x, y = np.asarray(d["CR"], float), np.asarray(d[metric_key], float); o = np.argsort(x)
                st = _SERIES_STYLE[key]
                ax.plot(x[o], y[o], marker=st["marker"], linestyle=st["ls"], color=st["color"],
                        linewidth=FFT2_LW, markersize=FFT2_MS, markerfacecolor="none",
                        markeredgewidth=1.8, label=lbl)
            ax.set_title(metric_lbl, fontsize=FFT2_METRIC_FS, pad=6)
            ax.set_xticks(FFT2_XTICKS)
            ax.tick_params(axis="both", labelsize=FFT2_TICK_FS)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
            ax.yaxis.get_offset_text().set_fontsize(FFT2_TICK_FS - 4)
            ax.grid(True, alpha=GRID_ALPHA)
    for axM, axP, title in pair_axes:
        pL, pR = axM.get_position(), axP.get_position()
        fig.text(0.5 * (pL.x0 + pR.x1), pL.y1 + 0.028, title, ha="center", va="bottom", fontsize=FFT2_DSTITLE_FS)
    fig.supxlabel("Effective Compression Ratio", fontsize=FFT2_LABEL_FS, fontweight="bold", y=0.015)
    lax = max(axes.ravel(), key=lambda a: len(a.get_lines()))
    handles, labels = lax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: _LEG_ORDER.index(labels[i]) if labels[i] in _LEG_ORDER else 99)
    top = max(a.get_position().y1 for a in axes[0]) + 0.055
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
               bbox_to_anchor=(0.5, top), fontsize=FFT2_LEGEND_FS, ncol=len(labels),
               frameon=True, framealpha=0.0, borderaxespad=0.0)
    for out in (save_stem + ".pdf", save_stem + ".png"):
        fig.savefig(out, dpi=150, bbox_inches="tight"); print("Saved:", out)
    plt.close(fig)


plot_fft_combined(OUTDIR + "sperr_fft_all6")
