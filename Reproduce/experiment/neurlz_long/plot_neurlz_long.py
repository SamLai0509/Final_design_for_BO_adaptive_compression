"""NeurLZ long-budget trajectories vs AdaMit's Table-2 point (SZ3 side), 2x3 figure."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(H, "neurlz_long_2000.json")))
INF = {"nyx_b":2.24,"nyx_t":2.22,"nyx_d":2.18,"miranda":24.10,"qmcpack":11.36,"mag":1.63}
ORDER = [("nyx_b","NYX baryon density"),("nyx_t","NYX temperature"),("nyx_d","NYX dark matter density"),
         ("miranda","Miranda"),("qmcpack","QMCPack"),("mag","Magnetic Reconnection")]
FS = 20
fig, axes = plt.subplots(2, 3, figsize=(19, 9.5))
for ax, (k, title) in zip(axes.ravel(), ORDER):
    d = r[k]; t, p = d["hist_time"], d["hist_psnr"]
    ax.axhline(d["psnr_base"], color="k", ls="--", lw=1.8, label="SZ3 base")
    ax.plot(t, p, color="#d62728", lw=2.2, label="SZ3 + NeurLZ (long budget)")
    ax.axhline(d["psnr_target"], color="#4355b9", ls=":", lw=2.2)
    ax.plot([d["adamit_budget_s"]], [d["psnr_target"]], marker="*", ms=22, color="#4355b9", mec="k", mew=1.2, ls="none",
            label="SZ3 + AdaMit (at its training budget)")
    ax.axvline(d["adamit_budget_s"], color="#4355b9", lw=1.0, alpha=0.5)
    ax.set_xscale("log"); ax.set_title(title, fontsize=FS+2, fontweight="bold")
    ax.tick_params(labelsize=FS-4); ax.grid(alpha=0.3, which="both")
    best = max(p); ax.text(0.98, 0.04, f"NeurLZ best {best-d['psnr_base']:+.2f} dB @ {d['cap_s']:.0f} s\nAdaMit {d['psnr_target']-d['psnr_base']:+.2f} dB @ {d['adamit_budget_s']:.0f} s",
                           transform=ax.transAxes, ha="right", va="bottom", fontsize=FS-5,
                           bbox=dict(boxstyle="round", fc="white", ec="0.7"))
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.965), ncol=3, fontsize=FS-2, frameon=False)
fig.supxlabel("NeurLZ pure training wall time (s, log scale)", fontsize=FS+1, fontweight="bold")
fig.supylabel("PSNR (dB)", fontsize=FS+1, fontweight="bold", x=0.005)
fig.tight_layout(rect=(0.01, 0, 1, 0.96))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(H, f"neurlz_time_to_match.{ext}"), dpi=110, bbox_inches="tight")
print("Saved neurlz_time_to_match.pdf")
