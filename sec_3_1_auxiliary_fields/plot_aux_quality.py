"""Aux-quality ladder figure for the methodology section (Sec. enhanced_aux).

Left: NYX baryon density (cascade order DMD->temperature->baryon).
Right: NYX temperature   (cascade order DMD->baryon->temperature).
Curves (SZ3 side, ours): original aux (upper bound) / enhanced aux (cascade) /
decompressed aux (paper protocol) / no aux; SZ3-only base as grey dashed.

The pickles behind each curve are pinned in AUXQ_LADDER_PINS.json next to this
script (tracked). If that file is missing it is rebuilt from the stage windows of
Reproduce/logs/auxq_ablation.log (written by run_auxq_ablation.sh).
    python plot_aux_quality.py -> Reproduce/figures/nyx_aux_quality_ladder.{pdf,png}
"""
import os, re, json, glob, pickle, datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(HERE), "Reproduce")
CACHE = os.path.join(os.path.dirname(HERE), "sec_4_evaluation", "sperr_fft_cache")
LOG = os.path.join(R, "logs", "auxq_ablation.log")
PINS = os.path.join(HERE, "AUXQ_LADDER_PINS.json")

def stage_windows():
    txt = open(LOG).read()
    pat = re.compile(r"\[auxq\] === (\w+) (nyx_\w) (start|exit=(\d+)) (.+?) ===")
    starts, wins = {}, {}
    for m in pat.finditer(txt):
        tag, task = m.group(1), m.group(2)
        t = datetime.datetime.strptime(m.group(5).strip(), "%a %b %d %I:%M:%S %p %Z %Y").timestamp()
        if m.group(3) == "start": starts[(tag, task)] = t
        elif m.group(4) == "0":   wins[(tag, task)] = (starts[(tag, task)], t)
        else: raise SystemExit(f"stage {tag} {task} exited nonzero -- not collecting")
    return wins

FIELD = {"nyx_b": "NYX_baryon_density", "nyx_t": "NYX_temperature"}
def pkl_in(win, task):
    lo, hi = win
    c = [p for p in glob.glob(f"{CACHE}/{FIELD[task]}__*.pkl") if lo <= os.path.getmtime(p) <= hi + 5]
    assert len(c) == 1, f"{task}: expected 1 pkl in window, got {c}"
    return os.path.basename(c[0])

if os.path.isfile(PINS):
    sel = json.load(open(PINS))["selection"]
else:   # rebuild the pin file from the ablation run log
    wins = stage_windows()
    pins = json.load(open(os.path.join(CACHE, "PAPER_FINAL_CACHES.json")))
    cB = json.load(open(os.path.join(R, "results", "CASCADE_B_CACHES.json")))   # single cascade DMD->T->BD
    new = {tag: {task: pkl_in(wins[(tag, task)], task) for task in ("nyx_b", "nyx_t")} for tag in ("origaux", "noaux")}
    sel = {  # panel -> label -> pkl
      "Baryon": {"Original aux": new["origaux"]["nyx_b"], "Enhanced aux": cB["Baryon"],
                 "Decompressed aux": pins["Baryon"], "No aux": new["noaux"]["nyx_b"]},
      "Temp":   {"Original aux": new["origaux"]["nyx_t"], "Enhanced aux": cB["Temp"],   # temperature is stage 2 (enhanced DMD only)
                 "Decompressed aux": pins["Temp"], "No aux": new["noaux"]["nyx_t"]},
    }
    json.dump({"selection": sel, "windows": {f"{k[0]}:{k[1]}": v for k, v in wins.items()},
               "note": "origaux/noaux on RTX PRO 6000, paper protocol (shuffled, lr [1e-3,1e-2]); "
                       "enhanced = single cascade DMD->T->BD: baryon stage 3, temperature stage 2; decompressed = PAPER_FINAL pins"},
              open(PINS, "w"), indent=1)

def load(p): return pickle.load(open(os.path.join(CACHE, p), "rb"))
STY = {"Original aux":     dict(color="#7f7f7f", marker="^", ls=":",  lw=2.6),
       "Enhanced aux":     dict(color="#1f77b4", marker="o", ls="-",  lw=2.8),
       "Decompressed aux": dict(color="#2ca02c", marker="s", ls="-",  lw=2.6),
       "No aux":           dict(color="#d62728", marker="v", ls="--", lw=2.4)}
FS = 31
fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2))
for ax, (panel, title) in zip(axes, [("Temp", "NYX temperature"), ("Baryon", "NYX baryon density")]):   # decode order: T (stage 2) left, BD (stage 3) right
    base = load(sel[panel]["Decompressed aux"])["sz3"]
    ax.plot(base["CR"], base["PSNR"], color="k", ls="--", marker="X", ms=17, lw=2.6, label="SZ3 only")
    for lab in ("No aux", "Decompressed aux", "Enhanced aux", "Original aux"):
        s = load(sel[panel][lab])["pipe"]
        ax.plot(s["CR"], s["PSNR"], ms=18, mfc="none", mew=3.2, label=lab, **STY[lab])
    ax.set_title(title, fontsize=FS + 2, fontweight="bold")
    ax.tick_params(labelsize=FS - 3); ax.grid(alpha=0.3)
axes[0].set_ylabel("PSNR (dB)", fontsize=FS, fontweight="bold")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=5, fontsize=FS,
           frameon=False, handletextpad=0.4, columnspacing=1.0)
fig.supxlabel("Effective Compression Ratio", fontsize=FS, fontweight="bold")
fig.tight_layout()
os.makedirs(os.path.join(R, "figures"), exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(R, "figures", f"nyx_aux_quality_ladder.{ext}"), dpi=120, bbox_inches="tight")
print("Saved: Reproduce/figures/nyx_aux_quality_ladder.pdf")
for panel in sel:
    for lab, p in sel[panel].items():
        s = load(p)["pipe"]; b = load(sel[panel]["Decompressed aux"])["sz3"]
        g = [f"{x - y:+.2f}" for x, y in zip(s["PSNR"], b["PSNR"])]
        print(f"{panel:6s} {lab:17s} gains vs SZ3: {' '.join(g)}")
