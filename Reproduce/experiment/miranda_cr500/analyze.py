#!/usr/bin/env python
"""Miranda CR~500 operating point: the two paper-chain samples + N standalone repeats.
Tabulates Phase-1 pick (axis, lr) and outcome per side, writes miranda_cr500.md and a small figure."""
import os, re, glob, os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); E = f"{REPO}/Reproduce/experiment/miranda_cr500"; L = f"{REPO}/Reproduce/logs"
pick_re = re.compile(r"PICK axis(\d) lr=([\d.e+-]+) proxy=[\d.]+ gain=[+\-\d.]+ axis_spread=([\d.]+) lr_spread=([\d.]+)")
res_re  = re.compile(r"(SZ3|SPERR)\s+([\d.]+)x/\s*([\d.]+)dB.*?\|\s*(?:SZ3\+model|SPERR\+Ours)\s+([\d.]+)x/\s*([\d.]+)dB")

def parse(logfile, want_cr=500.0):
    """(side, base CR, base PSNR, ours PSNR, axis, lr, lr_spread) for the point(s) closest to want_cr."""
    rows, cur = [], None
    for line in open(logfile):
        m = pick_re.search(line)
        if m: cur = (int(m[1]), float(m[2]), float(m[3]), float(m[4])); continue
        m = res_re.search(line)
        if m and cur:
            rows.append(dict(side=m[1], cr=float(m[2]), base=float(m[3]), ours=float(m[5]), axis="ZYX"[cur[0]], lr=cur[1], lr_spread=cur[3])); cur = None
    out = {}
    for side in ("SZ3", "SPERR"):
        cand = [r for r in rows if r["side"] == side]
        if cand: out[side] = min(cand, key=lambda r: abs(np.log(r["cr"] / want_cr)))
    return out

samples = []
for name, lf in (("paper chain (08-28 16:28, shuffled)", f"{REPO}/sec_4_evaluation/run_logs/2026-08-28/shuf6000_miranda.log"),
                 ("paper chain (08-29 03:11 rerun)", f"{L}/miranda.log")):
    if os.path.isfile(lf): samples.append((name, parse(lf)))
for lf in sorted(glob.glob(f"{E}/rep*.log")):
    samples.append((f"standalone {os.path.basename(lf)[:-4]}", parse(lf)))

lines = ["# Miranda CR≈500: repeat analysis", "",
         "Paper configuration (shuffled, lr ∈ [1e-3, 1e-2], plain argmax). Each row is an independent run of this one",
         "operating point (SZ3 band 4; SPERR aligned to the same CR). 'gain' = PSNR(base+Ours) − PSNR(base).", "",
         "| run | SZ3: pick (axis, lr) | lr spread | SZ3 gain | SPERR: pick (axis, lr) | lr spread | SPERR gain |", "|---|---|---|---|---|---|---|"]
g = {"SZ3": [], "SPERR": []}
for name, s in samples:
    cells = [name]
    for side in ("SZ3", "SPERR"):
        r = s.get(side)
        if r:
            gain = r["ours"] - r["base"]; g[side].append((name, r["lr"], gain))
            cells += [f"{r['axis']}, {r['lr']:.1e}", f"{r['lr_spread']:.2f} dB", f"**{gain:+.2f}**"]
        else: cells += ["—", "—", "—"]
    lines.append("| " + " | ".join(cells) + " |")
lines.append("")
for side in ("SZ3", "SPERR"):
    v = np.array([x[2] for x in g[side]]); ok = v > 0.05
    if len(v):
        lines.append(f"- {side}: {ok.sum()}/{len(v)} runs healthy; healthy gains {', '.join(f'{x:+.2f}' for x in v[ok])} "
                     f"(median {np.median(v[ok]) if ok.any() else float('nan'):+.2f}); collapsed picks: "
                     f"{', '.join(f'{x[1]:.1e}' for x in g[side] if x[2] <= 0.05) or 'none'}.")
open(f"{E}/miranda_cr500.md", "w").write("\n".join(lines) + "\n")
fig, ax = plt.subplots(figsize=(8, 4))
for i, side in enumerate(("SZ3", "SPERR")):
    for j, (name, lr, gain) in enumerate(g[side]):
        ax.scatter(lr, gain, s=120, marker="o" if side == "SZ3" else "s", color="#4355b9" if side == "SZ3" else "#e6550d",
                   facecolors="none" if gain <= 0.05 else None, linewidths=2, label=side if j == 0 else None)
ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("Phase-1 pick: lr"); ax.set_ylabel("gain over base (dB)")
ax.set_title("Miranda CR≈500 — independent runs (hollow = collapsed)"); ax.grid(True, alpha=0.3); ax.legend()
fig.savefig(f"{E}/miranda_cr500_repeats.pdf", bbox_inches="tight"); fig.savefig(f"{E}/miranda_cr500_repeats.png", dpi=120, bbox_inches="tight")
print("\n".join(lines))
