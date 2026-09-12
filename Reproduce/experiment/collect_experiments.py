#!/usr/bin/env python
"""Gather the NYX sibling-protocol experiments into Reproduce/experiment/:
     reference   paper-final run (CR-matched decompressed siblings)       Reproduce/results/PAPER_FINAL_CACHES.json
     v3          draft run (lossless siblings)                             sperr_fft_cache/PAPER_V3_CACHES.json
     noaux       single-field (no siblings), 5090 pinned -- indicative     sperr_fft_cache/DIAG_NOAUX_5090.json
     novel       velocity siblings dropped                                 pkls written during run_novel_6000.sh
     cascadeA    DMD -> baryon -> temperature with ENHANCED siblings      pkls written during run_cascade_6000.sh
     cascadeB    DMD -> temperature -> baryon with ENHANCED siblings
Writes pins + pkl copies + summary.md (per-point PSNR gain / FFT reductions) + a figure.
"""
import os, re, glob, json, time, shutil, pickle
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); CACHE = f"{REPO}/sec_4_evaluation/sperr_fft_cache"
L = f"{REPO}/Reproduce/logs"; OUT = f"{REPO}/Reproduce/experiment"; os.makedirs(f"{OUT}/pkl", exist_ok=True)
PREFIX = {"Baryon": "NYX_baryon_density", "Temp": "NYX_temperature", "DMD": "NYX_dark_matter_density"}
TASK2KEY = {"nyx_b": "Baryon", "nyx_t": "Temp", "nyx_d": "DMD"}
NAME = {"Baryon": "Baryon density", "Temp": "Temperature", "DMD": "Dark matter density"}


def load(f):
    r = pickle.load(open(f, "rb")); return r.get("results", r) if "results" in r else r


def stamps(logfile, tag):
    """{stage_label: (start_epoch, exit_epoch)} from '[tag] === label start <date> ===' / 'exit=' lines."""
    if not os.path.isfile(logfile):
        return {}
    out, cur = {}, {}
    for line in open(logfile):
        m = re.search(rf"\[{tag}\] === (\S+)(?: \((\S+)\))? (start|exit=\d+) (.+) ===", line)
        if not m:
            continue
        label, task, what, date = m.group(1), m.group(2), m.group(3), m.group(4).strip()
        t = time.mktime(time.strptime(date, "%a %b %d %I:%M:%S %p %Z %Y"))   # 12-hour clock + AM/PM (as `date` prints here)
        key = (label, task or label)
        if what == "start":
            cur[key] = t
        elif key in cur:
            out[key] = (cur.pop(key), t)
    return out


def pkl_between(prefix, t0, t1):
    fs = [f for f in glob.glob(f"{CACHE}/{prefix}__*.pkl") if t0 - 5 <= os.path.getmtime(f) <= t1 + 5]
    return max(fs, key=os.path.getmtime) if fs else None


exps = {}
ref_pin = f"{REPO}/Reproduce/results/PAPER_FINAL_CACHES.json"
if os.path.isfile(ref_pin):
    ref = json.load(open(ref_pin))
    exps["reference (CR-matched siblings)"] = {k: f"{REPO}/Reproduce/results/{ref[k]}" for k in PREFIX}
else:   # collect.py has not run yet: take the paper chain's NYX pkls from its own start/exit stamps
    rf = {}
    for (label, task), (t0, t1) in stamps(f"{L}/run_all.log", "run_all").items():
        if task in TASK2KEY:
            f = pkl_between(PREFIX[TASK2KEY[task]], t0, t1)
            if f: rf[TASK2KEY[task]] = f
    exps["reference (CR-matched siblings)"] = rf
v3 = json.load(open(f"{CACHE}/PAPER_V3_CACHES.json"))
exps["v3 draft (lossless siblings)"] = {k: f"{CACHE}/{v3[k]}" for k in PREFIX}
if os.path.isfile(f"{CACHE}/DIAG_NOAUX_5090.json"):
    na = json.load(open(f"{CACHE}/DIAG_NOAUX_5090.json"))
    exps["no siblings (5090, indicative)"] = {k: f"{CACHE}/{na[k]}" for k in PREFIX if k in na}
nv = {}
for (label, task), (t0, t1) in stamps(f"{L}/run_novel_6000.log", "novel").items():
    if task in TASK2KEY:
        f = pkl_between(PREFIX[TASK2KEY[task]], t0, t1)
        if f: nv[TASK2KEY[task]] = f
if nv: exps["no velocity siblings"] = nv
cas = {"A": {}, "B": {}}
for (label, task), (t0, t1) in stamps(f"{L}/run_cascade_6000.log", "cascade").items():
    f = pkl_between(PREFIX[TASK2KEY[task]], t0, t1) if task in TASK2KEY else None
    if not f: continue
    if label.startswith("stage1"):
        cas["A"][TASK2KEY[task]] = f; cas["B"][TASK2KEY[task]] = f
    elif label.startswith("A_"): cas["A"][TASK2KEY[task]] = f
    elif label.startswith("B_"): cas["B"][TASK2KEY[task]] = f
if cas["A"]: exps["cascade A: DMD -> baryon -> temperature (enhanced siblings)"] = cas["A"]
if cas["B"]: exps["cascade B: DMD -> temperature -> baryon (enhanced siblings)"] = cas["B"]

pins = {}
for name, d in exps.items():
    pins[name] = {}
    for k, f in d.items():
        shutil.copy2(f, f"{OUT}/pkl/"); pins[name][k] = os.path.basename(f)
json.dump(pins, open(f"{OUT}/EXPERIMENT_PINS.json", "w"), indent=1)

lines = ["# NYX sibling-protocol experiments", "",
         "All runs: shuffled slice order, lr in [1e-3, 1e-2], 10 % Phase-1, no trust gates, 30k-param model, same NeurLZ",
         "protocol; RTX PRO 6000 unless marked. Gains are PSNR(base + Ours) - PSNR(base) in dB; FFT columns are the",
         "reductions of the per-slice FFT magnitude / phase L1 error vs the base. Siblings, where used, are the decoder's",
         "copies (CR level closest to the target); 'enhanced' = the output of an earlier cascade stage.", ""]
summary = {}
for k in PREFIX:
    lines += [f"## {NAME[k]}", "", "| experiment | base | " + " | ".join(f"CR≈{c}" for c in (100, 150, 220, 320, 500)) + " | mean gain | mean FFT-mag red. | mean FFT-pha red. |",
              "|---|---|" + "---|" * 5 + "---|---|---|"]
    for name, d in exps.items():
        if k not in d: continue
        r = load(d[k])
        for side, b, p in (("SZ3", "sz3", "pipe"), ("SPERR", "sperr", "sperr_pipe")):
            if not r[b]["CR"]: continue
            g = np.array(r[p]["PSNR"]) - np.array(r[b]["PSNR"])
            mr = 1 - np.array(r[p]["fft_mag"]) / np.array(r[b]["fft_mag"]); pr = 1 - np.array(r[p]["fft_phase"]) / np.array(r[b]["fft_phase"])
            lines.append(f"| {name} | {side} | " + " | ".join(f"{x:+.2f}" for x in g) + f" | {g.mean():+.2f} | {100*mr.mean():+.0f}% | {100*pr.mean():+.0f}% |")
            summary.setdefault(name, {})[(k, side)] = (g, mr, pr)
    lines.append("")
open(f"{OUT}/summary.md", "w").write("\n".join(lines))
for f in glob.glob(f"{L}/noaux5090_*.log") + glob.glob(f"{L}/novel6000_*.log") + glob.glob(f"{L}/cascade_*.log") + [f"{L}/run_noaux_5090.log", f"{L}/run_novel_6000.log", f"{L}/run_cascade_6000.log"]:
    if os.path.isfile(f): os.makedirs(f"{OUT}/logs", exist_ok=True); shutil.copy2(f, f"{OUT}/logs/")

# figure: gain vs CR per field/side, one line per experiment
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), sharex=True)
markers = ["o", "s", "^", "D", "v", "P", "X"]
for c, k in enumerate(PREFIX):
    for rr, side in enumerate(("SZ3", "SPERR")):
        ax = axes[rr, c]
        for i, (name, d) in enumerate(exps.items()):
            if (k, side) not in summary.get(name, {}): continue
            r = load(d[k]); b = "sz3" if side == "SZ3" else "sperr"
            ax.plot(r[b]["CR"], summary[name][(k, side)][0], marker=markers[i % len(markers)], lw=2, ms=7, label=name)
        ax.set_title(f"{NAME[k]} — {side} base", fontsize=13); ax.grid(True, alpha=0.3); ax.axhline(0, color="k", lw=0.8)
        if c == 0: ax.set_ylabel("PSNR gain over base (dB)", fontsize=12)
        if rr == 1: ax.set_xlabel("Effective CR", fontsize=12)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=11, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(f"{OUT}/nyx_sibling_experiments.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/nyx_sibling_experiments.png", dpi=130, bbox_inches="tight")
print("\n".join(lines)); print("\nwritten to", OUT)
