"""Generate /home/sam/Halo_Finder/Final_design/REPRODUCE.ipynb (AdaMit reproducibility guide)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# AdaMit — Reproducibility Guide

**Adaptive, frequency-aware online neural error mitigation over SZ3 / SPERR** (paper: *AdaMit*, USENIX FAST submission).
This notebook is the single entry point for reproducing every number and figure in the paper. It has two modes:

| mode | what it does | time |
|---|---|---|
| **from cache** (default, this notebook as saved) | loads the pinned result caches (`SPERR/sperr_fft_cache/*.pkl`), rebuilds every table and figure | minutes, no GPU |
| **from scratch** | `bash Reproduce/run_all.sh` re-runs the six-field pipeline and the Fig. 8 study, then `Reproduce/collect.py` rebuilds this folder and this notebook | ≈ 4.5 h on one RTX PRO 6000 (see §3) |

Everything the paper reports comes from one pipeline script, **`SPERR/SPERR_fft.py`**, plus a handful of standalone
notebooks for the ablation / analysis figures (§5–§9 below). Each section states *which paper element* it reproduces,
*how it was run*, and *shows the result*.

> **Paths are hard-coded** (`/home/sam/...`, see `README.md`). To run from scratch on another machine edit the
> data / SZ3 / SPERR paths at the top of `SPERR/SPERR_fft.py`; the cache-reading cells below work anywhere the repo is checked out.
""")

code(r"""
import os, sys, json, glob, pickle, re, subprocess, time
import numpy as np
from IPython.display import Image, display, Markdown

REPO  = "/home/sam/Halo_Finder/Final_design"
SPERR = f"{REPO}/SPERR"
OUT   = "__ADAMIT_OUT__"
CACHE = f"{OUT}/results"                # pinned copies of the result pickles (collect.py)
FIGS  = f"{OUT}/figures"
os.environ["ADAMIT_CACHE_DIR"] = CACHE; os.environ["ADAMIT_FIG_DIR"] = FIGS
sys.path.insert(0, SPERR)
import paper_numbers as pn          # the paper's number definitions (iso-PSNR CR gain, Rel L2, FFT reductions, ...)

# ── which six-field run the tables/figures below are built from ────────────────────────
#   PAPER_FINAL_CACHES.json  the paper-final run (Reproduce/run_all.sh): shuffled slice order, lr in [1e-3, 1e-2] (as in Sec. 3.5), CR-matched decompressed siblings, no trust gates
#   PAPER_V3_CACHES.json     the 2026-08-16 run behind the first draft (random sampling, lossless siblings, lr [1e-4,3e-3], gates on) -- for old->new tables
#   (the SPERR/sperr_fft_cache/ directory keeps every intermediate run: *_SEQ_CRMATCHED_*, *_SHUF_CRMATCHED_* = the 2026-08-28 sequential/shuffled chains at lr max 1e-2)
PIN = os.environ.get("ADAMIT_PIN", "PAPER_FINAL_CACHES.json")
print("available pins:", sorted(os.path.basename(p) for p in glob.glob(f"{CACHE}/*.json")))
print("using:", PIN)
pin = json.load(open(f"{CACHE}/{PIN}"))
for k, f in pin.items():
    print(f"  {k:9s} <- {f}  (modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(f'{CACHE}/{f}')))})")
""")

md(r"""
## 1. Environment, platform and data

* Python 3 + `numpy`, `torch` (CUDA), `matplotlib`, `optuna`, `monai` (only for the NeurLZ baseline's `BasicUNet`), `pysz` (SZ3's Python wrapper). `pip install -r requirements.txt` covers the pip part.
* **SZ3** (`libSZ3c.so` + `tools/pysz`) and **SPERR** (`sperr3d` binary) built separately; paths are set at the top of `SPERR/SPERR_fft.py` (`SZ3_LIB`, `PYSZ_PATH`, `SPERR_BIN`).
* **Paper platform**: one NVIDIA RTX PRO 6000 Blackwell (cuda:0 on this machine), 20-core host, 62 GB RAM. All time-budgeted results (every table/figure below) were produced on this GPU with nothing else running on the machine — the pipeline is *wall-clock budgeted*, so a slower/contended GPU trains fewer steps and gives lower PSNR.

Datasets (SDRBench single-precision volumes, all `float32` raw files):

| paper name | file(s) | shape (D,H,W) | bytes | target rel bands (SZ3) | Phase-2 budget | model budget |
|---|---|---|---|---|---|---|
| NYX baryon density | `SDRBENCH-EXASKY-NYX-512x512x512/origin/baryon_density.f32` (+5 siblings) | 512³ | 537 MB | 1.38e-6 … 1.17e-5 | 10 s | 30 000 params |
| NYX temperature | `.../temperature.f32` (+5 siblings) | 512³ | 537 MB | 1.45e-4 … 8.19e-4 | 10 s | 30 000 |
| NYX dark-matter density | `.../dark_matter_density.f32` (+5 siblings) | 512³ | 537 MB | 1.35e-4 … 7.82e-4 | 10 s | 30 000 |
| Miranda | `miranda_1024x1024x1024_float32.raw` | 1024³ | 4.3 GB | 4.55e-3 … 2.54e-2 | 80 s | (see `MIR_PARAMS`) |
| QMCPack | `SDRBENCH-QMCPack/288x115x69x69/einspline_288_115_69_69.pre.f32` | 33120×69×69 | 631 MB | 1.97e-4 … 3.26e-3 | 60 s | 35 000 (bg_h=23 → 34 504) |
| Magnetic reconnection | `magnetic_reconnection_512x512x512_float32.raw` | 512³ | 537 MB | 4.06e-3 … 2.29e-2 | 10 s | (see `MAG_PARAMS`) |

The five rel bands per dataset were bisected once so that SZ3's CR lands on ≈100/150/220/320/500 (see `align_all_to_100_500.py`
comments in `SPERR_fft.py`); SPERR operating points are chosen by PSNR target to land on the same CR grid (`SPERR_MAX_CR = 500`).
""")

code(r"""
# The paper's configuration, read live from SPERR_fft.py so this table can never drift from the code.
src = open(f"{SPERR}/SPERR_fft.py").read()
def const(name):
    m = re.search(rf"^{name}\s*=\s*(.+?)(?:\s+#.*)?$", src, re.M)
    if not m:
        return "?"
    v = m.group(1).strip()
    e = re.search(r"os\.environ\.get\('(\w+)',\s*'([^']*)'\)", v)      # env-overridable constant -> show its default
    return f"{e.group(2)}  (env {e.group(1)})" if e else v
rows = [
 ("Sec. 3.1  slice sampling",          "BG_SAMPLE_MODE",      "how Phase-2 walks the depth slices (env SPERR_SAMPLE_MODE: sequential | shuffled | random)"),
 ("Sec. 3.1  batch",                   "BG_BATCH",            "slices per step (full H×W slice, no crop)"),
 ("Sec. 3.3  frequency loss weight",   "BG_FREQ_WEIGHT",      "w_freq"),
 ("Sec. 3.3  phase weight",            "BG_FFT_PHASE_WEIGHT", "lambda_phase (env SPERR_PHASE_W)"),
 ("Sec. 3.4  weight storage",          "BYTES_PER_PARAM",     "bf16 → 2 bytes/param counted in the effective CR"),
 ("Sec. 3.5  Phase-1 trials",          "BO_N_TRIALS",         "Optuna TPE trials per operating point (no wall-clock timeout: always 10)"),
 ("Sec. 3.5  TPE startup",             "BO_N_STARTUP",        "random startup trials"),
 ("Sec. 3.5  enqueued default lr",     "BO_ENQUEUE_LR",       "first trial = default lr"),
 ("Sec. 3.5  lr search interval",      "BO_LR_MIN",           "lower end (env SPERR_LR_MIN)"),
 ("Sec. 3.5  lr search interval",      "BO_LR_MAX",           "upper end (env SPERR_LR_MAX)"),
 ("Sec. 3.5  slice-direction search",  "BO_SEARCH_DIRECTION", "axes {0,1,2} searched (QMCPack: axis 0 only, bo_axes=[0])"),
 ("Sec. 3.5  Phase-1 / Phase-2 split", "BO_TIME_SPLIT",       "fraction of the time budget for Phase 1 (env SPERR_P1_SPLIT)"),
 ("Sec. 3.5  trust gates",             "BO_MIN_SPREAD_DB",    "-1 = disabled (plain argmax)"),
 ("Sec. 3.5  trust gates",             "BO_MIN_LR_SPREAD_DB", "-1 = disabled"),
 ("Sec. 3.5  trust gates",             "BO_LR_LOW_REJECT_FRAC","-1 = disabled"),
 ("Sec. 3.6  cosine schedule",         "BG_SCHED_TIME_CALIBRATE", "cosine horizon re-planned from measured epoch cost"),
 ("Sec. 3.6  cosine schedule",         "BG_SCHED_STEP_CALIB", "…and from measured STEP cost when one epoch would exceed 80% of the budget (QMCPack)"),
 ("Sec. 4.1  sibling (aux) protocol",  "AUX_MODE",            "cr_matched: siblings archived by the same base compressor at the nearest CR level; train = infer"),
 ("Sec. 4.1  sibling CR levels",       "AUX_CR_LEVELS",       "candidate levels; the one closest (log-CR) to the target's CR is used"),
 ("Sec. 4.1  FFT metric",              "N_FFT_SLICES",        "None = every slice (full-volume FFT magnitude / phase L1 error)"),
 ("Sec. 4.1  SPERR grid cap",          "SPERR_MAX_CR",        "SPERR operating points capped at CR 500"),
 ("Sec. 4.1  NeurLZ baseline",         "NEURLZ_FEATURES",     "BasicUNet features (parameter-matched to ours; in_ch = 1 + #siblings)"),
 ("Sec. 4.1  NeurLZ baseline",         "NEURLZ_LR",           "NeurLZ lr (their recipe), min-max normalization, Adam"),
 ("Sec. 4.1  NeurLZ baseline",         "NEURLZ_BATCH",        "NeurLZ batch (slices)"),
 ("Sec. 4.1  NeurLZ baseline",         "NEURLZ_EPOCH_MULT",   "×1 → NeurLZ gets the SAME wall-clock budget as ours (Phase 1 + Phase 2)"),
 ("Sec. 4.1  QMCPack model budget",    "QMC_PARAMS",          "35 000 (env SPERR_QMC_PARAMS)"),
 ("determinism",                       "SEED",                "torch / numpy / python seed"),
 ("determinism",                       "DETERMINISTIC",       "0 = cuDNN autotune ON (paper); SPERR_DET=1 for deterministic kernels"),
]
print(f"{'paper element':36s} {'constant':24s} {'value':44s} note")
for sec, name, note in rows:
    print(f"{sec:36s} {name:24s} {const(name)[:44]:44s} {note}")
""")

md(r"""
## 2. Method ↔ code map (Sec. 3)

| paper | code | notes |
|---|---|---|
| Sec. 3.1 slice-wise residual model, per-field z-score input normalization, normalized-residual target | `base_script/bg_stage.py::train_bg_only`, `bg_field_norm="zscore"` in `build_bg_only_cfg` | the target is the SZ3/SPERR residual `X − X′`, normalized (Sec. 3.2 derivation) |
| Sec. 3.2 error-bound guarantee | `base_script/bg_stage.py::run_bg_inference` (clamp to `[X′−eb, X′+eb]`) | decoder-side clamp ⇒ `|X̂ − X| ≤ 2·eb` worst case; the best-weights guard returns `X′` if no epoch beat it |
| Sec. 3.3 split-band frequency loss (low/mid/high + FFT magnitude/phase) | `base_script/frequency_losses.py`, `cfg.bg_split_mode="three"`, `bg_freq_weight`, `bg_fft_phase_weight` | weights in the table above |
| Sec. 3.4 model & budget | `base_script/siren_fft_backbone_model.py::UNET_Model`, `bg_shard.pick_bg_h_under_budget` | width `bg_h` chosen as the largest that fits the parameter budget; bf16 weights are charged to the CR |
| Sec. 3.5 Phase 1 (BO over lr × slice direction on a strided proxy) | `SPERR_fft.py::bench_field_fft._phase1_best_fast` | proxy = every 8th slice, spatially /4 (NYX), 10 TPE trials, scored on the middle 16 proxy slices; 10 % of the budget |
| Sec. 3.5 Phase 2 (full-resolution training under the remaining 90 %) | `SPERR_fft.py::bench_field_fft._train_residual` → `train_bg_only` | cosine schedule with warmup, horizon planned from the measured epoch/step cost |
| Sec. 4.1 sibling protocol | `SPERR_fft.py::_aux_at_cr_level`, `_use_aux_for_cr` | siblings archived at the CR level nearest the target's by the same base compressor; the **same** decompressed siblings feed training and inference |
| Sec. 4.1 NeurLZ baseline | `SPERR_fft.py::run_neurlz` | MONAI `BasicUNet`, min-max normalization, Adam lr 1e-2, per-epoch shuffle, parameter- and time-matched, same decompressed siblings |
| Sec. 4.1 metrics | `compute_psnr`, `_global_fft_err`, `paper_numbers.py` | PSNR on the field's range; FFT magnitude/phase L1 over every slice; Rel L2 = drange·10^(−PSNR/20)/RMS |
""")

md(r"""
## 3. How to run the pipeline (from scratch)

All six datasets are driven by one script; each `--task` computes one dataset and writes its result to
`SPERR/sperr_fft_cache/<name>__<config-hash>.pkl` (the hash covers every constant in the table above, so a changed
setting never silently reuses an old result; `SPERR_FORCE_RETRAIN=1` overrides a cache hit).

```bash
cd SPERR
# 0) one-time: archive the NYX sibling fields at CR levels 100..500 for both base compressors (CPU only, ~27 min)
python SPERR_fft.py --task aux_prep

# 1) the six paper datasets, sequentially on the paper GPU (never two at once: the runs are wall-clock budgeted)
for T in nyx_b nyx_t nyx_d mag qmcpack miranda; do python -u SPERR_fft.py --task $T > run_$T.log 2>&1; done
#    measured wall time: NYX ~16 min each | Magnetic ~14 min | QMCPack ~43 min | Miranda ~1.5 h  (aux_prep excluded)

# 2) pin the six pkls (so figures/tables are reproducible) and rebuild everything
python paper_numbers.py  PAPER_SEQ_CRMATCHED_CACHES.json        # every number in Sec. 4 / Table 2 / Sec. 4.3
python plot_paper_figs.py PAPER_SEQ_CRMATCHED_CACHES.json        # Fig. 9  (PSNR vs CR, 3x2)  -> overleaf_figures/sperr_cmp_all6_3x2.pdf
python plot_paper_fft.py  PAPER_SEQ_CRMATCHED_CACHES.json        # Fig. 10 (FFT mag|phase vs CR, 3x4) -> overleaf_figures/sperr_fft_all6.pdf
```

Environment knobs (all optional, defaults = paper): `SPERR_SAMPLE_MODE` (sequential | shuffled | random),
`SPERR_LR_MIN` / `SPERR_LR_MAX`, `SPERR_P1_SPLIT`, `SPERR_PHASE_W`, `SPERR_AUX_MODE` (cr_matched | orig),
`SPERR_AUX_CR_LEVELS`, `SPERR_QMC_PARAMS`, `SPERR_TAU_AXIS` / `SPERR_TAU_LR` (trust gates, −1 = off),
`SPERR_STEP_CALIB`, `SPERR_DET`, `SPERR_FORCE_RETRAIN`.

What one operating point does (per rel band, per base compressor):
1. compress the target with SZ3 (`rel`) or SPERR (`--psnr` target) → `X′`, CR, PSNR, FFT errors;
2. swap the sibling fields to the CR level nearest the target's CR (`[aux]` log lines);
3. **Phase 1**: 10 TPE trials of (lr, slice axis) on the strided proxy inside 10 % of the budget → argmax;
4. **Phase 2**: train the residual model at full resolution for the remaining 90 %; evaluate with the error-bounded clamp; keep the best epoch;
5. **NeurLZ** on the same `X′`, same siblings, same total wall-clock (Phase 1 + Phase 2), parameter-matched;
6. record CR (base bytes + bf16 model bytes), PSNR, FFT magnitude/phase errors for base / +Ours / +NeurLZ.

Run logs of the paper run are in `Reproduce/logs/` (one log per task, plus `run_all.log` with start/exit stamps); the exploratory 2026-08-28 chains are under `SPERR/run_logs/2026-08-28/`.
""")

code(r"""
# The exact script that produced everything under Reproduce/ (paper GPU, unattended):
print(open(f"{OUT}/run_all.sh").read())
""")

md(r"""
## 4. Rate–distortion results (Sec. 4.2 — Fig. 9, Fig. 10, Table 2)

Everything below is computed from the pinned caches with the exact definitions used in the paper (`SPERR/paper_numbers.py`):

* **gain** = PSNR(base + X) − PSNR(base) at the same operating point;
* **iso-PSNR CR gain** = CR_ours / CR_ref − 1, where CR_ref is the reference curve's CR at PSNR(ours) (log-CR linear in PSNR; `*` marks extrapolation beyond the reference curve);
* **Rel L2** = ‖x − x̂‖₂ / ‖x‖₂ = drange · 10^(−PSNR/20) / RMS(x);
* **FFT reductions** = 1 − err(ours)/err(ref) for the per-slice FFT magnitude / phase L1 errors;
* **Table 2 row** = the operating point whose base CR is closest to 500.
""")

code(r"""
data = pn.load(PIN)
A = {k: pn.analyse(data[k], k) for k in pn.ORDER if k in data}
for k in pn.ORDER:
    print(f"\n=== {pn.META[k]['name']} ===")
    for side in ("sz3", "sperr"):
        a = A[k][side]; base = side.upper()
        print(f" [{base}]  {'CR':>8s} {'PSNR base':>10s} {'+NeurLZ':>9s} {'+Ours':>9s} | {'gain ours':>9s} {'gain NeurLZ':>11s} | {'iso-PSNR CR gain':>16s} {'FFT-mag red.':>12s} {'FFT-pha red.':>12s}")
        for i in range(len(a['crb'])):
            print(f"        {a['crb'][i]:8.1f} {a['psb'][i]:10.2f} {a['psn'][i]:9.2f} {a['psp'][i]:9.2f} | {a['gain_p'][i]:+9.2f} {a['gain_n'][i]:+11.2f} | "
                  f"{100*a['iso_b_pct'][i]:+15.0f}%{'*' if a['iso_b_ext'][i] else ' '} {100*a['mag_red_b'][i]:+11.0f}% {100*a['pha_red_b'][i]:+11.0f}%")
""")

code(r"""
# Fig. 9 and Fig. 10, regenerated from the pin (≈10 s, CPU only)
for script in ("plot_paper_figs.py", "plot_paper_fft.py"):
    out = subprocess.run([sys.executable, f"{SPERR}/{script}", PIN], capture_output=True, text=True, cwd=SPERR)
    print(out.stdout.strip().splitlines()[-1])
display(Markdown("**Fig. 9 — PSNR vs effective CR** (`Reproduce/figures/Fig9_psnr_vs_cr.pdf`)"))
display(Image(f"{FIGS}/sperr_cmp_all6_3x2.png", width=700))
display(Markdown("**Fig. 10 — FFT magnitude | phase error vs effective CR** (`Reproduce/figures/Fig10_fft_error_vs_cr.pdf`)"))
display(Image(f"{FIGS}/sperr_fft_all6.png", width=1000))
""")

code(r"""
# Table 2 (operating point at CR ≈ 500) and the six-dataset summary lines quoted in Sec. 4.2 / 4.3 / Abstract
print(f"{'dataset':24s} {'base':6s} {'CR ours':>7s} {'PSNR base':>9s} {'+NeurLZ':>8s} {'+Ours':>8s} {'gain':>6s} {'vs NeurLZ':>9s} "
      f"{'RelL2 base':>10s} {'RelL2 NeurLZ':>12s} {'RelL2 ours':>10s} {'RelL2 red.':>10s} {'iso-CR gain':>11s} {'FFT-mag red.':>12s}")
for k in pn.ORDER:
    for side in ("sz3", "sperr"):
        a = A[k][side]; i = a["i500"]
        print(f"{pn.META[k]['name']:24s} {side.upper():6s} {a['crp'][i]:7.0f} {a['psb'][i]:9.2f} {a['psn'][i]:8.2f} {a['psp'][i]:8.2f} "
              f"{a['gain_p'][i]:+6.2f} {a['psp'][i]-a['psn'][i]:+9.2f} {a['rel_b'][i]:10.3e} {a['rel_n'][i]:12.3e} {a['rel_p'][i]:10.3e} "
              f"{100*(1-a['rel_p'][i]/a['rel_b'][i]):+9.1f}% {100*a['iso_b_pct'][i]:+10.0f}% {100*a['mag_red_b'][i]:+11.0f}%")
print()
for side in ("sz3", "sperr"):
    g  = [A[k][side]["gain_p"].mean() for k in pn.ORDER]; gu = [A[k][side]["gain_p"].max() for k in pn.ORDER]
    gn = [A[k][side]["gain_n"].mean() for k in pn.ORDER]; t2 = [A[k][side]["gain_p"][A[k][side]["i500"]] for k in pn.ORDER]
    mg = [A[k][side]["mag_red_b"].mean() for k in pn.ORDER]
    print(f"[{side.upper():5s}] mean gain over the 5 points, per dataset: {' '.join(f'{x:+.2f}' for x in g)} -> six-dataset avg {np.mean(g):+.2f} dB "
          f"| up-to {max(gu):+.2f} dB | NeurLZ avg {np.mean(gn):+.2f} dB | Table-2 avg {np.mean(t2):+.2f} dB | mean FFT-mag reduction {100*np.mean(mg):+.0f}%")
""")

code(r"""
# Sibling (aux) archive used by the NYX runs: one bitstream per (compressor, field, CR level), shared by all three targets.
metas = sorted(glob.glob(f"{CACHE}/aux_streams/*.json"))
print(f"{len(metas)} archived sibling streams in sperr_fft_cache/aux_streams/")
print(f"{'field':22s} {'comp':6s} {'level':>5s} {'CR':>7s} {'knob (SZ3 rel | SPERR psnr)':>28s} {'bytes':>10s}")
for m in metas:
    j = json.load(open(m))
    print(f"{os.path.basename(j['file']):22s} {j['compressor']:6s} {j['level']:5d} {j['cr']:7.1f} {j['knob']:28.4g} {j['nbytes']:10d}")
print("\nDecompressed-sibling PSNRs per level are printed by `python SPERR_fft.py --task aux_prep` (see run_logs/2026-08-28/auxprep.log).")
""")

md(r"""
## 5. Phase-1 search figure (Sec. 3.5 / Fig. 8)

`BO_Adaptive/nyx_miranda.ipynb` (SZ3 base) and `BO_Adaptive/nyx_miranda_sperr.ipynb` (SPERR base) run the *standalone* two-phase
study: Phase 1 = 10 TPE trials over lr ∈ [1e-3, 1e-2] (log) × slice axis on the proxy (10 % of the budget, no timeout),
Phase 2 = every (axis, lr) candidate trained at full resolution so the proxy's pick can be compared with the true optimum.
Results are pickled to `BO_Adaptive/bo_results/{sz3_nyx,sperr_mir}.pkl`; `BO_Adaptive/bo_combined_plot.ipynb` draws the
composite (NYX on SZ3 at CR≈440 left, Miranda on SPERR at CR≈145 right) → `Reproduce/figures/Fig8_bo_phase1_phase2.pdf` (pickles in `Reproduce/results/bo/`).
Other lr windows explored are archived as `bo_results/lr5e-4_3e-3/`, `bo_results/lr1e-3_1e-2/` and the `*_lr*.pdf` backups.

**2026-08-28 rerun under the paper-final sibling protocol.** The original study fed the NYX model *lossless* siblings; both notebooks now
load the CR-matched decompressed siblings exactly as `SPERR_fft.py` does (`BO_AUX_MODE=cr_matched`, streams from `aux_streams/`) and take the
slice-sampling mode from `BO_SAMPLE_MODE`. Results: `bo_results/sz3_nyx_crmatched_{sequential,shuffled}.pkl`, `bo_results/sperr_mir_shuffled.pkl`
(`sperr_mir.pkl` = sequential; Miranda has no siblings); the lossless-sibling originals are kept in `bo_results/orig_aux/`. Composites:
`NYX_SZ3_Miranda_SPERR_1x4_crmatched_{seq,shuf}.pdf` (`bo_combined_plot.ipynb` with `BO_NYX_PKL` / `BO_MIR_PKL` / `BO_OUT_PDF`).
""")

code(r"""
for tag in ("sz3_nyx_final", "sperr_mir_final"):
    p = f"{CACHE}/bo/{tag}.pkl"
    if not os.path.isfile(p):
        continue
    r = pickle.load(open(p, "rb"))
    fp = {k: float(v) for k, v in r["final_psnr"].items()}; best = max(fp, key=fp.get); pk = (r["best_lr"], r["best_direction"])
    print(f"{tag:30s} {r['dataset_name']:8s} base CR {r['sz_cr']:6.1f} | Phase-1 pick {pk[1]} @ lr {pk[0]:.2e} -> {fp[pk]:.2f} dB "
          f"| full-resolution best {best[1]} @ {best[0]:.2e} = {fp[best]:.2f} dB | gap {fp[best]-fp[pk]:.2f} dB | {r['n_done']} trials, Phase-1 {r['phase1_elapsed']:.1f} s, Phase-2 {r['phase2_time']:.0f} s")
display(Markdown("**Fig. 8** (`Reproduce/figures/Fig8_bo_phase1_phase2.pdf`)"))
display(Image(f"{FIGS}/Fig8_bo_phase1_phase2.png", width=1000))
""")

md(r"""
## 6. Model-size scaling at matched epochs (Sec. 4.x — iso-epoch figure)

`Model_parameter_Scaling/nyx_miranda_isoepoch.ipynb` (NYX baryon + Miranda; Miranda budgets 75k/136k/240k/400k/500k params) and
`Model_parameter_Scaling/nyx_temp_qmcpack_isoepoch.ipynb` (NYX temperature + QMCPack, 3k–30k params) train every model size for the
same `FIXED_EPOCHS = 10` at each of the five rel bands, so the comparison isolates capacity from wall-clock. Outputs:
`psnr_vs_cr_isoepoch_nyx_miranda500k.pdf`, `psnr_vs_cr_isoepoch_nyxtemp_qmcpack.pdf`, `psnr_vs_cr_isoepoch_nyx_baryon_temp.pdf`.
""")

code(r"""
for f in ("psnr_vs_cr_isoepoch_nyx_baryon_temp", "psnr_vs_cr_isoepoch_nyx_miranda500k", "psnr_vs_cr_isoepoch_nyxtemp_qmcpack"):
    display(Markdown(f"`Reproduce/figures/scaling/{f}.pdf`")); display(Image(f"{FIGS}/scaling/{f}.png", width=800))
""")

md(r"""
## 7. Timing, overhead and the storage-system view (Table 3, Sec. 4.x)

* **Table 3 (Overhead = (Train + Infer) / (Comp + Decomp))** uses the per-point Phase-1 + Phase-2 wall time recorded in the caches
  (`[time] phase1(BO) … | phase2(train) …` log lines) and the inference time of one full-volume pass.
* **CR ≈ 300 one-epoch timing on the paper GPU** (`benchmarks/cr300_1epoch_timing.py`): SZ3 compress / decompress, one epoch of ours
  (512 steps at batch 1) and one inference pass on NYX baryon density.
* **End-to-end I/O benchmark on an NVMe SSD** (`benchmarks/fast_io_bench.py` → `fast_io_bench.json`): raw write/read vs SZ3 vs
  SZ3 + NeurLZ (100 epochs) vs SZ3 + ours (1 epoch), all at effective CR ≈ 300.
""")

code(r"""
print("\n".join(open(f"{OUT}/benchmarks/cr300_1epoch_timing.log").read().strip().splitlines()[-9:]))
print()
J = json.load(open(f"{OUT}/benchmarks/fast_io_bench.json"))
print(f"{'pipeline':26s} {'comp s':>7s} {'write s':>8s} {'read s':>7s} {'decomp s':>9s} {'train s':>8s} {'infer s':>8s} {'total s':>8s} {'stream':>10s} {'CR':>6s} {'PSNR':>7s}")
for r in J:
    tot = r['comp']+r['write']+r['read']+r['decomp']+r['train']+r['infer']
    print(f"{r['name']:26s} {r['comp']:7.2f} {r['write']:8.3f} {r['read']:7.3f} {r['decomp']:9.2f} {r['train']:8.2f} {r['infer']:8.2f} {tot:8.2f} {r['bytes']:10d} {r['cr']:6.1f} {r['psnr']:7.2f}")
""")

md(r"""
## 8. Multi-GPU / parallel experiment (Table 4)

`MultiGPU_DDP/parallel_compute.ipynb` trains one model with `torch.nn.DataParallel` over four GPUs (`bg_batch = 4`, `bg_sample_mode = z_shard`:
the four slices of a step come from four z-shards) and compares against one GPU at **equal wall-clock** (the paper's iso-quality/time framing),
on NYX dark-matter density. `MultiGPU_DDP/data_parallel_for_{NYX,Miranda}.py` are the `torchrun` DDP variants; `shard_expert.ipynb` is the
per-z-chunk sharded-expert ablation.

## 9. Ablations and discussion figures

| paper element | notebook / script | output |
|---|---|---|
| frequency head & split-band loss ablation | `frequency_head_loss/frequency_loss.ipynb`, `frequency_head.ipynb`, `fft_err.ipynb` | figures in that folder |
| normalization ablation (z-score vs min-max, Sec. 3.2 / Discussion) | `Normalization/normalization.ipynb`; `Discussion/gen_norm_discussion.py` | `Normalization/ablation_norm_temperature.pdf`, `Discussion/norm_hist_nyx_baryon*.pdf` |
| NeurLZ training trajectory (Discussion) | `Discussion/gen_neurlz_traj.py` → `neurlz_traj_plot.ipynb` | `Discussion/neurlz_traj_nyx_baryon.pdf` |
| bf16 vs fp32 weights | `bf_16_vs_32/bf_16.ipynb` | — |
| 2.5-D slab model variant | `25Dmodel/25Dslab.ipynb` | — |
| qualitative slices / zooms (Miranda at CR 1000 / 2000 on SPERR) | `Final_visualization/make_viz_data.ipynb` (cell `miranda-sperr-hicr`), `figures/visualization.ipynb` | `/storage/sam/Final_visualization/miranda_temperature_aligned/` |

## 10. Provenance

* `Reproduce/results/`: the six pinned result pickles of the paper-final run + `PAPER_FINAL_CACHES.json`; the draft's v3 pickles + `PAPER_V3_CACHES.json`
  (for the old→new tables); `paper_numbers.txt` / `paper_numbers_vs_v3.txt` (every number in Sec. 4); `bo/` = the Fig. 8 study pickles.
* `Reproduce/figures/`: `Fig8_bo_phase1_phase2.pdf`, `Fig9_psnr_vs_cr.pdf`, `Fig10_fft_error_vs_cr.pdf` (+ PNG previews), `scaling/` = the iso-epoch model-size figures.
* `Reproduce/logs/`: one log per task of the run, `run_all.log` with start/exit stamps, `RUN_START`, the executed Fig. 8 notebooks.
* `Reproduce/benchmarks/`: the CR≈300 timing and NVMe I/O benchmark scripts, logs and JSON.
* `Reproduce/run_all.sh` reproduces all of the above from scratch; `Reproduce/collect.py` rebuilds this folder (and this notebook) from the caches.
* Repo-wide: `SPERR/sperr_fft_cache/` keeps every run (`<dataset>__<hash>.pkl`, one hash per configuration) and the sibling archive `aux_streams/`;
  `SPERR/overleaf_figures/` holds the figure copies handed to Overleaf; `SPERR/run_logs/` the exploratory chains.
""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
import os
OUTDIR = os.environ.get("ADAMIT_REPRODUCE_DIR", "/home/sam/Halo_Finder/Final_design/Reproduce")
out = f"{OUTDIR}/REPRODUCE.ipynb"
for c in nb["cells"]:
    c["source"] = c["source"].replace("__ADAMIT_OUT__", OUTDIR)
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
