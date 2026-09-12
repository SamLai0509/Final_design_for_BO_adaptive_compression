#!/usr/bin/env python
"""Collect the paper-final run into Reproduce/ (results, figures as PDF, number tables, logs, notebook).

    python Reproduce/collect.py [--since EPOCH_SECONDS]

Picks, per dataset, the newest sec_4_evaluation/sperr_fft_cache/<name>__*.pkl modified after --since (default: the
timestamp in Reproduce/logs/RUN_START, else 0), pins them as Reproduce/results/PAPER_FINAL_CACHES.json, and
then rebuilds every table/figure from that pin only.
"""
import os, sys, json, glob, shutil, subprocess, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPERR, BO = f"{REPO}/sec_4_evaluation", f"{REPO}/sec_3_5_bayesian_opt"
OUT = os.environ.get("ADAMIT_REPRODUCE_DIR", f"{REPO}/Reproduce")
IS_MAIN = os.path.abspath(OUT) == f"{REPO}/Reproduce"      # only the main deliverable updates repo-wide copies
SUF = os.environ.get("BO_OUT_SUFFIX", "_final")              # Fig. 8 pickle suffix used by the chain
CACHE = f"{SPERR}/sperr_fft_cache"
RES, FIG, LOGS, BEN = f"{OUT}/results", f"{OUT}/figures", f"{OUT}/logs", f"{OUT}/benchmarks"
PY, JUP = sys.executable, os.path.join(os.path.dirname(sys.executable), "jupyter")
for d in (RES, f"{RES}/bo", FIG, f"{FIG}/scaling", LOGS, BEN):
    os.makedirs(d, exist_ok=True)

since = 0.0
if "--since" in sys.argv:
    since = float(sys.argv[sys.argv.index("--since") + 1])
elif os.path.isfile(f"{LOGS}/RUN_START"):
    since = float(open(f"{LOGS}/RUN_START").read().strip())

# ── 1. pin the six result pickles ────────────────────────────────────────────────────────
PREFIX = {"Baryon": "NYX_baryon_density", "Temp": "NYX_temperature", "DMD": "NYX_dark_matter_density",
          "Miranda": "Miranda", "QMC": "QMCPack", "Magnetic": "Magnetic"}
import re
TASK = {"Baryon": "nyx_b", "Temp": "nyx_t", "DMD": "nyx_d", "Miranda": "miranda", "QMC": "qmcpack", "Magnetic": "mag"}
def _stamps():
    """{task: (start, exit)} of the LAST successful run of each task, from run_all.log and run_after_chain.log
    (a task re-run after an OOM lands in the second log). Other experiments (no-aux, cascade, ...) write
    pkls with the same prefixes, so 'newest after RUN_START' is NOT a safe rule."""
    out, cur = {}, {}
    for lf in (f"{LOGS}/run_all.log", f"{LOGS}/run_after_chain.log"):
        if not os.path.isfile(lf):
            continue
        for line in open(lf):
            m = re.search(r"=== (\S+) (start|exit=(\d+)) (.+) ===", line)
            if not m:
                continue
            task, what, code, date = m.group(1), m.group(2), m.group(3), m.group(4).strip()
            t = time.mktime(time.strptime(date, "%a %b %d %I:%M:%S %p %Z %Y"))
            if what == "start":
                cur[task] = t
            elif task in cur and code == "0":
                out[task] = (cur.pop(task), t)
    return out
stamps = _stamps()
pin = {}
for k, p in PREFIX.items():
    win = stamps.get(TASK[k])
    if win:
        fs = [(os.path.getmtime(f), f) for f in glob.glob(f"{CACHE}/{p}__*.pkl") if win[0] - 5 <= os.path.getmtime(f) <= win[1] + 5]
    else:
        fs = [(os.path.getmtime(f), f) for f in glob.glob(f"{CACHE}/{p}__*.pkl") if os.path.getmtime(f) >= since]
    if not fs:
        sys.exit(f"no {p} pkl for task {TASK[k]} (window {win}) -- run not complete?")
    f = max(fs)[1]; shutil.copy2(f, RES); pin[k] = os.path.basename(f)
    print(f"  {k:9s} <- {pin[k]}  ({time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(f)))})")
ov = f"{RES}/PIN_OVERRIDES.json"          # explicit per-dataset overrides (e.g. a merged pkl); they must already be in RES
if os.path.isfile(ov):
    for k, f in json.load(open(ov)).items():
        assert os.path.isfile(f"{RES}/{f}"), f"override {f} missing in {RES}"
        pin[k] = f; print(f"  {k:9s} <- {f}  (override)")
json.dump(pin, open(f"{RES}/PAPER_FINAL_CACHES.json", "w"), indent=1)
if IS_MAIN:
    json.dump(pin, open(f"{CACHE}/PAPER_FINAL_CACHES.json", "w"), indent=1)      # keep the repo-wide pin in step
# the draft's numbers (v3) travel along so old->new tables can be rebuilt from Reproduce/ alone
v3 = json.load(open(f"{CACHE}/PAPER_V3_CACHES.json"))
for f in v3.values():
    shutil.copy2(f"{CACHE}/{f}", RES)
json.dump(v3, open(f"{RES}/PAPER_V3_CACHES.json", "w"), indent=1)

env = dict(os.environ, ADAMIT_CACHE_DIR=RES, ADAMIT_FIG_DIR=FIG, CUDA_VISIBLE_DEVICES="")

# ── 2. number tables ─────────────────────────────────────────────────────────────────────
with open(f"{RES}/paper_numbers.txt", "w") as fh:
    subprocess.run([PY, f"{SPERR}/paper_numbers.py", "PAPER_FINAL_CACHES.json"], stdout=fh, stderr=subprocess.STDOUT, env=env, check=True)
with open(f"{RES}/paper_numbers_vs_v3.txt", "w") as fh:
    subprocess.run([PY, f"{SPERR}/paper_numbers.py", "PAPER_FINAL_CACHES.json", "PAPER_V3_CACHES.json"], stdout=fh, stderr=subprocess.STDOUT, env=env, check=True)

# ── 3. Fig. 9 / Fig. 10 (PDF + PNG) ──────────────────────────────────────────────────────
subprocess.run([PY, f"{SPERR}/plot_paper_figs.py", "PAPER_FINAL_CACHES.json"], env=env, check=True)
subprocess.run([PY, f"{SPERR}/plot_paper_fft.py", "PAPER_FINAL_CACHES.json"], env=env, check=True)
for src, dst in (("sperr_cmp_all6_3x2", "Fig9_psnr_vs_cr"), ("sperr_fft_all6", "Fig10_fft_error_vs_cr")):
    for ext in (".pdf", ".png"):
        shutil.copy2(f"{FIG}/{src}{ext}", f"{FIG}/{dst}{ext}")
        if IS_MAIN:
            shutil.copy2(f"{FIG}/{src}{ext}", f"{SPERR}/overleaf_figures/{src}{ext}")     # Overleaf copies stay in step

# ── 4. Fig. 8 (Phase-1 / Phase-2 study) ──────────────────────────────────────────────────
nyx_pkl, mir_pkl = f"{BO}/bo_results/sz3_nyx{SUF}.pkl", f"{BO}/bo_results/sperr_mir{SUF}.pkl"
for f in (nyx_pkl, mir_pkl):
    if os.path.isfile(f):
        shutil.copy2(f, f"{RES}/bo")
if os.path.isfile(nyx_pkl) and os.path.isfile(mir_pkl):
    subprocess.run([JUP, "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=300",
                    f"{BO}/bo_combined_plot.ipynb", "--output", "_tmp_combined_final.ipynb"],
                   env=dict(env, BO_NYX_PKL=nyx_pkl, BO_MIR_PKL=mir_pkl, BO_OUT_PDF=f"{FIG}/Fig8_bo_phase1_phase2.pdf"),
                   cwd=BO, check=True)
    os.remove(f"{BO}/_tmp_combined_final.ipynb")
    if IS_MAIN:
        shutil.copy2(f"{FIG}/Fig8_bo_phase1_phase2.pdf", f"{BO}/NYX_SZ3_Miranda_SPERR_1x4.pdf")
    subprocess.run(["pdftoppm", "-png", "-r", "70", "-singlefile", f"{FIG}/Fig8_bo_phase1_phase2.pdf", f"{FIG}/Fig8_bo_phase1_phase2"])
else:
    print("  [warn] Fig. 8 pickles missing -- skipped")

# ── 5. figures that are not regenerated here (model-size scaling), benchmarks, logs ─────────
for f in glob.glob(f"{REPO}/sec_3_4_model_scaling/psnr_vs_cr_isoepoch_*.pdf") + glob.glob(f"{REPO}/sec_3_4_model_scaling/psnr_vs_cr_isoepoch_*.png"):
    shutil.copy2(f, f"{FIG}/scaling")
for f in glob.glob(f"{REPO}/benchmarks/*"):
    if os.path.isfile(f) and not f.endswith("make_reproduce_nb.py"):
        shutil.copy2(f, BEN)

# ── 6. the notebook ──────────────────────────────────────────────────────────────────────
subprocess.run([PY, f"{REPO}/Reproduce/make_reproduce_nb.py"], check=True, env=dict(os.environ, ADAMIT_REPRODUCE_DIR=OUT))
subprocess.run([JUP, "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=900",
                f"{OUT}/REPRODUCE.ipynb"], env=env, check=True)
print("collected into", OUT)
