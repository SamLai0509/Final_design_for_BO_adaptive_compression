#!/bin/bash
# Paper-final run, end to end, on the paper GPU (cuda:0 = RTX PRO 6000). Nothing else may use the
# machine while this runs: every stage is wall-clock budgeted.
#   config: sec_4_evaluation/SPERR_fft.py defaults (shuffled slice sampling, lr in [1e-3, 1e-2], CR-matched siblings,
#           10% Phase-1 split, no trust gates) -- see Reproduce/REPRODUCE.ipynb §1 for the full table.
#   ~4.5 h: aux archive (cached, ~2 min) -> qmcpack (~45 min) -> nyx_b/t/d (~16 min each) -> mag (~15 min)
#           -> miranda (~1.5 h) -> Fig. 8 NYX (~5 min) -> Fig. 8 Miranda (~20 min) -> collect (~3 min)
set -u
REPO=${ADAMIT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PYTHON:-python}; JUP=${JUPYTER:-jupyter}
OUT=${ADAMIT_REPRODUCE_DIR:-$REPO/Reproduce}; LOGS=$OUT/logs; mkdir -p $LOGS
if [ "$OUT" = "$REPO/Reproduce" ]; then export BO_OUT_SUFFIX=_final; else export BO_OUT_SUFFIX=_final_$(basename $OUT); fi
date +%s > $LOGS/RUN_START
echo "[run_all] start $(date)" >> $LOGS/run_all.log

cd $REPO/sec_4_evaluation
CUDA_VISIBLE_DEVICES="" $PY SPERR_fft.py --task aux_prep > $LOGS/aux_prep.log 2>&1
echo "[run_all] aux_prep exit=$? $(date)" >> $LOGS/run_all.log
for T in qmcpack nyx_b nyx_t nyx_d mag miranda; do
  echo "[run_all] === $T start $(date) ===" >> $LOGS/run_all.log
  SPERR_FORCE_RETRAIN=1 $PY -u SPERR_fft.py --task $T > $LOGS/$T.log 2>&1   # forced: a fresh sample even if this config was run before
  echo "[run_all] === $T exit=$? $(date) ===" >> $LOGS/run_all.log
done

# Fig. 8: NYX (SZ3 base, CR-matched siblings) and Miranda (SPERR base), same lr window / sampling
cd $REPO/sec_3_5_bayesian_opt
$PY - <<'EOF'
import json
nb = json.load(open("nyx_miranda.ipynb")); nb["cells"] = nb["cells"][:6]
for c in nb["cells"]:
    if c["cell_type"] == "code": c["outputs"] = []; c["execution_count"] = None
json.dump(nb, open("_tmp_bo_nyx_final.ipynb", "w"), indent=1)
nb = json.load(open("nyx_miranda_sperr.ipynb")); nb["cells"] = nb["cells"][:5] + [nb["cells"][6]]
for c in nb["cells"]:
    if c["cell_type"] == "code": c["outputs"] = []; c["execution_count"] = None
json.dump(nb, open("_tmp_bo_mir_final.ipynb", "w"), indent=1)
EOF
for NB in bo_nyx bo_mir; do
  echo "[run_all] === $NB start $(date) ===" >> $LOGS/run_all.log
  BO_DEVICE=cuda:0 $JUP nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 \
    _tmp_${NB}_final.ipynb --output $LOGS/${NB}_final_executed.ipynb > $LOGS/$NB.log 2>&1
  echo "[run_all] === $NB exit=$? $(date) ===" >> $LOGS/run_all.log
done
rm -f _tmp_bo_nyx_final.ipynb _tmp_bo_mir_final.ipynb

cd $REPO
ADAMIT_REPRODUCE_DIR=$OUT $PY Reproduce/collect.py > $LOGS/collect.log 2>&1
echo "[run_all] collect exit=$? $(date)" >> $LOGS/run_all.log
echo "[run_all] ALL DONE $(date)" >> $LOGS/run_all.log
