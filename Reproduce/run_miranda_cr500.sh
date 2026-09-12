#!/bin/bash
# Standalone repeats of the Miranda CR~500 operating point (SZ3 band 4 + SPERR aligned to the same CR),
# paper config, 3 independent runs -> Reproduce/experiment/miranda_cr500/. Nothing else may run meanwhile.
REPO=${ADAMIT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}; PY=${PYTHON:-python}
L=$REPO/Reproduce/logs; E=$REPO/Reproduce/experiment/miranda_cr500; C=$REPO/sec_4_evaluation/sperr_fft_cache
cd $REPO/sec_4_evaluation; echo "[mir500] start $(date)" >> $L/run_miranda_cr500.log
for i in 1 2 3; do
  echo "[mir500] === rep$i start $(date) ===" >> $L/run_miranda_cr500.log
  SPERR_ONLY_BANDS=4 SPERR_FORCE_RETRAIN=1 $PY -u SPERR_fft.py --task miranda > $E/rep$i.log 2>&1
  echo "[mir500] === rep$i exit=$? $(date) ===" >> $L/run_miranda_cr500.log
  cp "$(ls -t $C/Miranda__*.pkl | head -1)" $E/rep$i.pkl
done
echo "[mir500] ALL DONE $(date)" >> $L/run_miranda_cr500.log
