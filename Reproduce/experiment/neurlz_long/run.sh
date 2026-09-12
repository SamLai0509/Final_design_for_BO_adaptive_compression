#!/bin/sh
# NeurLZ time-to-match experiment: one process per dataset (memory), serialized on the paper GPU.
REPO=${ADAMIT_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}; PY=${PYTHON:-python}
OUT=$REPO/Reproduce/experiment/neurlz_long; L=$OUT/run.log; cd $REPO/sec_4_evaluation
for ds in nyx_b nyx_t nyx_d mag qmcpack miranda; do
  echo "[nlz] === $ds start $(date) ===" >> $L
  SPERR_NLZ_DATASETS=$ds SPERR_NLZ_CAPS="nyx:300,mag:300,qmcpack:600,miranda:800" SPERR_NLZ_OUT=$OUT/neurlz_long.json \
    $PY -u SPERR_fft.py --task neurlz_long >> $L 2>&1
  echo "[nlz] === $ds exit=$? $(date) ===" >> $L
done
echo "[nlz] ALL DONE $(date)" >> $L
