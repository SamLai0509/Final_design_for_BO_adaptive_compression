#!/bin/sh
# NeurLZ time-to-match, generous caps + stop when AdaMit's PSNR is reached.
REPO=${ADAMIT_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}; PY=${PYTHON:-python}
OUT=$REPO/Reproduce/experiment/neurlz_long; L=$OUT/run_2000.log; cd $REPO/sec_4_evaluation
for ds in nyx_b nyx_t nyx_d mag qmcpack miranda; do
  echo "[nlz] === $ds start $(date) ===" >> $L
  SPERR_NLZ_DATASETS=$ds SPERR_NLZ_CAPS="nyx:2000,mag:1500,qmcpack:1500,miranda:2000" SPERR_NLZ_STOP_AT_TARGET=1 \
    SPERR_NLZ_OUT=$OUT/neurlz_long_2000.json $PY -u SPERR_fft.py --task neurlz_long >> $L 2>&1
  echo "[nlz] === $ds exit=$? $(date) ===" >> $L
done
echo "[nlz] ALL DONE $(date)" >> $L
