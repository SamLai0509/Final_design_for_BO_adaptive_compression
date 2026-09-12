#!/bin/bash
# No-aux (single-field) ablation of the three NYX fields on the 5090, pinned to 4 cores / nice 19 so the
# RTX 6000 paper chain is barely touched. Indicative numbers only (budgeted training on a throttled box).
REPO=${ADAMIT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}; PY=${PYTHON:-python}; L=$REPO/Reproduce/logs
cd $REPO/sec_4_evaluation
echo "[noaux] start $(date)" >> $L/run_noaux_5090.log
for T in nyx_b nyx_t nyx_d; do
  echo "[noaux] === $T start $(date) ===" >> $L/run_noaux_5090.log
  CUDA_VISIBLE_DEVICES=1 SPERR_USE_AUX=0 taskset -c 16-19 nice -n 19 $PY -u SPERR_fft.py --task $T > $L/noaux5090_$T.log 2>&1
  echo "[noaux] === $T exit=$? $(date) ===" >> $L/run_noaux_5090.log
done
echo "[noaux] ALL DONE $(date)" >> $L/run_noaux_5090.log
