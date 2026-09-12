#!/bin/sh
# Aux-quality ladder for the methodology figure: original-aux and no-aux runs of
# NYX baryon + temperature on the paper GPU (cuda:0 = RTX PRO 6000), paper protocol.
REPO=${ADAMIT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}; PY=${PYTHON:-python}
L=$REPO/Reproduce/logs/auxq_ablation.log; cd $REPO/sec_4_evaluation
for task in nyx_b nyx_t; do
  echo "[auxq] === origaux $task start $(date) ===" >> $L
  SPERR_AUX_MODE=orig $PY -u SPERR_fft.py --task $task >> $L 2>&1
  echo "[auxq] === origaux $task exit=$? $(date) ===" >> $L
done
B=sperr_fft_cache/_experiments/2026-08-31_noaux_5090_backup; mkdir -p $B
for p in NYX_baryon_density__1cb97d57d925 NYX_temperature__f8d1ea9d8e9b NYX_dark_matter_density__4cced88aaad9; do
  cp -n sperr_fft_cache/$p.pkl $B/ 2>/dev/null
done
for task in nyx_b nyx_t; do
  echo "[auxq] === noaux $task start $(date) ===" >> $L
  SPERR_USE_AUX=0 SPERR_FORCE_RETRAIN=1 $PY -u SPERR_fft.py --task $task >> $L 2>&1
  echo "[auxq] === noaux $task exit=$? $(date) ===" >> $L
done
echo "[auxq] ALL DONE $(date)" >> $L
