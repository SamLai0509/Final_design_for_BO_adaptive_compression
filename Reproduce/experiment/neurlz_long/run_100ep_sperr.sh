#!/bin/sh
REPO=/home/sam/Halo_Finder/Final_design; PY=/home/sam/miniconda3/bin/python
OUT=$REPO/Reproduce/experiment/neurlz_long; L=$OUT/run_100ep_sperr.log; cd $REPO/SPERR
for ds in nyx_b nyx_t nyx_d mag miranda qmcpack; do
  echo "[nlz100sp] === $ds start $(date) ===" >> $L
  SPERR_NLZ_SIDE=sperr SPERR_NLZ_DATASETS=$ds SPERR_NLZ_EPOCHS=100 SPERR_NLZ_STOP_AT_TARGET=0 SPERR_NLZ_EVAL_EVERY=10 \
    SPERR_NLZ_OUT=$OUT/neurlz_100ep_sperr.json $PY -u SPERR_fft.py --task neurlz_long >> $L 2>&1
  echo "[nlz100sp] === $ds exit=$? $(date) ===" >> $L
done
echo "[nlz100sp] ALL DONE $(date)" >> $L
