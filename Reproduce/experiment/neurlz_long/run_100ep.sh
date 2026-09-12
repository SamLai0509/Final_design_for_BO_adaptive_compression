#!/bin/sh
# NeurLZ at its default 100 epochs, clean timing (evaluate every 10 epochs), one process per dataset.
# Waits for the running QMCPack 100-epoch job before starting (never share the GPU).
REPO=/home/sam/Halo_Finder/Final_design; PY=/home/sam/miniconda3/bin/python
OUT=$REPO/Reproduce/experiment/neurlz_long; L=$OUT/run_100ep.log; cd $REPO/SPERR
while pgrep -f "python -u SPERR_fft[.]py" > /dev/null; do sleep 30; done
for ds in nyx_b nyx_t nyx_d mag miranda; do
  echo "[nlz100] === $ds start $(date) ===" >> $L
  SPERR_NLZ_DATASETS=$ds SPERR_NLZ_EPOCHS=100 SPERR_NLZ_STOP_AT_TARGET=0 SPERR_NLZ_EVAL_EVERY=10 \
    SPERR_NLZ_OUT=$OUT/neurlz_100ep.json $PY -u SPERR_fft.py --task neurlz_long >> $L 2>&1
  echo "[nlz100] === $ds exit=$? $(date) ===" >> $L
done
echo "[nlz100] ALL DONE $(date)" >> $L
