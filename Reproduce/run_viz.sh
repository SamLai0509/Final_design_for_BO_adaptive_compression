#!/bin/bash
# Visualization volumes under the paper-final protocol (same code path as the paper run):
#   NYX baryon density at the CR~500 band (index 4) and NYX temperature at the CR~300 band (index 3),
#   SZ3 side + SPERR side aligned to the same CR; base / +Ours / +NeurLZ / err_* as .f32 + .vtk (ParaView).
# Waits for the paper chain (bash Reproduce/run_all.sh) so the GPU is idle.
REPO=/home/sam/Halo_Finder/Final_design; PY=/home/sam/miniconda3/bin/python
OUT=/storage/sam/Final_visualization/final_2026-08-29; LOG=$REPO/Reproduce/logs
while pgrep -f "bash Reproduce/run_all.sh" > /dev/null; do sleep 60; done
cd $REPO/SPERR
echo "[viz] start $(date)" >> $LOG/run_viz.log
for spec in "nyx_b 4" "nyx_t 3"; do set -- $spec
  echo "[viz] === $1 band $2 start $(date) ===" >> $LOG/run_viz.log
  SPERR_SAVE_RECONS_DIR=$OUT SPERR_ONLY_BANDS=$2 SPERR_FORCE_RETRAIN=1 $PY -u SPERR_fft.py --task $1 > $LOG/viz_$1.log 2>&1
  echo "[viz] === $1 exit=$? $(date) ===" >> $LOG/run_viz.log
done
cp $LOG/run_viz.log $OUT/ 2>/dev/null; echo "[viz] ALL DONE $(date)" >> $LOG/run_viz.log
