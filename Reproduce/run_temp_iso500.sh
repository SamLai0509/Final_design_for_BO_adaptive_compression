#!/bin/sh
# Fig-1-style temperature panels at iso-CR ~500.
# run1: ours with ENHANCED siblings (cascade A), rel 9.25385e-4 -> base 530.7, ours eff 501.2
# run2: plain protocol, rel 8.2e-4 -> base 501.5 (SZ3 panel), NeurLZ eff 498.3
REPO=/home/sam/Halo_Finder/Final_design; PY=/home/sam/miniconda3/bin/python
L=$REPO/Reproduce/logs/temp_iso500.log; V=/storage/sam/Final_visualization/final_2026-08-29
while pgrep -f "SPERR_fft.p[y]" > /dev/null; do sleep 30; done
cd $REPO/SPERR
echo "[t500] === enhanced run start $(date) ===" >> $L
SPERR_NYX_RELS=9.25385e-4 SPERR_AUX_ENHANCED_DIR=/storage/sam/Final_visualization/cascade_2026-08-29/A \
  SPERR_SAVE_RECONS_DIR=$V/enhanced SPERR_FORCE_RETRAIN=1 $PY -u SPERR_fft.py --task nyx_t >> $L 2>&1
echo "[t500] === enhanced run exit=$? $(date) ===" >> $L
echo "[t500] === plain run start $(date) ===" >> $L
SPERR_NYX_RELS=8.2e-4 SPERR_SAVE_RECONS_DIR=$V SPERR_FORCE_RETRAIN=1 $PY -u SPERR_fft.py --task nyx_t >> $L 2>&1
echo "[t500] === plain run exit=$? $(date) ===" >> $L
echo "[t500] ALL DONE $(date)" >> $L
