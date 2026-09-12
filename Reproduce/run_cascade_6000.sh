#!/bin/bash
# Cross-field cascade on the paper GPU (invoked from run_after_chain.sh; nothing else running).
# Every stage's inputs are decoder-side: a field enhanced by an earlier stage is fed ENHANCED to the
# later ones (SPERR_AUX_ENHANCED_DIR), everything else stays decompressed at the matched CR level.
#   stage 1 (shared):  DMD          <- decompressed siblings                    -> export enhanced DMD
#   order A:           baryon       <- ENH DMD + dec T + dec velocities          -> export enhanced baryon
#                      temperature  <- ENH DMD + ENH baryon + dec velocities
#   order B:           temperature  <- ENH DMD + dec baryon + dec velocities     -> export enhanced T
#                      baryon       <- ENH DMD + ENH T + dec velocities
REPO=/home/sam/Halo_Finder/Final_design; PY=/home/sam/miniconda3/bin/python; L=$REPO/Reproduce/logs
C=/storage/sam/Final_visualization/cascade_2026-08-29; mkdir -p $C/stage1 $C/A $C/B
cd $REPO/SPERR; echo "[cascade] start $(date)" >> $L/run_cascade_6000.log

run_stage () {   # $1 tag  $2 task  $3 enhanced-dir (also export dir)
  echo "[cascade] === $1 ($2) start $(date) ===" >> $L/run_cascade_6000.log
  SPERR_AUX_ENHANCED_DIR=$3 SPERR_SAVE_RECONS_DIR=$3 SPERR_SAVE_RECONS_ONLY=ours SPERR_FORCE_RETRAIN=1 \
    $PY -u SPERR_fft.py --task $2 > $L/cascade_$1.log 2>&1
  echo "[cascade] === $1 ($2) exit=$? $(date) ===" >> $L/run_cascade_6000.log
}

run_stage stage1_nyx_d nyx_d $C/stage1
for O in A B; do ln -sfn $C/stage1/NYX_dark_matter_density_sz3_cr* $C/stage1/NYX_dark_matter_density_sperr_cr* $C/$O/ 2>/dev/null; done
run_stage A_nyx_b nyx_b $C/A
run_stage A_nyx_t nyx_t $C/A
run_stage B_nyx_t nyx_t $C/B
run_stage B_nyx_b nyx_b $C/B
echo "[cascade] ALL DONE $(date)" >> $L/run_cascade_6000.log
