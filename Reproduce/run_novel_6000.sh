#!/bin/bash
# Ablation: NYX with the velocity siblings dropped (siblings = the two other density/temperature
# fields), on the paper GPU (RTX 6000). Invoked from run_after_chain.sh (nothing else running).
REPO=/home/sam/Halo_Finder/Final_design; PY=/home/sam/miniconda3/bin/python; L=$REPO/Reproduce/logs
cd $REPO/SPERR; echo "[novel] start $(date)" >> $L/run_novel_6000.log
for T in nyx_b nyx_t nyx_d; do
  echo "[novel] === $T start $(date) ===" >> $L/run_novel_6000.log
  SPERR_AUX_FIELDS=baryon_density,temperature,dark_matter_density $PY -u SPERR_fft.py --task $T > $L/novel6000_$T.log 2>&1
  echo "[novel] === $T exit=$? $(date) ===" >> $L/run_novel_6000.log
done
echo "[novel] ALL DONE $(date)" >> $L/run_novel_6000.log
cd $REPO && CUDA_VISIBLE_DEVICES="" $PY Reproduce/experiment/collect_experiments.py > $L/collect_experiments.log 2>&1
echo "[novel] experiments collected exit=$? $(date)" >> $L/run_novel_6000.log
