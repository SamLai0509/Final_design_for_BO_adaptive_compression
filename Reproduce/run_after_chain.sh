#!/bin/bash
# Serialized tail of the paper run (nothing may overlap: Miranda 1024^3 peaks near 50 GB host RAM):
#   Miranda rerun (its first attempt was OOM-killed by a concurrent job) -> collect -> viz export
#   -> cross-field cascade -> no-velocity ablation.  All on the RTX 6000.
REPO=/home/sam/Halo_Finder/Final_design; PY=/home/sam/miniconda3/bin/python; L=$REPO/Reproduce/logs
while pgrep -f "bash Reproduce/run_all.sh" > /dev/null || pgrep -f "run_noaux_5090.sh" > /dev/null; do sleep 30; done
echo "[after] start $(date)" >> $L/run_after_chain.log
cd $REPO/SPERR
echo "[after] === miranda start $(date) ===" >> $L/run_after_chain.log
SPERR_FORCE_RETRAIN=1 $PY -u SPERR_fft.py --task miranda > $L/miranda.log 2>&1
echo "[after] === miranda exit=$? $(date) ===" >> $L/run_after_chain.log
cd $REPO
$PY Reproduce/collect.py > $L/collect.log 2>&1
echo "[after] collect exit=$? $(date)" >> $L/run_after_chain.log
bash Reproduce/run_viz.sh;          echo "[after] viz done $(date)" >> $L/run_after_chain.log
bash Reproduce/run_cascade_6000.sh; echo "[after] cascade done $(date)" >> $L/run_after_chain.log
bash Reproduce/run_novel_6000.sh;   echo "[after] no-velocity done $(date)" >> $L/run_after_chain.log
echo "[after] ALL DONE $(date)" >> $L/run_after_chain.log
