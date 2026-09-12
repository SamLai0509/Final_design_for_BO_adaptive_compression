#!/bin/bash
# Fig. 6 (iso-epoch NYX panels) and the fp32-vs-BF16 figure, re-run under the paper-final sibling
# protocol. Queued behind run_after_chain.sh (never overlap Miranda-scale jobs).
REPO=${ADAMIT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}; PY=${PYTHON:-python}; L=$REPO/Reproduce/logs
while pgrep -f "run_after_chain.sh" > /dev/null; do sleep 60; done
echo "[figs] start $(date)" >> $L/run_figs_after.log
cd $REPO/sec_3_4_model_scaling
cp -n psnr_vs_cr_isoepoch_nyx_baryon_temp.pdf psnr_vs_cr_isoepoch_nyx_baryon_temp_origaux.pdf 2>/dev/null
echo "[figs] === isoepoch start $(date) ===" >> $L/run_figs_after.log
$PY -u isoepoch_nyx_rerun.py > $L/isoepoch_nyx_rerun.log 2>&1
echo "[figs] === isoepoch exit=$? $(date) ===" >> $L/run_figs_after.log
cd $REPO/sec_3_4_bf16_storage
cp -n bf16_vs_fp32_1x2.pdf bf16_vs_fp32_1x2_origaux.pdf 2>/dev/null
echo "[figs] === bf16 start $(date) ===" >> $L/run_figs_after.log
$PY -u bf16_rerun.py > $L/bf16_rerun.log 2>&1
echo "[figs] === bf16 exit=$? $(date) ===" >> $L/run_figs_after.log
mkdir -p $REPO/Reproduce/figures/scaling
cp $REPO/sec_3_4_model_scaling/psnr_vs_cr_isoepoch_nyx_baryon_temp.pdf $REPO/sec_3_4_model_scaling/psnr_vs_cr_isoepoch_nyx_baryon_temp.png $REPO/Reproduce/figures/scaling/ 2>/dev/null
cp $REPO/sec_3_4_bf16_storage/bf16_vs_fp32_1x2.pdf $REPO/sec_3_4_bf16_storage/bf16_vs_fp32_1x2.png $REPO/Reproduce/figures/ 2>/dev/null
echo "[figs] ALL DONE $(date)" >> $L/run_figs_after.log
