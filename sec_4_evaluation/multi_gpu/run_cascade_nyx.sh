#!/usr/bin/env bash
# NYX cascade: decode the three target fields in order, feeding each stage's
# corrected output to the stages after it as an ENHANCED sibling.
#
#   stage 1  dark_matter_density  cr_matched siblings   (DMD is the field least
#                                                        sensitive to sibling
#                                                        quality, so it goes
#                                                        first)
#   stage 2  temperature          enhanced by stage 1
#   stage 3  baryon_density       enhanced by stages 1-2 (gains the most)
#
# Every stage exports its reconstruction, so stage N reads whatever stages
# 1..N-1 produced and falls back to the cr_matched archive for the rest.
#
#   bash run_cascade_nyx.sh [1|4]            # GPUs, default 1
# env: OUT=<json dir>  ENH=<export dir>  CR=<target CR>  TRAIN_S=<pure-training budget, s>
#      LAUNCH1 / LAUNCH4: launcher prefix for the 1- and 4-GPU steps (empty = run on
#      this machine; under Slurm e.g. LAUNCH4='srun -p <partition> --gres=gpu:4')
#      LOG_DIR: where per-stage logs go (default /tmp)
set -euo pipefail
cd "$(dirname "$0")"

NG=${1:-1}
OUT=${OUT:-bench_out/table_cascade}
ENH=${ENH:-bench_out/enhanced_siblings}
CR=${CR:-500}
TRAIN_S=${TRAIN_S:-10}
LAUNCH1=${LAUNCH1:-}
LAUNCH4=${LAUNCH4:-}
LOG_DIR=${LOG_DIR:-/tmp}
PORT=${PORT:-29850}
PY=${PYTHON:-python}

mkdir -p "$OUT" "$ENH"

run () {                      # run <task> <aux-mode> <tag>
  local task=$1 mode=$2 tag=$3
  echo "=================== $task ($mode, ${NG} gpu) ==================="
  local args=(--dataset "$task" --cr "$CR" --train-s "$TRAIN_S" --out "$OUT"
              --tag "$tag" --aux-mode "$mode"
              --export-enhanced-dir "$ENH")
  [[ "$mode" == "enhanced" ]] && args+=(--aux-enhanced-dir "$ENH")

  if [[ "$NG" == "1" ]]; then
    $LAUNCH1 "$PY" bench_compress_table.py "${args[@]}" 2>&1 | tee "$LOG_DIR/casc_${task}_n${NG}.log"
  else
    $LAUNCH4 "$PY" -m torch.distributed.run --nproc_per_node="$NG" --master_port="$PORT" \
      bench_compress_table.py "${args[@]}" 2>&1 | tee "$LOG_DIR/casc_${task}_n${NG}.log"
  fi
}

# Stage 1 has no enhanced siblings to read yet, so it runs the paper protocol.
run nyx_d cr_matched "n${NG}"
run nyx_t enhanced   "n${NG}"
run nyx_b enhanced   "n${NG}"

echo "=================== cascade done ==================="
echo "results  : $OUT"
echo "enhanced : $ENH"
ls -1 "$ENH"
