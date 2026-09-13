#!/usr/bin/env bash
# The three NYX rows of the paper table, under the corrected sibling protocol.
#
#   * siblings are the ARCHIVED decompressed volumes at the closest CR level,
#     the same ones a decoder would have (--aux-mode cr_matched)
#   * training runs to the paper's wall-clock budget (10 s for NYX), epochs
#     only as a cap (--iso-time)
#   * both base codecs, 1 GPU and 4 GPUs
#
#   bash run_nyx_paper.sh              # everything
#   bash run_nyx_paper.sh nyx_b sz3    # one field, one codec
set -uo pipefail
cd "$(dirname "$0")"

OUT=${OUT:-bench_out/table_nyx_paper}
CR=${CR:-500}
AUXMODE=${AUXMODE:-cr_matched}
ENH=${ENH:-}
PORT=${PORT:-29870}
PY=${PYTHON:-python}
LAUNCH1=${LAUNCH1:-}   # launcher prefix for 1-GPU steps (empty = run here)
LAUNCH4=${LAUNCH4:-}   # launcher prefix for 4-GPU steps
LOG_DIR=${LOG_DIR:-/tmp}
TASKS=${1:-"nyx_d nyx_t nyx_b"}
CODECS=${2:-"sz3 sperr"}

mkdir -p "$OUT"
common=(--cr "$CR" --iso-time 1 --out "$OUT" --aux-mode "$AUXMODE")
[[ -n "$ENH" ]] && common+=(--aux-enhanced-dir "$ENH")

for codec in $CODECS; do
  for t in $TASKS; do
    for ng in 1 4; do
      echo "=================== $t / $codec / ${ng}gpu ==================="
      log="$LOG_DIR/nyxp_${t}_${codec}_n${ng}.log"
      if [[ "$ng" == "1" ]]; then
        $LAUNCH1 "$PY" bench_compress_table.py --dataset "$t" --codec "$codec" \
          --tag "${t}_${codec}_n1" "${common[@]}" 2>&1 | tee "$log" | grep -E "\[aux\]|\[setup\]|\[done\]|Error"
      else
        $LAUNCH4 "$PY" -m torch.distributed.run --nproc_per_node=4 --master_port="$PORT" \
          bench_compress_table.py --dataset "$t" --codec "$codec" \
          --tag "${t}_${codec}_n4" "${common[@]}" 2>&1 | tee "$log" | grep -E "\[setup\]|\[done\]|Error"
      fi
    done
  done
done
echo "=================== done -> $OUT ==================="
