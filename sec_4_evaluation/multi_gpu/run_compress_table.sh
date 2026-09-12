#!/bin/bash
# Drive the 6-dataset compression/decompression timing table.
#
# For each dataset the single-GPU job binary-searches the codec to CR 500 and,
# if BO=1, runs the Phase-1 TPE over (slice axis, lr). Both are written to its
# JSON and handed to the 4-GPU job, so the two columns share an operating point
# and a training configuration and the bisection/search is paid once.
#
#   bash run_compress_table.sh                    # all six, Phase-1 BO on
#   BO=0 OUT=bench_out/table bash run_compress_table.sh   # fixed axis0 + lr x8
#   CODEC=sperr bash run_compress_table.sh        # same six on top of SPERR
#   bash run_compress_table.sh nyx_t mag          # a subset
set -u
cd "$(dirname "$0")"

PY=${PYTHON:-python}
TORCHRUN="$PY -m torch.distributed.run"
BO=${BO:-1}
CODEC=${CODEC:-sz3}
# Default output dir tracks the codec so an SZ3 table is never overwritten by a
# SPERR one -- the JSON filenames are keyed on dataset only.
if [ "$CODEC" = "sz3" ]; then OUT=${OUT:-bench_out/table_bo}
else OUT=${OUT:-bench_out/table_$CODEC}; fi
EPOCHS=${EPOCHS:-10}
PORT=${PORT:-29800}

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(nyx_b nyx_t nyx_d mag qmc miranda)
fi

get() {  # get <json> <key>
    "$PY" -c "
import json
try: print(json.load(open('$1'))['$2'])
except Exception: print('')"
}

for ds in "${DATASETS[@]}"; do
    echo "=================== $ds ==================="

    if [ ! -f "$OUT/${ds}_n1.json" ]; then
        echo "--- $ds : 1 GPU, $EPOCHS epochs, BO=$BO ---"
        srun -p gpuquick --gres=gpu:1 --cpus-per-task=8 --time=01:55:00 \
            "$TORCHRUN" --nproc_per_node=1 --master_port=$((PORT++)) \
            bench_compress_table.py --dataset "$ds" --tag "${ds}_n1" \
            --codec "$CODEC" --epochs "$EPOCHS" --bo "$BO" --out "$OUT" \
            > "/tmp/tb_${ds}_n1.log" 2>&1
        grep -E "\[bo\]|\[phase1\]|\[setup\]|\[codec\]|\[done\]" "/tmp/tb_${ds}_n1.log"
    else
        echo "--- $ds : 1 GPU already done, skipping ---"
    fi

    # The JSON is written on the compute node; give NFS a moment to expose it
    # to this one before deciding the run produced nothing.
    for _ in $(seq 1 20); do
        [ -s "$OUT/${ds}_n1.json" ] && break
        sleep 3
    done
    AXIS=$(get "$OUT/${ds}_n1.json" axis)
    LR=$(get "$OUT/${ds}_n1.json" lr)
    # Hand the 4-GPU job the operating point the 1-GPU job found, in whichever
    # knob this codec exposes: SZ3 bisects an L-inf bound, SPERR a PSNR target.
    if [ "$CODEC" = "sperr" ]; then
        OP=$(get "$OUT/${ds}_n1.json" sperr_q); OPFLAG=--sperr_q
    else
        OP=$(get "$OUT/${ds}_n1.json" rel); OPFLAG=--rel
    fi
    if [ -z "$OP" ] || [ "$OP" = "None" ] || [ -z "$AXIS" ]; then
        echo "!!! $ds : single-GPU run produced no usable JSON, skipping 4-GPU"
        continue
    fi

    if [ ! -f "$OUT/${ds}_n4.json" ]; then
        echo "--- $ds : 4 GPU, $OPFLAG=$OP axis=$AXIS lr=$LR ---"
        srun -p gpuquick --gres=gpu:4 --cpus-per-task=16 --time=01:55:00 \
            "$TORCHRUN" --nproc_per_node=4 --master_port=$((PORT++)) \
            bench_compress_table.py --dataset "$ds" --tag "${ds}_n4" \
            --codec "$CODEC" --epochs "$EPOCHS" "$OPFLAG" "$OP" \
            --axis "$AXIS" --lr_abs "$LR" \
            --out "$OUT" > "/tmp/tb_${ds}_n4.log" 2>&1
        grep -E "\[phase1\]|\[setup\]|\[codec\]|\[done\]" "/tmp/tb_${ds}_n4.log"
    else
        echo "--- $ds : 4 GPU already done, skipping ---"
    fi
done

echo
echo "=================== TABLE ==================="
"$PY" bench_compress_table.py --table "$OUT"/*.json
