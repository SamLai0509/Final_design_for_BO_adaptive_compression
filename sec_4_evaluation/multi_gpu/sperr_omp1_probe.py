"""Single-thread sperr3d encode/decode times at each SPERR row's operating
point (the --psnr each run bisected). Median of 3. Training/inference are
untouched -- only the codec halves of Comp./Dec. change."""
import json, sys
import numpy as np
sys.path.append(".")
from bench_compress_table import (DATASETS, NYX_DIR, sperr_compress,
                                  sperr_decompress, sperr_source_path)
import argparse, os, statistics

RUNS = [("nyx_b", "table_nyx_enh_r3/nyx_b_sperr_n1"),
        ("nyx_t", "table_b4_r2/nyx_t_sperr_n1"),
        ("nyx_d", "table_nyx_enh/nyx_d_sperr_n1"),
        ("miranda", "table_sperr_rand/miranda_n1"),
        ("qmc", "table_qmc1024_n1_r2/qmc_sperr_n1"),
        ("mag", "table_sperr_rand/mag_n1")]
ns = argparse.Namespace(tmp="/dev/shm", tag="omp1probe")
for ds, jf in RUNS:
    spec = DATASETS[ds]
    d = json.load(open(f"bench_out/{jf}.json"))
    q = float(d["sperr_q"]); shape = tuple(spec["shape"])
    src, src_tmp = sperr_source_path(spec, ns.tmp, ns.tag)
    if src_tmp:
        from bench_compress_table import load_target
        np.ascontiguousarray(load_target(spec), np.float32).tofile(src)
    bit = f"/dev/shm/omp1_{ds}.bit"; dec = f"/dev/shm/omp1_{ds}.dec"
    tc, td = [], []
    for _ in range(3):
        _, t = sperr_compress(src, shape, q, bit, 1); tc.append(t)
        td.append(sperr_decompress(bit, dec, 1))
    for f in (bit, dec) + ((src,) if src_tmp else ()):
        if os.path.exists(f): os.remove(f)
    print(f"{ds:8s} q={q:8.3f}  encode {statistics.median(tc):6.2f}s  "
          f"decode {statistics.median(td):6.2f}s   (was omp4: "
          f"{d['t_sz_compress']:.2f}/{d['t_sz_decompress']:.2f})")
