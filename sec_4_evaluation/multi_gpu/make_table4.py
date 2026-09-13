"""Rebuild the paper's Table 4 (one vs. four A100 GPUs at matched PSNR) from the
run JSONs in bench_out/.

Protocol (Section 4.4): the single GPU trains for the dataset's wall-clock
budget (10 s for 512^3 fields, 60 s QMCPack, 80 s Miranda) and the PSNR it
reaches becomes the target; the four-GPU DDP run keeps the same slices per
optimizer update and is timed to that target. Compression = base encoding +
training, decompression = base decoding + one full-volume inference (z-slab
data-parallel on four GPUs). Speedup = single-GPU / four-GPU
(training + inference) time.

Which run backs which row is pinned in ROWS. Codec times for SPERR are the
single-thread measurements of sperr_omp1_probe.py (the runs themselves were
timed with 4 OpenMP threads); inference times come from bench_infer_mgpu.py.
Both are recorded here so the table is reproducible without a GPU.

    python make_table4.py            # LaTeX rows
    python make_table4.py --plain    # aligned text
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bench_out")

BUDGET_S = {"nyx_b": 10.0, "nyx_t": 10.0, "nyx_d": 10.0, "miranda": 80.0, "qmc": 60.0, "mag": 10.0}
LABEL = {"nyx_b": "NYX, baryon density", "nyx_t": "NYX, temperature", "nyx_d": "NYX, dark matter density",
         "miranda": "Miranda", "qmc": "QMCPack", "mag": "Magnetic Reconnection"}

# full-volume inference, seconds, (1 GPU, 4 GPUs z-slab) -- bench_infer_mgpu.py, median of 3
INFER = {"nyx_b": (1.339, 0.827), "nyx_t": (1.339, 0.827), "nyx_d": (1.339, 0.827),
         "mag": (1.167, 0.749), "miranda": (20.278, 8.568), "qmc": (2.334, 1.103)}
# single-thread sperr3d (encode, decode) at each row's operating point -- sperr_omp1_probe.py
SPERR_CODEC = {"nyx_b": (3.31, 2.10), "nyx_t": (3.22, 2.08), "nyx_d": (3.26, 2.06),
               "miranda": (24.73, 16.35), "qmc": (4.92, 2.31), "mag": (3.32, 2.08)}

# (codec, dataset, single-GPU json, four-GPU json)
ROWS = [
    ("sz3", "nyx_b",   "table_nyx_enh_r2/nyx_b_sz3_n1",      "table_n4_fixed/nyx_b_sz3_n4"),
    ("sz3", "nyx_t",   "table_tempA_n1/nyx_t_sz3_n1",        "table_tempA_n4/nyx_t_sz3_n4"),
    ("sz3", "nyx_d",   "table_nyx_enh/nyx_d_sz3_n1",         "table_n4_fixed/nyx_d_sz3_n4"),
    ("sz3", "miranda", "table_rand_bo/miranda_n1",           "table_n4_seedfix/miranda_sz3_n4"),
    ("sz3", "qmc",     "table_qmc1024_n1/qmc_sz3_n1",        "table_qmc1024_n4/qmc_sz3_n4"),
    ("sz3", "mag",     "table_rand/mag_n1",                  "table_n4_seedfix/mag_sz3_n4"),
    ("sperr", "nyx_b",   "table_nyx_enh_r3/nyx_b_sperr_n1",  "table_n4_fixed/nyx_b_sperr_n4"),
    ("sperr", "nyx_t",   "table_tempA_n1/nyx_t_sperr_n1",    "table_tempA_n4/nyx_t_sperr_n4"),
    ("sperr", "nyx_d",   "table_nyx_enh/nyx_d_sperr_n1",     "table_n4_fixed/nyx_d_sperr_n4"),
    ("sperr", "miranda", "table_mir_fixlr_n1/miranda_sperr_n1", "table_mir_fixlr_n4/miranda_sperr_n4"),
    ("sperr", "qmc",     "table_qmc1024_n1_r2/qmc_sperr_n1", "table_qmc1024_n4_r2/qmc_sperr_n4"),
    ("sperr", "mag",     "table_sperr_rand/mag_n1",          "table_n4_seedfix/mag_sperr_n4"),
]
CODEC_NAME = {"sz3": "SZ3", "sperr": "SPERR"}


def load(rel):
    with open(os.path.join(OUT, rel + ".json")) as f:
        return json.load(f)


def crossing(t, p, target):
    """First time the best-so-far PSNR reaches target (linear interpolation)."""
    t = np.asarray(t, float); p = np.maximum.accumulate(np.asarray(p, float))
    if p[-1] < target:
        return None
    i = int(np.argmax(p >= target))
    if i == 0 or p[i] == p[i - 1]:
        return float(t[i])
    return float(t[i - 1] + (target - p[i - 1]) / (p[i] - p[i - 1]) * (t[i] - t[i - 1]))


def row(codec, ds, f1, f4):
    a, b = load(f1), load(f4)
    T = BUDGET_S[ds]
    t1v = np.asarray(a["time"], float); p1 = np.maximum.accumulate(np.asarray(a["psnr"], float))
    t1 = min(T, float(t1v[-1]))
    target = float(np.interp(t1, t1v, p1))
    t4 = crossing(b["time"], b["psnr"], target)
    I1, I4 = INFER[ds]
    if codec == "sperr":
        e1 = e4 = SPERR_CODEC[ds][0]; d1 = d4 = SPERR_CODEC[ds][1]
    else:
        e1, e4 = a["t_sz_compress"], b["t_sz_compress"]
        d1, d4 = a["t_sz_decompress"], b["t_sz_decompress"]
    dag = t4 is None
    if dag:
        t4 = float(np.asarray(b["time"], float)[-1])
    return dict(label=LABEL[ds], psnr=target, comp1=e1 + t1, dec1=d1 + I1, comp4=e4 + t4, dec4=d4 + I4,
                speedup=None if dag else (t1 + I1) / (t4 + I4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", action="store_true")
    args = ap.parse_args()
    cur = None
    for codec, ds, f1, f4 in ROWS:
        if codec != cur:
            cur = codec
            if args.plain:
                print(f"--- {CODEC_NAME[codec]} + Ours")
            else:
                if codec == "sperr":
                    print(r"    \midrule")
                print(r"    \multicolumn{7}{@{}l}{\textit{%s + Ours}} \\" % CODEC_NAME[codec])
        r = row(codec, ds, f1, f4)
        spd = "--$^\\dagger$" if r["speedup"] is None else f"{r['speedup']:.2f}$\\times$"
        if args.plain:
            spd = "  --  " if r["speedup"] is None else f"{r['speedup']:.2f}x"
            print(f"  {r['label']:26s} {r['psnr']:7.2f} {r['comp1']:7.2f} {r['dec1']:6.2f} {r['comp4']:7.2f} {r['dec4']:6.2f}  {spd}")
        else:
            print(f"    {r['label']:26s} & {r['psnr']:6.2f} & {r['comp1']:6.2f} & {r['dec1']:5.2f} & "
                  f"{r['comp4']:5.2f} & {r['dec4']:5.2f} & {spd} \\\\")


if __name__ == "__main__":
    main()
