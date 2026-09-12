"""End-to-end P(k) comparison: build an h5 shell for each reconstructed
.f32 field (see make_recon_h5.py), run gimlet2's sim_stats.ex on it, then
report the per-bin relative error in P(k) against h5/ori.h5 for k < K_MAX.

To compare different data: edit RECONS below (tag -> .f32 path) and/or
K_MAX, then run: python3 compare_pk.py
"""
import subprocess
import sys

import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))  # Gimlet build lives in $GIMLET_DIR
SPERR_RECONS = os.environ.get("ADAMIT_RECONS_DIR", "/path/to/recons_cr500")
MPIRUN = os.environ.get("MPIRUN", "mpirun")
SIM_STATS = os.path.join(os.environ.get("GIMLET_DIR", "/path/to/gimlet2"), "apps", "sim_stats", "sim_stats.ex")

# tag -> path to the reconstructed baryon_density .f32 file to compare
RECONS = {
    "sz3":          f"{SPERR_RECONS}/NYX_baryon_density__sz3.f32",
    "sz3_ours":     f"{SPERR_RECONS}/NYX_baryon_density__sz3_ours.f32",
    "sz3_neurlz":   f"{SPERR_RECONS}/NYX_baryon_density__sz3_neurlz.f32",
    "sperr":        f"{SPERR_RECONS}/NYX_baryon_density__sperr.f32",
    "sperr_ours":   f"{SPERR_RECONS}/NYX_baryon_density__sperr_ours.f32",
    "sperr_neurlz": f"{SPERR_RECONS}/NYX_baryon_density__sperr_neurlz.f32",
}

K_MAX = 10.0

OUT_DIR = f"{HERE}/ps3d_out"
H5_DIR = os.environ.get("GIMLET_H5_DIR", os.path.join(HERE, "h5"))


def run_sim_stats(h5_path, prefix):
    subprocess.run(
        [MPIRUN, "-n", "1", SIM_STATS, h5_path, prefix],
        check=True, cwd=HERE,
    )


def main():
    run_sim_stats(f"{H5_DIR}/ori.h5", f"{OUT_DIR}/ori_")

    for tag, f32_path in RECONS.items():
        h5_path = f"{H5_DIR}/cmp_{tag}.h5"
        subprocess.run(
            [sys.executable, f"{HERE}/make_recon_h5.py", f32_path, h5_path],
            check=True,
        )
        run_sim_stats(h5_path, f"{OUT_DIR}/{tag}_")

    ori = np.loadtxt(f"{OUT_DIR}/ori_rhob_ps3d.txt")
    k, po = ori[:, 2], ori[:, 3]
    mask = k < K_MAX

    tags = list(RECONS.keys())
    rel_errs = {}
    for tag in tags:
        d = np.loadtxt(f"{OUT_DIR}/{tag}_rhob_ps3d.txt")
        assert np.array_equal(ori[:, 1], d[:, 1]), f"{tag}: bins not aligned with ori.h5"
        rel_errs[tag] = (d[:, 3] - po) / po

    lines = []
    header = f'{"k":>8} {"P(k)_ori":>12}' + "".join(f'  {t + "_relerr":>12}' for t in tags)
    lines.append(header)
    for i in np.where(mask)[0]:
        row = f"{k[i]:8.4f} {po[i]:12.5e}"
        for t in tags:
            row += f"  {rel_errs[t][i]:+12.4%}"
        lines.append(row)

    lines.append("")
    lines.append(f'{"method":>8} {"mean|relerr|":>14} {"max|relerr|":>14} {"at k":>8}')
    for t in tags:
        re = rel_errs[t][mask]
        kk = k[mask]
        lines.append(
            f"{t:>8} {np.mean(np.abs(re)):14.4%} {np.max(np.abs(re)):14.4%} "
            f"{kk[np.argmax(np.abs(re))]:8.4f}"
        )

    report = "\n".join(lines)
    print(report)
    out_txt = f"{HERE}/pk_relerr_cr500_k_lt_10.txt"
    with open(out_txt, "w") as f:
        f.write(report + "\n")
    print(f"\nwrote {out_txt}")


if __name__ == "__main__":
    main()
