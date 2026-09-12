"""P(k) relative error for the two experiments, against the paper's rows.

    sz3 / sz3_ours / sz3_neurlz : the Table-3 rows (recons_cr500, base CR 508)
    sz3_ours_530                : Ours on an SZ3 base bisected to CR 530, so
                                  the EFFECTIVE CR (bf16 weights charged) is
                                  ~500 instead of Table 3's 480.9
    neurlz_ep100_recipe         : NeurLZ trained a fixed 100 epochs with its
                                  own recipe (CosineAnnealingLR T_max=1500,
                                  periodic at this length)
    neurlz_ep100_anneal         : same, but one clean cosine over all steps

Writes pk_relerr_exp_k_lt_10.txt; the paper's pk_relerr_cr500_k_lt_10.txt is
left untouched. Reuses ps3d_out/ori_rhob_ps3d.txt if present.
"""
import os
import subprocess
import sys

import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))  # Gimlet build lives in $GIMLET_DIR
SPERR_DIR = os.environ.get("ADAMIT_RECONS_ROOT", "/path/to/recons")
MPIRUN = os.environ.get("MPIRUN", "mpirun")
SIM_STATS = os.path.join(os.environ.get("GIMLET_DIR", "/path/to/gimlet2"), "apps", "sim_stats", "sim_stats.ex")

RECONS = {
    "sz3":                 f"{SPERR_DIR}/recons_cr500/NYX_baryon_density__sz3.f32",
    "sz3_ours":            f"{SPERR_DIR}/recons_cr500/NYX_baryon_density__sz3_ours.f32",
    "sz3_neurlz":          f"{SPERR_DIR}/recons_cr500/NYX_baryon_density__sz3_neurlz.f32",
    "sz3_ours_530":        f"{SPERR_DIR}/recons_cr530/NYX_baryon_density__sz3_ours.f32",
    "sz3_ours_540":        f"{SPERR_DIR}/recons_cr540/NYX_baryon_density__sz3_ours.f32",
    "neurlz_ep100":        f"{SPERR_DIR}/recons_exp/NYX_baryon_density__sz3_neurlz_ep100_recipe.f32",
    # seed spread for the recipe variant: is the low-k breach systematic or luck?
    "neurlz_ep100_s18":    f"{SPERR_DIR}/recons_exp/NYX_baryon_density__sz3_neurlz_ep100_recipe_s18.f32",
    "neurlz_ep100_s19":    f"{SPERR_DIR}/recons_exp/NYX_baryon_density__sz3_neurlz_ep100_recipe_s19.f32",
    # same seed 17 run twice: separates the seed's deterministic share (init +
    # shuffle order) from cuDNN nondeterminism at fixed seed
    "neurlz_ep100_r2":     f"{SPERR_DIR}/recons_exp/NYX_baryon_density__sz3_neurlz_ep100_recipe_r2.f32",
    "neurlz_ep100_s20":    f"{SPERR_DIR}/recons_exp/NYX_baryon_density__sz3_neurlz_ep100_recipe_s20.f32",
    "neurlz_ep100_s21":    f"{SPERR_DIR}/recons_exp/NYX_baryon_density__sz3_neurlz_ep100_recipe_s21.f32",
    "neurlz_ep100_s22":    f"{SPERR_DIR}/recons_exp/NYX_baryon_density__sz3_neurlz_ep100_recipe_s22.f32",
    "neurlz_ep100_s23":    f"{SPERR_DIR}/recons_exp/NYX_baryon_density__sz3_neurlz_ep100_recipe_s23.f32",
    # third nondeterministic sample of the stock recipe at seed 17
    "neurlz_ep100_r3":     f"{SPERR_DIR}/recons_exp/NYX_baryon_density__sz3_neurlz_ep100_recipe_r3.f32",
    # seed 17 under --deterministic: bit-reproducible (md5-identical twice);
    # note it lands 3+ dB below the nondeterministic runs (113.5 vs 116.5-118.8)
    "neurlz_ep100_det":    f"{SPERR_DIR}/recons_exp/NYX_baryon_density__sz3_neurlz_ep100_recipe_det1.f32",
    # ---- CR 400 operating point: same recipe/config, easier base ----
    "cr400_sz3":           f"{SPERR_DIR}/recons_cr400/NYX_baryon_density__sz3.f32",
    "cr400_sz3_ours":      f"{SPERR_DIR}/recons_cr400/NYX_baryon_density__sz3_ours.f32",
    "cr400_sz3_neurlz":    f"{SPERR_DIR}/recons_cr400/NYX_baryon_density__sz3_neurlz.f32",
    "cr400_nlz_ep100":     f"{SPERR_DIR}/recons_exp/NYX_baryon_density__cr400_sz3_neurlz_ep100_recipe.f32",
    "cr400_nlz_ep100_s18": f"{SPERR_DIR}/recons_exp/NYX_baryon_density__cr400_sz3_neurlz_ep100_recipe_s18.f32",
    "cr400_nlz_ep100_s19": f"{SPERR_DIR}/recons_exp/NYX_baryon_density__cr400_sz3_neurlz_ep100_recipe_s19.f32",
    # ---- CR 200 operating point ----
    "cr200_sz3":           f"{SPERR_DIR}/recons_cr200/NYX_baryon_density__sz3.f32",
    "cr200_sz3_ours":      f"{SPERR_DIR}/recons_cr200/NYX_baryon_density__sz3_ours.f32",
    "cr200_sz3_neurlz":    f"{SPERR_DIR}/recons_cr200/NYX_baryon_density__sz3_neurlz.f32",
    "cr200_nlz_ep100":     f"{SPERR_DIR}/recons_exp/NYX_baryon_density__cr200_sz3_neurlz_ep100_recipe.f32",
    "cr200_nlz_ep100_s18": f"{SPERR_DIR}/recons_exp/NYX_baryon_density__cr200_sz3_neurlz_ep100_recipe_s18.f32",
    "cr200_nlz_ep100_s19": f"{SPERR_DIR}/recons_exp/NYX_baryon_density__cr200_sz3_neurlz_ep100_recipe_s19.f32",
    # ---- CR 300 operating point ----
    "cr300_sz3":           f"{SPERR_DIR}/recons_cr300/NYX_baryon_density__sz3.f32",
    "cr300_sz3_ours":      f"{SPERR_DIR}/recons_cr300/NYX_baryon_density__sz3_ours.f32",
    "cr300_sz3_neurlz":    f"{SPERR_DIR}/recons_cr300/NYX_baryon_density__sz3_neurlz.f32",
    "cr300_nlz_ep100":     f"{SPERR_DIR}/recons_exp/NYX_baryon_density__cr300_sz3_neurlz_ep100_recipe.f32",
    "cr300_nlz_ep100_s18": f"{SPERR_DIR}/recons_exp/NYX_baryon_density__cr300_sz3_neurlz_ep100_recipe_s18.f32",
    "cr300_nlz_ep100_s19": f"{SPERR_DIR}/recons_exp/NYX_baryon_density__cr300_sz3_neurlz_ep100_recipe_s19.f32",
    # Ours stopped at the parallel-table 10-s single-GPU budget, CR~500
    "cr500_ours_10s":      f"{SPERR_DIR}/recons_exp/NYX_baryon_density__cr500_ours_10s.f32",
    # A T_max=len(steps) scheduler variant was also run (recons_exp/..._anneal.f32,
    # ps3d_out/neurlz_ep100_anneal_*); it squeaks under 1% (max 0.55%) at 117.6 dB.
    # The reported comparison uses NeurLZ's own recipe only, per the author.
}

K_MAX = 10.0
REQUIREMENT = 0.01          # the NYX 1% P(k) requirement line

OUT_DIR = f"{HERE}/ps3d_out"
H5_DIR = os.environ.get("GIMLET_H5_DIR", os.path.join(HERE, "h5"))


def run_sim_stats(h5_path, prefix):
    subprocess.run([MPIRUN, "-n", "1", SIM_STATS, h5_path, prefix],
                   check=True, cwd=HERE)


def main():
    if not os.path.isfile(f"{OUT_DIR}/ori_rhob_ps3d.txt"):
        run_sim_stats(f"{H5_DIR}/ori.h5", f"{OUT_DIR}/ori_")

    for tag, f32 in RECONS.items():
        if not os.path.isfile(f32):
            print(f"!! missing {f32} -- skipped"); continue
        if os.path.isfile(f"{OUT_DIR}/{tag}_rhob_ps3d.txt") and tag.startswith("sz3"):
            # Table-3 rows were already run; only recompute the new ones
            if tag in ("sz3", "sz3_ours", "sz3_neurlz"):
                continue
        h5 = f"{H5_DIR}/cmp_{tag}.h5"
        subprocess.run([sys.executable, f"{HERE}/make_recon_h5.py", f32, h5], check=True)
        run_sim_stats(h5, f"{OUT_DIR}/{tag}_")

    ori = np.loadtxt(f"{OUT_DIR}/ori_rhob_ps3d.txt")
    k, po = ori[:, 2], ori[:, 3]
    mask = k < K_MAX

    tags = [t for t in RECONS if os.path.isfile(f"{OUT_DIR}/{t}_rhob_ps3d.txt")]
    rel = {}
    for t in tags:
        d = np.loadtxt(f"{OUT_DIR}/{t}_rhob_ps3d.txt")
        assert np.array_equal(ori[:, 1], d[:, 1]), f"{t}: bins not aligned"
        rel[t] = (d[:, 3] - po) / po

    lines = [f'{"k":>8} {"P(k)_ori":>12}' + "".join(f'  {t + "_relerr":>22}' for t in tags)]
    for i in np.where(mask)[0]:
        lines.append(f"{k[i]:8.4f} {po[i]:12.5e}"
                     + "".join(f"  {rel[t][i]:+22.4%}" for t in tags))
    lines.append("")
    lines.append(f'{"method":>22} {"mean|relerr|":>14} {"max|relerr|":>14} {"at k":>8}'
                 f' {"bins>1%":>8}  meets 1% req?')
    for t in tags:
        re, kk = rel[t][mask], k[mask]
        n_over = int(np.sum(np.abs(re) > REQUIREMENT))
        lines.append(f"{t:>22} {np.mean(np.abs(re)):14.4%} {np.max(np.abs(re)):14.4%} "
                     f"{kk[np.argmax(np.abs(re))]:8.4f} {n_over:>8}  "
                     f"{'YES' if n_over == 0 else 'no'}")
    report = "\n".join(lines)
    print(report)
    with open(f"{HERE}/pk_relerr_exp_k_lt_10.txt", "w") as f:
        f.write(report + "\n")
    print(f"\nwrote {HERE}/pk_relerr_exp_k_lt_10.txt")


if __name__ == "__main__":
    main()
