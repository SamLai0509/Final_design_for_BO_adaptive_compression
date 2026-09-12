"""Build an ori.h5-shaped HDF5 shell with a reconstructed baryon_density
field swapped in, so gimlet2's sim_stats.ex can compute its P(k) the same
way it does for ori.h5.

Usage: python3 make_recon_h5.py <recon.f32> <output.h5>
"""
import shutil
import sys

import h5py
import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))  # Gimlet build lives in $GIMLET_DIR


def main():
    recon_path, out_path = sys.argv[1], sys.argv[2]

    shutil.copyfile(f"{HERE}/h5/ori.h5", out_path)

    with h5py.File(out_path, "r+") as f:
        shape = tuple(f["domain"].attrs["shape"])
        data = np.fromfile(recon_path, dtype=np.float32).reshape(shape)
        f["native_fields/baryon_density"][...] = data

    print(f"wrote {out_path} with baryon_density from {recon_path}")


if __name__ == "__main__":
    main()
