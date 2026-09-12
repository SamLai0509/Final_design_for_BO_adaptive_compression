"""Why is SPERR+Ours worse than plain SPERR in P(k), despite better PSNR?

Computes, for each reconstruction:
  - global mean / std vs the original (a large-scale bias check)
  - P(k) of the ERROR FIELD (recon - orig), radially binned

The error-field spectrum is the direct diagnostic: it shows *at which
scales* each method's error lives. PSNR only reports its total integral,
so a method can lower total error while moving more of it to large scales
-- which is exactly what P(k) fidelity at small k cares about.
"""
import numpy as np

import os
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
F32 = f"{HERE}/zarija-gimlet2-91820b6f87fa/f32_data"
SHAPE = (512, 512, 512)
BOX = 14.25779819  # Mpc/h, from ori.h5 domain.size

FIELDS = {
    "sperr":      f"{F32}/baryon_density_sperr_cr500.f32",
    "sperr_ours": f"{F32}/baryon_density_sperr_ours_cr500_10epochs.f32",
    "sz3":        f"{F32}/baryon_density_sz3_cr500_10epochs.f32",
    "ours":       f"{F32}/baryon_density_ours_cr500_10epochs.f32",
}
ORIG = f"{HERE}/Dataset/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.f32"


def radial_power(field, box=BOX, nbins=18, kmax=10.0):
    """|FFT(field)|^2 averaged in log-spaced radial k shells."""
    n = field.shape[0]
    kfund = 2.0 * np.pi / box
    fk = np.fft.rfftn(field.astype(np.float32))
    p3d = (fk.real.astype(np.float64) ** 2 + fk.imag.astype(np.float64) ** 2)
    del fk

    kx = np.fft.fftfreq(n, d=1.0 / n) * kfund
    kz = np.fft.rfftfreq(n, d=1.0 / n) * kfund
    kk = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)

    edges = np.logspace(np.log10(kfund * 0.9), np.log10(kmax), nbins + 1)
    idx = np.digitize(kk.ravel(), edges) - 1
    p = p3d.ravel()
    del p3d, kk

    valid = (idx >= 0) & (idx < nbins)
    counts = np.bincount(idx[valid], minlength=nbins)
    sums = np.bincount(idx[valid], weights=p[valid], minlength=nbins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        pk = sums / counts
    return centers, pk, counts


def main():
    orig = np.fromfile(ORIG, dtype=np.float32).reshape(SHAPE)
    print(f"{'field':>12} {'mean':>12} {'std':>12} {'mean-orig':>12} {'RMS err':>12}")
    print(f"{'orig':>12} {orig.mean():12.6f} {orig.std():12.6f} {'-':>12} {'-':>12}")

    errs = {}
    for tag, path in FIELDS.items():
        x = np.fromfile(path, dtype=np.float32).reshape(SHAPE)
        e = (x.astype(np.float64) - orig.astype(np.float64))
        print(f"{tag:>12} {x.mean():12.6f} {x.std():12.6f} "
              f"{x.mean()-orig.mean():+12.3e} {np.sqrt((e**2).mean()):12.3e}")
        errs[tag] = e.astype(np.float32)
        del x

    print("\nP(k) of the ERROR field (recon - orig), log-binned:")
    res = {}
    for tag, e in errs.items():
        c, pk, cnt = radial_power(e)
        res[tag] = pk
        centers = c

    hdr = f"{'k':>8}" + "".join(f"{t:>14}" for t in FIELDS)
    print(hdr)
    for i in range(len(centers)):
        row = f"{centers[i]:8.3f}"
        for t in FIELDS:
            row += f"{res[t][i]:14.4e}"
        print(row)

    print("\nerror power relative to plain SPERR (>1 = the model ADDED error at that k):")
    variants = [t for t in FIELDS if t != "sperr"]
    print(f"{'k':>8}" + "".join(f"{t:>14}" for t in variants))
    for i in range(len(centers)):
        row = f"{centers[i]:8.3f}"
        for t in variants:
            row += f"{res[t][i]/res['sperr'][i]:14.3f}"
        print(row)


if __name__ == "__main__":
    main()
