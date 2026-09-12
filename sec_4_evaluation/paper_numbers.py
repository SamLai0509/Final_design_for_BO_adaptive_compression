#!/usr/bin/env python
"""Every number the paper quotes, recomputed from a cache pin file.

    python paper_numbers.py [PIN.json] [OLD_PIN.json]

PIN.json maps dataset keys (Baryon, Temp, DMD, Miranda, QMC, Magnetic) to pkl files in
sperr_fft_cache/ (same format as PAPER_V3_CACHES.json / PAPER_FINAL_CACHES.json). With a
second pin, key numbers are printed old -> new.

Definitions (match the paper text):
  gain            = PSNR(base+X) - PSNR(base) at the same operating point (dB)
  iso-PSNR CR gain= CR_ours / CR_ref - 1, CR_ref = CR of the reference curve at PSNR(ours),
                    linear interpolation of log(CR) vs PSNR along the reference curve
                    (extrapolated on the nearest segment when ours exceeds the curve; flagged *)
  Rel L2          = ||x - x_hat||_2 / ||x||_2 = drange * 10^(-PSNR/20) / RMS(x)
  FFT reductions  = 1 - err(ours)/err(ref) for the per-slice FFT magnitude / phase L1 errors
  RMSE reduction  = 1 - 10^(-gain/20)
  Table 2 row     = the SZ3 band whose base CR is closest to 500 (SPERR: closest to 500 too)
"""
import json, os, pickle, sys
import numpy as np

CACHE = os.environ.get("ADAMIT_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sperr_fft_cache"))
PIN = sys.argv[1] if len(sys.argv) > 1 else "PAPER_FINAL_CACHES.json"
OLD = sys.argv[2] if len(sys.argv) > 2 else None

# drange / RMS of the ORIGINAL field (for Rel L2), measured once from the raw files
META = {
    "Baryon":   dict(name="NYX baryon density",  drange=1.159e5,  rms=24.03),
    "Temp":     dict(name="NYX temperature",     drange=4.78e6,   rms=1.783e4),
    "DMD":      dict(name="NYX dark-matter dens.", drange=1.378e4, rms=8.521),
    "Miranda":  dict(name="Miranda",             drange=2.001,    rms=1.885),
    "QMC":      dict(name="QMCPack",             drange=33.01,    rms=0.7496),
    "Magnetic": dict(name="Magnetic reconn.",    drange=24.18,    rms=0.5232),
}
ORDER = ["Baryon", "Temp", "DMD", "Miranda", "QMC", "Magnetic"]
SIDES = [("sz3", "sz3", "pipe", "neurlz"), ("sperr", "sperr", "sperr_pipe", "sperr_neurlz")]


def load(pin):
    j = json.load(open(pin if os.path.isabs(pin) else os.path.join(CACHE, pin)))
    out = {}
    for k, f in j.items():
        r = pickle.load(open(os.path.join(CACHE, f), "rb"))
        if isinstance(r, dict) and "results" in r and isinstance(r["results"], dict):
            r = r["results"]
        out[k] = r
    return out


def arr(series, key):
    return np.asarray(series[key], np.float64)


def cr_at_psnr(ref, psnr):
    """CR of the reference curve at a given PSNR (log-CR linear in PSNR). Returns (cr, extrapolated)."""
    cr, ps = arr(ref, "CR"), arr(ref, "PSNR")
    o = np.argsort(ps); cr, ps = cr[o], ps[o]
    lcr = np.log(cr)
    if ps.min() <= psnr <= ps.max():
        return float(np.exp(np.interp(psnr, ps, lcr))), False
    i = (0, 1) if psnr < ps.min() else (len(ps) - 2, len(ps) - 1)
    slope = (lcr[i[1]] - lcr[i[0]]) / (ps[i[1]] - ps[i[0]])
    return float(np.exp(lcr[i[0]] + slope * (psnr - ps[i[0]]))), True


def rel_l2(psnr, m):
    return m["drange"] * 10.0 ** (-psnr / 20.0) / m["rms"]


def analyse(r, key):
    m = META[key]
    res = {}
    for side, b, p, n in SIDES:
        B, P, N = r[b], r[p], r[n]
        if not B["CR"] or not P["CR"]:
            continue
        crb, psb = arr(B, "CR"), arr(B, "PSNR")
        crp, psp = arr(P, "CR"), arr(P, "PSNR")
        crn, psn = arr(N, "CR"), arr(N, "PSNR")
        gain_p = psp - psb; gain_n = psn - psb
        iso_b = [cr_at_psnr(B, v) for v in psp]
        iso_n = [cr_at_psnr(N, v) for v in psp] if len(crn) else [(np.nan, False)] * len(psp)
        iso_b_pct = crp / np.array([c for c, _ in iso_b]) - 1.0
        iso_n_pct = crp / np.array([c for c, _ in iso_n]) - 1.0
        mag_b, mag_p, mag_n = arr(B, "fft_mag"), arr(P, "fft_mag"), arr(N, "fft_mag")
        pha_b, pha_p, pha_n = arr(B, "fft_phase"), arr(P, "fft_phase"), arr(N, "fft_phase")
        i500 = int(np.argmin(np.abs(crb - 500.0)))
        res[side] = dict(
            crb=crb, psb=psb, crp=crp, psp=psp, crn=crn, psn=psn, gain_p=gain_p, gain_n=gain_n,
            iso_b_pct=iso_b_pct, iso_b_ext=[e for _, e in iso_b], iso_n_pct=iso_n_pct,
            mag_red_b=1 - mag_p / mag_b, mag_red_n=1 - mag_p / mag_n, nlz_mag_red=1 - mag_n / mag_b,
            pha_red_b=1 - pha_p / pha_b, pha_red_n=1 - pha_p / pha_n, nlz_pha_red=1 - pha_n / pha_b,
            rmse_red=1 - 10 ** (-gain_p / 20.0), i500=i500,
            rel_b=rel_l2(psb, m), rel_p=rel_l2(psp, m), rel_n=rel_l2(psn, m),
        )
    return res


def fmt(v, f="{:+.2f}"):
    return " ".join(f.format(x) for x in np.atleast_1d(v))


def report(pin):
    data = load(pin)
    print(f"\n{'=' * 100}\nPIN {pin}\n{'=' * 100}")
    A = {k: analyse(data[k], k) for k in ORDER if k in data}
    for k in ORDER:
        if k not in A:
            continue
        print(f"\n--- {META[k]['name']} ---")
        for side in ("sz3", "sperr"):
            if side not in A[k]:
                continue
            a = A[k][side]
            ext = "".join("*" if e else " " for e in a["iso_b_ext"])
            print(f" [{side:5s}] base CR   {fmt(a['crb'], '{:7.1f}')}")
            print(f"         base PSNR {fmt(a['psb'], '{:7.2f}')}   ours PSNR {fmt(a['psp'], '{:7.2f}')}   NeurLZ PSNR {fmt(a['psn'], '{:7.2f}')}")
            print(f"         gain ours {fmt(a['gain_p'])}  (up to {a['gain_p'].max():+.2f}, mean {a['gain_p'].mean():+.2f})"
                  f" | NeurLZ {fmt(a['gain_n'])}  (up to {a['gain_n'].max():+.2f}, mean {a['gain_n'].mean():+.2f})"
                  f" | ours-NeurLZ up to {(a['gain_p'] - a['gain_n']).max():+.2f}, mean {(a['gain_p'] - a['gain_n']).mean():+.2f}")
            print(f"         iso-PSNR CR gain vs base {fmt(100 * a['iso_b_pct'], '{:+.0f}%')} {ext} (up to {100 * a['iso_b_pct'].max():+.0f}%, mean {100 * a['iso_b_pct'].mean():+.0f}%)"
                  f" | vs NeurLZ (up to {100 * np.nanmax(a['iso_n_pct']):+.0f}%, mean {100 * np.nanmean(a['iso_n_pct']):+.0f}%)")
            print(f"         FFT mag reduction vs base {fmt(100 * a['mag_red_b'], '{:+.0f}%')} (up to {100 * a['mag_red_b'].max():+.0f}%, mean {100 * a['mag_red_b'].mean():+.0f}%)"
                  f" | vs NeurLZ (up to {100 * a['mag_red_n'].max():+.0f}%, mean {100 * a['mag_red_n'].mean():+.0f}%)"
                  f" | NeurLZ own vs base (mean {100 * a['nlz_mag_red'].mean():+.0f}%)")
            print(f"         FFT phase reduction vs base (up to {100 * a['pha_red_b'].max():+.0f}%, mean {100 * a['pha_red_b'].mean():+.0f}%)"
                  f" | vs NeurLZ (mean {100 * a['pha_red_n'].mean():+.0f}%) | NeurLZ own (mean {100 * a['nlz_pha_red'].mean():+.0f}%)")
            print(f"         RMSE reduction (from gain) {fmt(100 * a['rmse_red'], '{:+.0f}%')} (up to {100 * a['rmse_red'].max():+.0f}%) vs FFT-mag reduction up to {100 * a['mag_red_b'].max():+.0f}%")
            i = a["i500"]
            print(f"         Table-2 row (base CR {a['crb'][i]:.0f}): ours CR {a['crp'][i]:.0f} | PSNR base {a['psb'][i]:.2f} NeurLZ {a['psn'][i]:.2f} ours {a['psp'][i]:.2f}"
                  f" | gain vs base {a['gain_p'][i]:+.2f} vs NeurLZ {a['psp'][i] - a['psn'][i]:+.2f}"
                  f" | Rel L2 base {a['rel_b'][i]:.3e} NeurLZ {a['rel_n'][i]:.3e} ours {a['rel_p'][i]:.3e} (ours vs base {100 * (1 - a['rel_p'][i] / a['rel_b'][i]):+.1f}%)")
    print("\n--- six-dataset summary ---")
    for side in ("sz3", "sperr"):
        g = [A[k][side]["gain_p"].mean() for k in ORDER if side in A.get(k, {})]
        gu = [A[k][side]["gain_p"].max() for k in ORDER if side in A.get(k, {})]
        gn = [A[k][side]["gain_n"].mean() for k in ORDER if side in A.get(k, {})]
        t2 = [A[k][side]["gain_p"][A[k][side]["i500"]] for k in ORDER if side in A.get(k, {})]
        print(f" [{side:5s}] mean gain per dataset {fmt(g)} -> avg {np.mean(g):+.2f} | up-to per dataset {fmt(gu)} (max {max(gu):+.2f})"
              f" | NeurLZ mean {fmt(gn)} -> avg {np.mean(gn):+.2f} | Table-2 gains {fmt(t2)} avg {np.mean(t2):+.2f}")
    return A


if __name__ == "__main__":
    new = report(PIN)
    if OLD:
        old = report(OLD)
        print(f"\n{'=' * 100}\nOLD -> NEW (per dataset, per side): up-to gain | mean gain | Table-2 gain | iso-PSNR up-to | FFT-mag mean\n{'=' * 100}")
        for k in ORDER:
            for side in ("sz3", "sperr"):
                if side not in old.get(k, {}) or side not in new.get(k, {}):
                    continue
                o, n = old[k][side], new[k][side]
                print(f" {META[k]['name']:24s} [{side:5s}] {o['gain_p'].max():+.2f}->{n['gain_p'].max():+.2f} | {o['gain_p'].mean():+.2f}->{n['gain_p'].mean():+.2f}"
                      f" | {o['gain_p'][o['i500']]:+.2f}->{n['gain_p'][n['i500']]:+.2f} | {100 * o['iso_b_pct'].max():+.0f}%->{100 * n['iso_b_pct'].max():+.0f}%"
                      f" | {100 * o['mag_red_b'].mean():+.0f}%->{100 * n['mag_red_b'].mean():+.0f}%")
