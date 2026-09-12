"""Figure 12: power-spectrum error of NYX baryon density, corrected protocol.

Left panel  eps(k) = |P_hat(k) - P(k)| / P(k) per spherical k-shell, for the
            base codec, the NeurLZ baseline and ours, at effective CR ~ 300.
Right panel best max_k eps(k) against pure training wall time, i.e. how long
            each method needs to meet the Nyx 1 % requirement.

What changed from the draft: the five sibling channels are the ARCHIVED
decompressed volumes at the CR level nearest the target's, not the originals.
A decoder only ever has the archived siblings, so training on originals is not
reproducible at decode time; on this field it was worth about 5 dB, and the
spectrum error moves with it. Both methods get the same siblings.

  python power_spectrum_fig12.py --cr 300 --budget 3 --neurlz-epochs 100
"""
from __future__ import annotations

import os
# Must precede the first cuBLAS call, as in neurlz_ep100.py.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import io
import json
import contextlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                  # repository root
sys.path.append(str(ROOT / "base_script"))
from local_paths import P  # noqa: E402
sys.path.append(P("ADAMIT_PYSZ"))
sys.path.append(str(HERE.parent / "multi_gpu"))          # aux_siblings

from pysz import SZ                                            # noqa: E402
from experiment import build_bg_only_cfg                       # noqa: E402
from bg_stage import train_bg_only, run_bg_inference, unwrap_bg_model  # noqa: E402
from bg_shard import pick_bg_h_under_budget                    # noqa: E402
from aux_siblings import load_siblings                         # noqa: E402

SZ3_LIB = P("ADAMIT_SZ3_LIB")
NYX_DIR = Path(P("ADAMIT_NYX_DIR"))
SHAPE = (512, 512, 512)
TARGET = "baryon_density"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def psnr_of(a, b):
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    r = float(a.max() - a.min())
    return float("inf") if m <= 0 else 10.0 * np.log10(r * r / m)


def rel_for_cr(sz, gt, target_cr, iters=22):
    lo, hi, best, bd = 1e-9, 1e-1, None, float("inf")
    orig = int(np.prod(SHAPE)) * 4
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        b, _ = sz.compress(gt, 1, 0, mid, 0)
        cr = orig / len(b)
        if abs(cr - target_cr) < bd:
            best, bd = mid, abs(cr - target_cr)
        lo, hi = (mid, hi) if cr < target_cr else (lo, mid)
    return best


# ── P(k) ────────────────────────────────────────────────────────────────────
class Spectrum:
    """Spherically binned |F|^2 on integer k-shells, k in fundamental modes.

    The volume is used as it is rather than as an overdensity, which is what
    the draft figure did. For k > 0 the two differ only through the per-field
    mean, and keeping the field raw avoids giving each reconstruction its own
    normalization.
    """

    def __init__(self, gt, device, kmax=9):
        self.device = device
        n = SHAPE[0]
        fx = torch.fft.fftfreq(n, d=1.0 / n).to(device)
        fz = torch.arange(n // 2 + 1, dtype=torch.float32, device=device)
        r = torch.sqrt(fx[:, None, None] ** 2 + fx[None, :, None] ** 2
                       + fz[None, None, :] ** 2)
        self.k = np.arange(1, kmax + 1)
        self.masks = [((r >= kk - 0.5) & (r < kk + 0.5)) for kk in self.k]
        self.P_gt = self._P(gt)

    def _P(self, vol):
        t = torch.from_numpy(np.ascontiguousarray(vol, np.float32)).to(self.device)
        F = torch.fft.rfftn(t.double())
        p2 = (F.real ** 2 + F.imag ** 2)
        out = np.array([float(p2[m].sum()) for m in self.masks])
        del F, p2, t
        torch.cuda.empty_cache()
        return out

    def eps(self, vol):
        """Relative P(k) error per shell, as a fraction (not %)."""
        return np.abs(self._P(vol) / self.P_gt - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cr", type=float, default=300.0)
    ap.add_argument("--budget", type=float, default=3.0, help="ours: seconds")
    ap.add_argument("--neurlz-epochs", dest="nl_ep", type=int, default=100)
    ap.add_argument("--aux-mode", dest="aux_mode", default="cr_matched",
                    choices=["cr_matched", "enhanced", "originals"])
    ap.add_argument("--aux-enhanced-dir", dest="aux_enh", default="")
    ap.add_argument("--neurlz-aux-mode", dest="nl_aux_mode", default="",
                    help="sibling mode for the NeurLZ baseline; defaults to "
                         "--aux-mode. Set it only to state plainly that the "
                         "two methods were given different inputs -- e.g. "
                         "ours=enhanced against neurlz=cr_matched, where the "
                         "cascade is claimed as part of our system.")
    ap.add_argument("--kmax", type=int, default=9)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--batch", type=int, default=4,
                    help="ours single-GPU batch; steps/epoch scales as 512/batch "
                         "so per-epoch slice coverage stays fixed")
    ap.add_argument("--out", default=str(HERE / "figures" / "fig12"))
    ap.add_argument("--skip-neurlz", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    gt = np.fromfile(NYX_DIR / f"{TARGET}.f32", np.float32).reshape(SHAPE)
    sz = SZ(SZ3_LIB)
    rel = rel_for_cr(sz, gt, args.cr)
    bits, _ = sz.compress(gt, 1, 0, rel, 0)
    x_lq = sz.decompress(bits, SHAPE, np.float32)
    cr_base = int(np.prod(SHAPE)) * 4 / len(bits)
    print(f"[base] rel={rel:.4e} CR_base={cr_base:.1f} "
          f"PSNR={psnr_of(gt, x_lq):.2f} dB", flush=True)

    aux, aux_names = load_siblings(TARGET, cr_base, SHAPE, sz=sz, codec="sz3",
                                   mode=args.aux_mode,
                                   enhanced_dir=(args.aux_enh or None))
    Xs, Xps = [gt] + aux, [x_lq] + aux

    # NeurLZ shares our siblings unless told otherwise. Keeping them identical
    # is the paper protocol; the override exists so a cascade run can be
    # reported honestly as ours=enhanced vs neurlz=cr_matched.
    nl_mode = args.nl_aux_mode or args.aux_mode
    if nl_mode == args.aux_mode:
        aux_nl = aux
    else:
        aux_nl, _ = load_siblings(TARGET, cr_base, SHAPE, sz=sz, codec="sz3",
                                  mode=nl_mode,
                                  enhanced_dir=(args.aux_enh or None))
        print(f"[aux] NeurLZ uses mode={nl_mode} (ours={args.aux_mode})", flush=True)

    spec = Spectrum(gt, device, kmax=args.kmax)
    eps_base = spec.eps(x_lq)
    print("[base] eps(k) % = " + " ".join(f"{e*100:.3f}" for e in eps_base), flush=True)

    res = dict(cr_target=args.cr, cr_base=cr_base, rel=rel,
               aux_mode=args.aux_mode, neurlz_aux_mode=nl_mode,
               aux_names=aux_names,
               k=spec.k.tolist(), eps_base=(eps_base * 100).tolist(),
               base_psnr=psnr_of(gt, x_lq))

    # ── ours ────────────────────────────────────────────────────────────────
    bg_h, n_params = pick_bg_h_under_budget(30000, shape=SHAPE, n_fields=len(Xs),
                                            bg_arch="spatial",
                                            h_candidates=tuple(range(3, 128)))
    set_seed(args.seed)
    cfg = build_bg_only_cfg(
        X_target=gt, Xps=Xps, max_train_time=args.budget, bg_h=int(bg_h), roi_h=4,
        epochs=200, steps_per_epoch=512 // int(args.batch), bg_patch_size=512,
        bg_batch=int(args.batch), lr=8e-3,
        bg_freq_weight=0.5, bg_fft_phase_weight=0.5, bg_freq_warmup_epochs=1,
        bg_field_norm="zscore")
    cfg.bg_arch = "spatial"; cfg.bg_split_mode = "three"; cfg.bg_split_bands = True
    cfg.bg_split_sigma = 0.12; cfg.bg_sigma_low = 0.08; cfg.bg_sigma_mid = 0.18
    cfg.bg_low_weight = 0.2; cfg.bg_mid_weight = 0.5; cfg.bg_high_weight = 1.0
    cfg.bg_cr_rel_err = rel; cfg.rel_err = rel
    cfg.bg_gpu_sampling = True; cfg.seed = args.seed
    cfg.bg_sample_mode = "sequential"
    cfg.amp = True; cfg.amp_dtype = "bf16"

    trace = []

    def ev(m):
        rec = run_bg_inference(unwrap_bg_model(m), Xs, Xps, cfg, rel)
        e = spec.eps(rec)
        trace.append(dict(max_eps=float(e.max() * 100),
                          eps=(e * 100).tolist(), psnr=psnr_of(gt, rec)))
        print(f"    [ours] max eps {e.max()*100:.3f}%  PSNR {trace[-1]['psnr']:.2f}",
              flush=True)
        return trace[-1]["psnr"], 0.0

    model, hist = train_bg_only(Xs=Xs, Xps=Xps, device=device, cfg=cfg, evaluator=ev)
    rec = run_bg_inference(unwrap_bg_model(model), Xs, Xps, cfg, rel)
    eps_ours = spec.eps(rec)
    res["ours"] = dict(params=int(n_params), train_s=float(hist["time"][-1]),
                       psnr=psnr_of(gt, rec),
                       eps=(eps_ours * 100).tolist(),
                       time=[float(t) for t in hist["time"]],
                       max_eps_trace=[t["max_eps"] for t in trace],
                       eps_trace=[t["eps"] for t in trace],
                       psnr_trace=[t["psnr"] for t in trace])
    np.ascontiguousarray(rec, np.float32).tofile(out / "ours.f32")
    print(f"[ours] {hist['time'][-1]:.2f}s PSNR {res['ours']['psnr']:.2f} "
          f"max eps {eps_ours.max()*100:.3f}%", flush=True)
    del model, rec
    torch.cuda.empty_cache()

    # ── NeurLZ, same siblings ───────────────────────────────────────────────
    if not args.skip_neurlz:
        from monai.networks.nets import BasicUNet
        D, H, W = SHAPE
        # NeurLZ's own recipe, verbatim from neurlz_ep100.py / SPERR.py's
        # run_neurlz: features (4,)*6, batch 10, Adam 1e-2, and the periodic
        # CosineAnnealingLR(T_max=1500) that the reference uses. An earlier
        # version of this script passed MONAI's DEFAULT features, which is a
        # 1.98M-parameter network -- 66x ours and 580x NeurLZ's own 3.4k. It
        # converged in 4 epochs instead of ~100 and made the baseline look
        # both faster and more accurate than it is.
        BATCH, LR, FEAT, T_MAX = 10, 1e-2, (4,) * 6, 1500

        def _mm(x, eps=1e-8):
            lo, hi = float(np.min(x)), float(np.max(x))
            return ((np.asarray(x, np.float32) - lo) / (hi - lo + eps)).astype(np.float32), (lo, hi)

        fields = [x_lq] + aux_nl
        lq_n = np.stack([_mm(f)[0] for f in fields], axis=1)
        err_n, (e_lo, e_hi) = _mm(gt - x_lq)
        Xlq = torch.from_numpy(lq_n); Yerr = torch.from_numpy(err_n[:, None])

        set_seed(args.seed)
        with contextlib.redirect_stdout(io.StringIO()):
            nl = BasicUNet(spatial_dims=2, features=FEAT, act="gelu",
                           in_channels=len(fields), out_channels=1).to(device)
        opt = torch.optim.Adam(nl.parameters(), lr=LR)
        spe = (D + BATCH - 1) // BATCH
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)
        mse = torch.nn.MSELoss()
        nl_params = sum(p.numel() for p in nl.parameters() if p.requires_grad)

        def enhanced():
            nl.eval(); o = x_lq.copy()
            with torch.no_grad():
                for st in range(0, D, BATCH):
                    bi = list(range(st, min(st + BATCH, D)))
                    pr = nl(Xlq[bi].to(device)).cpu().numpy()[:, 0, :H, :W]
                    o[bi] = x_lq[bi] + (pr * (e_hi - e_lo + 1e-8) + e_lo)
            nl.train(); return o

        nl_t, nl_tr, nl_eps, nl_ps, elapsed = [], [], [], [], 0.0
        nl_best_psnr, nl_best_state = -1.0, None
        idx = np.arange(D)
        for ep in range(args.nl_ep):
            t0 = time.perf_counter()
            np.random.shuffle(idx)
            for st in range(0, D, BATCH):
                bi = idx[st:st + BATCH]
                opt.zero_grad(set_to_none=True)
                loss = mse(nl(Xlq[bi].to(device)), Yerr[bi].to(device))
                loss.backward(); opt.step(); sched.step()
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0
            _rec = enhanced()
            e = spec.eps(_rec)
            _p = psnr_of(gt, _rec)
            if _p > nl_best_psnr:
                nl_best_psnr = _p
                nl_best_state = {k: v.detach().cpu().clone()
                                 for k, v in nl.state_dict().items()}
            nl_t.append(elapsed); nl_tr.append(float(e.max() * 100))
            nl_eps.append((e * 100).tolist()); nl_ps.append(_p)
            if ep % 10 == 0 or ep == args.nl_ep - 1:
                print(f"    [neurlz] ep{ep+1} {elapsed:.1f}s max eps {e.max()*100:.3f}%",
                      flush=True)
        if nl_best_state is not None:
            nl.load_state_dict({k: v.to(device) for k, v in nl_best_state.items()})
        rec_nl = enhanced()
        eps_nl = spec.eps(rec_nl)
        res["neurlz"] = dict(params=int(nl_params), epochs=args.nl_ep,
                             train_s=float(elapsed), psnr=psnr_of(gt, rec_nl),
                             eps=(eps_nl * 100).tolist(),
                             time=nl_t, max_eps_trace=nl_tr,
                             eps_trace=nl_eps, psnr_trace=nl_ps)
        np.ascontiguousarray(rec_nl, np.float32).tofile(out / "neurlz.f32")
        print(f"[neurlz] {elapsed:.1f}s PSNR {res['neurlz']['psnr']:.2f} "
              f"max eps {eps_nl.max()*100:.3f}%", flush=True)

    (out / "fig12_data.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {out/'fig12_data.json'}", flush=True)


if __name__ == "__main__":
    main()
