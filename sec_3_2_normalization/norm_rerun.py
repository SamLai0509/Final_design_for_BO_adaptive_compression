"""Normalization ablation figure (norm_cr_time_temperature.pdf) re-run under the paper-final sibling protocol.

Both curves are the NeurLZ-style BasicUNet (features (4,)*6, lr 1e-2, batch 10, 100 epochs, pure MSE on the
normalized SZ3 residual), differing ONLY by normalization (z-score vs min-max) -- exactly the notebook's cells.
The single change: the five sibling channels are the decoder's copies (SZ3-archived at the CR level closest to
the target's CR, from sec_4_evaluation/sperr_fft_cache/aux_streams/) instead of the lossless originals.
    python norm_rerun.py  -> norm_cr_time_temperature.{pdf,png} + norm_rerun_results.json
"""
import os, sys, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
NB = os.path.join(HERE, "normalization.ipynb")
sys.path.insert(0, os.path.join(HERE, "..", "base_script"))
from local_paths import P
AUX_STREAM_DIR = os.path.join(P("ADAMIT_CACHE_DIR"), "aux_streams")
LEVELS = (100, 200, 300, 400, 500, 600)
cells = ["".join(c["source"]) for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
def cell(marker): return next(c for c in cells if marker in c)
ns = {"__name__": "__main__"}
exec(cell("def _global_diag("), ns)               # imports, device, seed
_TARGET = os.environ.get("NORM_TARGET", "temperature")            # e.g. baryon_density
_RELS   = os.environ.get("NORM_RELS")                               # comma list overriding REL_SETTINGS
_c = cell("def build_Xps_for_rel(").replace('TARGET_STEM = "temperature"', f'TARGET_STEM = "{_TARGET}"')
if _RELS:
    _c = _c.replace(next(l for l in _c.splitlines() if l.startswith("REL_SETTINGS =")),
                    "REL_SETTINGS = " + repr([(f"r{i}", float(r)) for i, r in enumerate(_RELS.split(","))]))
exec(_c, ns)          # paths, Xs/Xps (originals), sz engine, gt_target, aux_fields
exec(cell("def run_basicunet_norm("), ns)         # BasicUNet ablation runner (BU_* settings)
exec(cell("def train_bg_variant("), ns)           # helpers used by the plot cell (psnr_np, get_run_psnr, ...)
sz, gt_target, data_shape, aux_paths = ns["sz"], ns["gt_target"], ns["data_shape"], ns["aux_paths"]

# ── the decoder's siblings: SZ3 streams at the CR level nearest (log) to the target's CR ──
_cache = {}
def sibling(a_path, cr, tol=0.03):
    level = min(LEVELS, key=lambda L: abs(np.log(L) - np.log(max(cr, 1e-9))))
    key = (a_path, level)
    if key in _cache:
        return _cache[key]
    if any(k[1] != level for k in _cache):
        _cache.clear()
    stem = f"{AUX_STREAM_DIR}/sz3_{os.path.basename(a_path).replace('.', '_')}_cr{level}"
    if not os.path.isfile(stem + ".bin"):                       # level not archived yet (600): bisect and store
        a = np.fromfile(a_path, dtype=np.float32).reshape(data_shape); nbytes = a.nbytes
        lo, hi, best = -8.0, -1.0, None
        for _ in range(12):
            mid = 0.5 * (lo + hi); rel = 10.0 ** mid
            bb, _ = sz.compress(a, 1, 0, float(rel), 0); cr_b = nbytes / len(bb); d = abs(np.log(cr_b / level))
            if best is None or d < best[0]: best = (d, cr_b, rel, np.asarray(bb, np.uint8).copy())
            if d < tol: break
            if cr_b > level: hi = mid
            else: lo = mid
        best[3].tofile(stem + ".bin")
        json.dump({"cr": float(best[1]), "knob": float(best[2]), "level": int(level), "compressor": "sz3", "file": a_path,
                   "nbytes": int(best[3].size)}, open(stem + ".json", "w"))
    dec = np.ascontiguousarray(sz.decompress(np.fromfile(stem + ".bin", np.uint8), data_shape, np.float32), np.float32)
    _cache[key] = dec
    print(f"  [aux] {os.path.basename(a_path):24s} target CR {cr:6.1f} -> level {level} (stream CR {json.load(open(stem + '.json'))['cr']:.1f})", flush=True)
    return dec

def build_Xps_for_rel(rel_err):
    b, cr = sz.compress(gt_target, 1, 0, float(rel_err), 0)
    x_lq = sz.decompress(b, gt_target.shape, np.float32)
    return [x_lq] + [sibling(p, float(cr)) for p in aux_paths], float(cr), b
ns["build_Xps_for_rel"] = build_Xps_for_rel                       # every runner/plot cell resolves it from ns

# ── the ablation sweep (notebook cell 5) ──
ns["results_compare"] = []
t0 = time.time()
for tag, rel in ns["REL_SETTINGS"]:
    for norm in ("zscore", "minmax"):
        print(f"\n==== [norm] {norm} | {tag} rel={rel} ====", flush=True)
        ns["results_compare"].append(ns["run_basicunet_norm"](rel, tag, norm))
print(f"\nsweep done in {(time.time() - t0) / 60:.1f} min", flush=True)
json.dump([{k: v for k, v in r.items() if k not in ("model", "x_hat")} for r in ns["results_compare"]],
          open(f"norm_rerun_results_{_TARGET}.json" if _TARGET != "temperature" else "norm_rerun_results.json", "w"), indent=1, default=float)

# ── the figure (notebook cell 6, verbatim; PDF + PNG) ──
src = cell('save_filename = f"norm_cr_time_')
src = src.replace("plt.show()", 'plt.savefig(save_filename.replace(".pdf", ".png"), dpi=110, bbox_inches="tight")')
exec(src, ns)
print("done", flush=True)
