"""fp32-vs-BF16 figure re-run under the paper-final sibling protocol (NYX panel) + Miranda panel.

Reuses the notebook's own code (cells 1-2 of bf_16.ipynb are exec'ed: run_ab / plot_ab) and changes:
  * NYX siblings: the SZ3-archived copies at the CR level closest to the target's CR (paper protocol),
    not the lossless originals;
  * slice sampling: 'shuffled' (paper protocol; the notebook left bg_stage's default).
Miranda has no siblings and is re-run only so both panels share the sampling setting.
    python bf16_rerun.py   -> bf16_vs_fp32_results.json + bf16_vs_fp32_1x2.{pdf,png}
"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "bf_16.ipynb")
AUX_STREAM_DIR = "/home/sam/Halo_Finder/Final_design/SPERR/sperr_fft_cache/aux_streams"
LEVELS = (100, 200, 300, 400, 500, 600)
NYX_DIR = "/home/sam/Halo_Finder/halo_finder_v1/SDRBENCH-EXASKY-NYX-512x512x512/origin/"
NYX_SHAPE = (512, 512, 512)

cells = ["".join(c["source"]) for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
ns = {}
def cell(marker):
    return next(c for c in cells if marker in c)
exec(cell("def bg_h_for_params("), ns)   # setup (device cuda:0 = paper GPU)
exec(cell("def run_ab("), ns)            # run_ab / plot_ab
_bb = ns["build_bg_only_cfg"]
def build_bg_only_cfg(*a, **k):
    cfg = _bb(*a, **k)
    cfg.bg_sample_mode = os.environ.get("BO_SAMPLE_MODE", "shuffled")   # paper protocol
    return cfg
ns["build_bg_only_cfg"] = build_bg_only_cfg
sz_engine, run_ab = ns["sz_engine"], ns["run_ab"]

# ── NYX baryon density @ rel 5e-6, siblings at the matched CR level ──────────────────────
tgt = np.fromfile(NYX_DIR + "baryon_density.f32", dtype=np.float32).reshape(NYX_SHAPE)
b, cr = sz_engine.compress(tgt, 1, 0, 5e-6, 0); cr = float(cr); del b
level = min(LEVELS, key=lambda L: abs(np.log(L) - np.log(cr)))
print(f"[NYX] rel 5e-6 -> SZ3 CR {cr:.1f} -> sibling level {level}", flush=True)
sibs = []
for f in ("dark_matter_density", "temperature", "velocity_x", "velocity_y", "velocity_z"):
    stem = f"{AUX_STREAM_DIR}/sz3_{f}_f32_cr{level}"
    sibs.append(np.ascontiguousarray(sz_engine.decompress(np.fromfile(stem + ".bin", np.uint8), NYX_SHAPE, np.float32), np.float32))
    print(f"  [aux] {f:20s} SZ3 stream CR {json.load(open(stem + '.json'))['cr']:.1f}", flush=True)
r_nyx = run_ab("NYX", tgt, sibs, rel=5e-6, params=30000, time_budget=10.5)
del tgt, sibs
import torch; torch.cuda.empty_cache()

# ── Miranda 1024^3 @ rel 6.9948e-3, 240k params, 80 s (as in the notebook) ───────────────
_v = np.fromfile("/home/sam/Halo_Finder/halo_finder_v1/miranda_1024x1024x1024_float32.raw", dtype=np.float32).reshape((1024, 1024, 1024))
r_mir = run_ab("Miranda", _v, [], rel=6.9948e-03, params=240000, time_budget=80.0)
del _v; torch.cuda.empty_cache()

json.dump({"nyx": r_nyx, "miranda": r_mir, "protocol": "NYX siblings = CR-matched SZ3 streams (level %d); shuffled sampling" % level},
          open(os.path.join(HERE, "bf16_vs_fp32_results.json"), "w"), indent=1, default=float)

# ── the notebook's v2 combined figure, verbatim ──────────────────────────────────────────
ns["r_nyx"], ns["r_mir"] = r_nyx, r_mir
os.chdir(HERE)
exec(cell("def plot_ab_v2(").replace("plt.show()", "fig.savefig('bf16_vs_fp32_1x2.png', dpi=120, bbox_inches='tight')"), ns)
print("Saved: bf16_vs_fp32_1x2.pdf / .png", flush=True)
