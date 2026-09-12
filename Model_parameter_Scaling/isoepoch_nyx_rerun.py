"""Fig. 6 (iso-epoch model-size sweep), NYX panels re-run under the paper-final sibling protocol.

Reuses the notebook's own engine (cells 1-2 of nyx_miranda_isoepoch.ipynb are exec'ed, so the sweep is
byte-identical) and changes exactly two things:
  * siblings: instead of the lossless originals, each sibling is the SZ3-archived copy at the CR level
    closest to the target's CR (SPERR/sperr_fft_cache/aux_streams/, the same streams the paper run uses);
  * slice sampling: 'shuffled' (paper protocol) instead of 'sequential'.
Everything else (10 fixed epochs, lr 1e-3, 3k-30k parameter ladder, rel bands) is unchanged.
    python isoepoch_nyx_rerun.py   -> isoepoch_nyx_crmatched.json + psnr_vs_cr_isoepoch_nyx_baryon_temp.{pdf,png}
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
NB   = os.path.join(HERE, "nyx_miranda_isoepoch.ipynb")
AUX_STREAM_DIR = "/home/sam/Halo_Finder/Final_design/SPERR/sperr_fft_cache/aux_streams"
LEVELS = (100, 200, 300, 400, 500, 600)
NYX_SHAPE = (512, 512, 512)
nyx_dir = "/home/sam/Halo_Finder/halo_finder_v1/SDRBENCH-EXASKY-NYX-512x512x512/origin/"

cells = ["".join(c["source"]) for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
ns = {}
def cell(marker):
    return next(c for c in cells if marker in c)
exec(cell("from bg_stage import").replace('torch.device("cuda:1"', 'torch.device("cuda:0"'), ns)   # imports; paper GPU
exec(cell("def run_sweep("), ns)                                                                 # engine (build_cfg, run_sweep, ...)
_build_cfg = ns["build_cfg"]
def build_cfg(*a, **k):
    cfg = _build_cfg(*a, **k)
    cfg.bg_sample_mode = os.environ.get("BO_SAMPLE_MODE", "shuffled")   # paper protocol
    return cfg
ns["build_cfg"] = build_cfg
sz_engine, run_sweep, build_sweep_plan = ns["sz_engine"], ns["run_sweep"], ns["build_sweep_plan"]

_sib_cache = {}
def sibling(field, cr):
    """The decoder's copy of `field`: SZ3 stream at the CR level nearest (log) to the target's CR."""
    level = min(LEVELS, key=lambda L: abs(np.log(L) - np.log(max(cr, 1e-9))))
    key = (field, level)
    if key not in _sib_cache:
        if any(k[1] != level for k in _sib_cache):
            _sib_cache.clear()                                  # keep one level resident (~2.5 GB)
        stem = f"{AUX_STREAM_DIR}/sz3_{field}_f32_cr{level}"
        assert os.path.isfile(stem + ".bin"), f"missing {stem}.bin (run `python SPERR/SPERR_fft.py --task aux_prep`)"
        _sib_cache[key] = np.ascontiguousarray(sz_engine.decompress(np.fromfile(stem + ".bin", np.uint8), NYX_SHAPE, np.float32), np.float32)
        meta = json.load(open(stem + ".json"))
        print(f"  [aux] {field:20s} target CR {cr:6.1f} -> level {level} (SZ3 stream CR {meta['cr']:.1f})", flush=True)
    return _sib_cache[key]


def run_field(target, aux_fields, rel_errs, param_targets, fixed_epochs=10):
    gt = np.fromfile(f"{nyx_dir}{target}.f32", dtype=np.float32).reshape(NYX_SHAPE)
    def compress_at(vol, rel):
        b, cr = sz_engine.compress(vol, 1, 0, float(rel), 0)
        x_lq = np.asarray(sz_engine.decompress(b, vol.shape, np.float32), np.float32)
        return [x_lq] + [sibling(f, float(cr)) for f in aux_fields], int(len(b)), float(cr)
    plan = build_sweep_plan(param_targets, gt.shape, n_fields=1 + len(aux_fields))
    print(f"\n### {target}: plan {plan}, {len(rel_errs) * len(plan)} trainings x {fixed_epochs} epochs", flush=True)
    res, base = run_sweep(gt, [gt], compress_at, gt.nbytes, gt.shape, rel_errs, plan, None,
                          label=target, fixed_epochs=fixed_epochs)
    del gt; _sib_cache.clear()
    return res, base


PARAM_TARGETS = [3000, 5000, 9500, 17000, 30000]
FIELDS = {
    "baryon_density": (["dark_matter_density", "temperature", "velocity_x", "velocity_y", "velocity_z"],
                       [1.3827e-06, 2.3587e-06, 4.0235e-06, 6.8636e-06, 1.1708e-05]),
    "temperature":    (["baryon_density", "dark_matter_density", "velocity_x", "velocity_y", "velocity_z"],
                       [1.4458e-04, 2.2308e-04, 3.4420e-04, 5.3109e-04, 8.1945e-04]),
}
out = {}
t0 = time.time()
for f, (aux, rels) in FIELDS.items():
    res, base = run_field(f, aux, rels, PARAM_TARGETS)
    out[f] = dict(results=res, baselines=base, aux=aux, rel_errs=rels, param_targets=PARAM_TARGETS, fixed_epochs=10,
                  protocol="CR-matched decompressed siblings (SZ3 streams), shuffled sampling, lr 1e-3")
    json.dump(out, open(os.path.join(HERE, "isoepoch_nyx_crmatched.json"), "w"), indent=1)
print(f"\nboth sweeps done in {(time.time() - t0) / 60:.1f} min", flush=True)

# ── the combined 1x2 figure (NYX baryon | NYX temperature), notebook style ──────────────────
import matplotlib.pyplot as plt
_pc = cell("def plot_rd(")
plot_src = _pc[:_pc.index("fig, axes = plt.subplots")]      # plot_rd + helpers only
exec(plot_src, ns)
plot_rd = ns["plot_rd"]
fig, axes = plt.subplots(1, 2, figsize=(22, 10))
plot_rd(axes[0], out["baryon_density"]["results"], out["baryon_density"]["baselines"], "NYX baryon density", show_legend=False)
plot_rd(axes[1], out["temperature"]["results"], out["temperature"]["baselines"], "NYX temperature", show_legend=False)
for ax in axes:
    ax.set_title(ax.get_title(), fontsize=35, fontweight="bold", y=1.02)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=6, fontsize=34, frameon=False,
           handletextpad=0.3, columnspacing=0.8, handlelength=1.2, markerscale=0.7)
fig.supxlabel("Effective Compression Ratio", fontsize=40, fontweight="bold", y=0.02)
fig.supylabel("PSNR (dB)", fontsize=40, fontweight="bold", x=0.02)
plt.subplots_adjust(left=0.09, right=0.99, bottom=0.14, top=0.86, wspace=0.22)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"psnr_vs_cr_isoepoch_nyx_baryon_temp.{ext}"), dpi=110, bbox_inches="tight")
print("Saved: psnr_vs_cr_isoepoch_nyx_baryon_temp.pdf / .png", flush=True)
