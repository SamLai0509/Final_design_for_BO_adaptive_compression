# AdaMit: Adaptive, Frequency-Aware Neural Error Mitigation for Scientific Lossy Compression

AdaMit improves an error-bounded lossy compressor (**SZ3** or **SPERR**) by training a small
2-D CNN *online, at compression time*, to predict the compressor's residual. The compressed
output is the base bitstream plus the BF16 model weights; the decoder reconstructs the base
field, adds the predicted residual and clamps it to the error bound.

```
reconstruction = base_decompress(stream) + clamp(model(base_recon, auxiliary_fields))
effective CR   = original_bytes / (base_stream_bytes + model_bytes)
```

Key components (paper Section 3):
- **Matched-CR / enhanced auxiliary fields** — sibling fields enter the model as the decoder holds
  them (decompressed at the CR level closest to the target's); fields decoded earlier in the cascade
  are fed to later ones in their enhanced form.
- **Z-score normalization** of inputs and residuals.
- **Frequency-split heads + dual-domain (spatial + spectral) loss**.
- **Two-phase adaptive training** — Phase 1: 10 TPE trials over (learning rate, slice axis) on a
  half-resolution proxy within 10% of the budget; Phase 2: full-resolution training for the rest.
- **GPU-resident slicing** and BF16 model storage. (This branch is the single-GPU version; the multi-GPU
  data-parallel experiments of the paper live on a separate branch.)

## Repository layout

| Folder | Contents |
|---|---|
| **`base_script/`** | Core library. `bg_stage.py` (training / inference), `experiment.py` (config builder, model-size budgeting), `siren_fft_backbone_model.py` (`UNET_Model`: micro U-Net + low/mid/high heads), `frequency_losses.py` (dual-domain FFT loss), `bg_sampling.py` / `Patch_data.py` (GPU-resident slice samplers), `bg_normalize.py` (Z-score), `config_io.py` (SZ3 I/O, seeding, error-bound clamp), `bg_shard.py` (model-size selection), `train.py` (`TrainConfig`). |
| **`SPERR/`** | The paper pipeline. `SPERR_fft.py --task {nyx_b,nyx_t,nyx_d,miranda,mag,qmcpack}` runs SZ3, SPERR, AdaMit and NeurLZ for one dataset and caches the result in `sperr_fft_cache/`; `--task aux_prep` archives the matched-CR auxiliary streams; `--task neurlz_long` is the NeurLZ cost study. `paper_numbers.py`, `plot_paper_figs.py`, `plot_paper_fft.py` turn pinned caches into the paper's numbers and Figs. 10/11. |
| **`Reproduce/`** | Reproducibility scripts and notes: `run_all.sh` (whole chain), `collect.py` (pin results, regenerate figures/numbers), `make_reproduce_nb.py`, `REPORT.md` (full results report), `HANDOFF.md`, `CLEANUP_PLAN.md`, `experiment/` (sibling-protocol, cascade and NeurLZ-cost experiments), `benchmarks/`. Result pickles, figures and logs are produced locally and are not tracked. |
| `BO_Adaptive/` | Two-phase BO example (Fig. 9): `nyx_miranda.ipynb`, `nyx_miranda_sperr.ipynb`, `bo_combined_plot.ipynb`. |
| `Model_parameter_Scaling/` | Model-size sweep at fixed epochs (Fig. 7): `nyx_miranda_isoepoch.ipynb`, `isoepoch_nyx_rerun.py`. |
| `Normalization/` | Z-score vs Min-Max ablation (Fig. 4): `normalization.ipynb`, `norm_rerun.py`. |
| `frequency_head_loss/` | Frequency head / loss ablation (Fig. 6): `fft_err.ipynb`. |
| `bf_16_vs_32/` | BF16 vs FP32 model storage (Fig. 8): `bf_16.ipynb`, `bf16_rerun.py`. |

Scripts and notebooks add `base_script/` to `sys.path` and import its modules by bare name
(`from bg_stage import ...`); `base_script/` depends on nothing else in the repository.

## Requirements

- **Hardware.** One NVIDIA GPU (the paper used an RTX PRO 6000 Blackwell, 96 GB; a 24 GB card is
  enough for the 512³ fields, Miranda 1024³ needs more) and ≥ 64 GB host RAM (the NeurLZ baseline on
  Miranda peaks around 50 GB). All results are wall-clock budgeted: run on an otherwise idle machine.
- **Software.** Linux, CUDA, Python ≥ 3.10 (tested with 3.13, PyTorch 2.10, NumPy 2.3, Optuna 4.9, MONAI 1.5).
  Python packages: `pip install -r requirements.txt`
  (`numpy torch matplotlib pandas optuna monai jupyter nbconvert`; MONAI provides the `BasicUNet`
  used by the NeurLZ baseline).
- **SZ3 + `pysz`** (error-bounded base compressor) and **SPERR** (`sperr3d` binary, wavelet base
  compressor) — built from source, see Step 3.

## Step-by-step setup

### Step 1 — Clone and create the environment

```bash
git clone <this repository> AdaMit && cd AdaMit
conda create -n adamit python=3.11 -y && conda activate adamit
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the wheel for your CUDA
pip install -r requirements.txt
```

### Step 2 — Download the datasets (all public, SDRBench)

| Dataset | Files the scripts expect | Shape (z, y, x) |
|---|---|---|
| NYX (EXASKY, cosmology) | `baryon_density.f32`, `dark_matter_density.f32`, `temperature.f32`, `velocity_x.f32`, `velocity_y.f32`, `velocity_z.f32` in one directory | 512 × 512 × 512 |
| Miranda (hydrodynamics) | `miranda_1024x1024x1024_float32.raw` | 1024 × 1024 × 1024 |
| Magnetic Reconnection (plasma) | `magnetic_reconnection_512x512x512_float32.raw` | 512 × 512 × 512 |
| QMCPack (einspline coefficients) | `288x115x69x69/einspline_288_115_69_69.pre.f32` | folded to 33120 × 69 × 69 |

All files are raw little-endian float32 volumes in C order. The data are **not** part of the
repository (`*.f32`, `*.raw`, `*.d64`, `*.sz` are gitignored).

### Step 3 — Build the base compressors

```bash
# SZ3 (https://github.com/szcompressor/SZ3) + its pysz wrapper
git clone https://github.com/szcompressor/SZ3 /path/to/SZ3
cmake -S /path/to/SZ3 -B /path/to/SZ3/build -DCMAKE_BUILD_TYPE=Release && cmake --build /path/to/SZ3/build -j
#  -> shared library  /path/to/SZ3/build/lib64/libSZ3c.so   (or lib/ on some distros)
#  -> pysz wrapper    /path/to/SZ3/tools/pysz               (imported as `from pysz import SZ`)

# SPERR (https://github.com/NCAR/SPERR)
git clone https://github.com/NCAR/SPERR /path/to/SPERR
cmake -S /path/to/SPERR -B /path/to/SPERR/build -DCMAKE_BUILD_TYPE=Release && cmake --build /path/to/SPERR/build -j
#  -> command-line tool /path/to/SPERR/build/bin/sperr3d
```

Quick check: `python -c "import sys; sys.path.append('/path/to/SZ3/tools/pysz'); from pysz import SZ; SZ('/path/to/SZ3/build/lib64/libSZ3c.so'); print('pysz ok')"`
and `/path/to/SPERR/build/bin/sperr3d --help`.

### Step 4 — Point the code at your paths

The scripts use absolute paths that you must edit once. The placeholders below (`/path/to/...`)
stand for your own locations.

**`SPERR/SPERR_fft.py`** (top of the file and the dataset blocks):

```python
SPERR_BIN    = "/path/to/SPERR/build/bin/sperr3d"
SZ3_LIB      = "/path/to/SZ3/build/lib64/libSZ3c.so"
PYSZ_PATH    = "/path/to/SZ3/tools/pysz"
SCRIPTS_PATH = "/path/to/AdaMit/base_script"
FFT_CACHE_DIR = "/path/to/AdaMit/SPERR/sperr_fft_cache"      # results + auxiliary streams go here
NYX_DIR   = "/path/to/data/NYX/"                             # directory with the six NYX .f32 files
MIR_FILE  = "/path/to/data/miranda_1024x1024x1024_float32.raw"
MAG_FILE  = "/path/to/data/magnetic_reconnection_512x512x512_float32.raw"
QMC_FILE  = "/path/to/data/SDRBENCH-QMCPack/288x115x69x69/einspline_288_115_69_69.pre.f32"
```

**Ablation notebooks and rerun scripts** (`BO_Adaptive/*.ipynb`, `Model_parameter_Scaling/isoepoch_nyx_rerun.py`,
`Normalization/norm_rerun.py`, `bf_16_vs_32/bf16_rerun.py`, `frequency_head_loss/fft_err.ipynb`) each set the same three things in their first cell / header:

```python
sys.path.append("/path/to/AdaMit/base_script")        # the core library
PYSZ_PATH   = "/path/to/SZ3/tools/pysz"               # pysz wrapper directory
sz_lib_path = "/path/to/SZ3/build/lib64/libSZ3c.so"   # SZ3 shared library
```

plus the data directory (`NYX_DIR` / `nyx_dir` and the Miranda file). **`Reproduce/*.sh`** set
`REPO=/path/to/AdaMit` and `PY=/path/to/python` in their first lines.

### Step 5 — Archive the matched-CR auxiliary streams (once)

```bash
cd SPERR
python SPERR_fft.py --task aux_prep
```

This compresses every NYX field with SZ3 and SPERR at CR levels 100…500 (level 600 is built on
demand) and stores the streams in `sperr_fft_cache/aux_streams/` (~140 MB, CPU only, ~25 min).
Every later run reads the siblings from this archive, so training and inference see exactly the
copies a decoder would hold.

### Step 6 — Run one dataset

```bash
cd SPERR
python -u SPERR_fft.py --task nyx_b        # NYX baryon density; also nyx_t, nyx_d, miranda, mag, qmcpack
```

Per operating point (five rel bounds per dataset, aligned to CR ≈ 100/150/220/320/500) the script
compresses with SZ3, runs Phase 1 + Phase 2 of AdaMit under the wall-clock budget (10 s for 512³
fields, 60 s QMCPack, 80 s Miranda), trains the NeurLZ baseline under the same budget, repeats both
on the SPERR side at the matched CR, and records CR / PSNR / FFT magnitude & phase errors. Results are
cached as `sperr_fft_cache/<Dataset>__<config-hash>.pkl` (a dict with the series `sz3`, `pipe`
(= AdaMit), `neurlz`, `sperr`, `sperr_pipe`, `sperr_neurlz`, each holding lists `CR`, `PSNR`,
`fft_mag`, `fft_phase`). A second invocation with the same configuration hits the cache; set
`SPERR_FORCE_RETRAIN=1` to retrain. Approximate wall time: 15–20 min per NYX field or Magnetic,
45 min QMCPack, 1.5 h Miranda.

Useful switches (environment variables):

| Variable | Effect |
|---|---|
| `SPERR_FORCE_RETRAIN=1` | ignore the cache and retrain |
| `SPERR_ONLY_BANDS=4` | run only the given rel-band indices (e.g. the CR≈500 point) |
| `SPERR_NYX_RELS=1.28e-5` | custom rel bounds for NYX (visualization runs) |
| `SPERR_SAVE_RECONS_DIR=/path/to/out` | export `base/ours/neurlz.{f32,vtk}` + error volumes per point (ParaView) |
| `SPERR_AUX_MODE=cr_matched\|orig`, `SPERR_USE_AUX=0` | sibling protocol / disable siblings |
| `SPERR_AUX_ENHANCED_DIR=/path/to/exports` | cascade: feed already-enhanced siblings from an earlier stage |
| `SPERR_SAMPLE_MODE`, `SPERR_LR_MIN/MAX`, `SPERR_STEP_CALIB`, `SPERR_DET=1` | sampling order, lr window, step-level schedule calibration, deterministic cuDNN |

### Step 7 — Reproduce the full paper chain

```bash
bash Reproduce/run_all.sh          # ~4.5 h: aux_prep -> qmcpack -> nyx_b/t/d -> mag -> miranda -> Fig. 9 BO halves
python Reproduce/collect.py        # pins the six result pickles, writes paper_numbers.txt, Figs. 9-11, REPRODUCE.ipynb
bash Reproduce/run_cascade_6000.sh # enhanced-auxiliary cascade for the NYX fields (Table 2 / Figs. 10-11 NYX rows)
python SPERR/plot_paper_figs.py PAPER_ENHANCED_MIXED_CACHES.json   # PSNR vs CR from a pin file
python SPERR/plot_paper_fft.py  PAPER_ENHANCED_MIXED_CACHES.json   # FFT errors vs CR
```

`collect.py` pins results by the run's start/exit stamps in `Reproduce/logs/run_all.log`, so run the
chain uninterrupted. Outputs land in `Reproduce/results/` (pins, `paper_numbers.txt`) and
`Reproduce/figures/`.

### Step 8 — Ablations and other figures

| Paper figure | Command |
|---|---|
| Fig. 4 normalization | `python Normalization/norm_rerun.py` (temperature) / `NORM_TARGET=baryon_density NORM_RELS=1e-6,1.4e-6,2.4e-6,4e-6,6.9e-6,1e-5,1.4e-5 python Normalization/norm_rerun.py` |
| Fig. 6 frequency head/loss | run `frequency_head_loss/fft_err.ipynb` |
| Fig. 7 model size | `python Model_parameter_Scaling/isoepoch_nyx_rerun.py` |
| Fig. 8 BF16 vs FP32 | `python bf_16_vs_32/bf16_rerun.py` |
| Fig. 9 BO example | `jupyter nbconvert --to notebook --execute BO_Adaptive/nyx_miranda.ipynb` (SZ3/NYX), `nyx_miranda_sperr.ipynb` (SPERR/Miranda), then `bo_combined_plot.ipynb` |
| Auxiliary-quality ladder (Fig. 3) | `bash Reproduce/run_auxq_ablation.sh && python Reproduce/plot_aux_quality.py` |
| NeurLZ cost study (Table 2, NeurLZ block) | `bash Reproduce/experiment/neurlz_long/run_100ep.sh` and `run_100ep_sperr.sh`, then `python Reproduce/experiment/neurlz_long/plot_neurlz_long.py` |

### Step 9 — Use AdaMit on your own field

```python
import sys; sys.path.append("/path/to/AdaMit/base_script")
import numpy as np, torch
from experiment import build_bg_only_cfg
from bg_stage import train_bg_only, run_bg_inference

gt  = np.fromfile("/path/to/field.f32", np.float32).reshape(D, H, W)   # original
lq  = ...                                                              # base-compressor reconstruction of gt
aux = [...]                                                            # optional sibling fields, decompressed the same way
cfg = build_bg_only_cfg(X_target=gt, Xps=[lq] + aux, max_train_time=10.0, bg_h=24, bg_batch=4,
                        bg_patch_size=W, lr=5e-3, epochs=100000, steps_per_epoch=D // 4)
cfg.bg_split_bands, cfg.bg_split_mode, cfg.bg_sample_mode, cfg.bg_full_slice = True, "three", "shuffled", True
cfg.bg_sched_time_calibrate = True
model, hist = train_bg_only([gt] + aux, [lq] + aux, torch.device("cuda:0"), cfg)
x_hat = run_bg_inference(model, [gt] + aux, [lq] + aux, cfg, rel_err)  # clamped to the base error bound
```

`SPERR_fft.py::bench_field_fft` shows the complete recipe including the Phase-1 search
(`_phase1_best_fast`) and the model-size budgeting (`bg_shard.pick_bg_h_under_budget`).

## Notes

- The pipeline is deterministic up to cuDNN autotuning; Phase-1 picks can differ between runs by a
  few tenths of a dB on 512² slices (QMCPack runs are bit-reproducible). `SPERR_DET=1` forces
  deterministic kernels at some speed cost.
- Miranda's NeurLZ pass needs ~50 GB of host RAM; never run two training jobs on one machine, the
  wall-clock budgets would no longer be comparable.
- `Reproduce/REPORT.md` documents every number in the paper and how it was produced.
