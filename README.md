# BO-Adaptive Neural Residual Compression for Scientific Data

Improve a lossy base compressor (**SZ3** or **SPERR**) by training a small neural
network to predict and add back its reconstruction **residual**, raising PSNR at a given
compression ratio. A two-phase **Bayesian optimization (BO)** stage adapts the training
configuration (learning rate, slice direction) *per error bound*, so each operating
point on the rate–distortion curve is trained with its own tuned setup.

```
reconstruction = base_decompress(stream)  +  model(base_recon, aux_fields)
CR = original_bytes / (base_stream_bytes + model_param_bytes)      # aux fields not charged
```

- **Base codec**: SZ3 (error-bounded) or SPERR (wavelet), compresses the target field.
- **Background (BG) model**: a compact 2-D CNN (`UNET_Model`) that refines the base
  reconstruction slice-by-slice, with optional auxiliary fields as extra input channels
  and an optional split-band (low/mid/high) frequency supervision.
- **Two-phase BO**: Phase 1 runs a cheap proxy Optuna/TPE search of `(lr, slice
  direction)` *for each `rel_err`*; Phase 2 trains the full data at that error bound
  using the per-`rel` result.

## Repo layout

| Folder | What it is |
|---|---|
| **`base_script/`** | The core library (imported by every experiment). `bg_stage.py` (train / inference), `experiment.py` (config builder + model-size budgeting), `siren_fft_backbone_model.py` (`UNET_Model`: micro U-Net backbone + low/mid/high heads), `frequency_losses.py` (dual-domain FFT loss), `bg_sampling.py` / `Patch_data.py` (GPU-resident slice samplers), `bg_normalize.py` (Z-score normalization), `config_io.py` (SZ3 I/O, seeding, error-bound clamp), `bg_shard.py` (model-size selection + sharded/DDP helpers), `train.py` (`TrainConfig`). |
| **`SPERR/`** | The paper pipeline: `SPERR_fft.py --task {nyx_b,nyx_t,nyx_d,miranda,mag,qmcpack}` runs SZ3 / SPERR, AdaMit and NeurLZ on one dataset and caches the result in `sperr_fft_cache/`; `--task aux_prep` archives the matched-CR auxiliary streams; `--task neurlz_long` is the NeurLZ cost study. `paper_numbers.py`, `plot_paper_figs.py`, `plot_paper_fft.py` turn pinned caches into the paper's numbers and figures. |
| **`Reproduce/`** | Reproducibility package: `run_all.sh` (whole chain), `collect.py` (pin + figures + numbers), `REPRODUCE.ipynb`, `REPORT.md` (results report), `HANDOFF.md`, `results/` (cache pins), `figures/`, `experiment/` (sibling-protocol, cascade and NeurLZ-cost experiments), `logs/`. |
| `BO_Adaptive/` | Two-phase Bayesian optimization example (Fig. 9): `nyx_miranda*.ipynb`, `bo_combined_plot.ipynb`. |
| `Model_parameter_Scaling/` | Model-size sweep at fixed epochs (Fig. 7): `nyx_miranda_isoepoch.ipynb`, `isoepoch_nyx_rerun.py`. |
| `Normalization/` | Z-score vs Min-Max ablation (Fig. 4): `normalization.ipynb`, `norm_rerun.py`. |
| `frequency_head_loss/` | Frequency head / loss ablation (Fig. 6): `fft_err.ipynb`. |
| `bf_16_vs_32/` | BF16 vs FP32 model storage (Fig. 8): `bf_16.ipynb`, `bf16_rerun.py`. |
| `MultiGPU_DDP/` | Multi-GPU data-parallel / DDP training (Table 4, Fig. 14): `parallel_compute.ipynb`, `data_parallel_for_{NYX,Miranda}.py`. |

Superseded notebooks and figures live in `_archive_*/` (not tracked). The experiment
notebooks/scripts add `base_script/` to `sys.path` and import its modules by bare name
(`from bg_stage import ...`); `base_script/` does not depend on any other folder.

## Prerequisites

Python packages (`pip install -r requirements.txt`): `numpy`, `torch`, `matplotlib`,
`pandas`, `optuna`.

External, set up separately (not pip-installable):
- **SZ3** with its `pysz` Python wrapper — the error-bounded base compressor.
- **SPERR** (`sperr3d` binary) — the wavelet base compressor (used in `SPERR/`).

## ⚠️ Paths are hard-coded

The notebooks and scripts contain **absolute paths** (`/home/sam/...`) for the data
volumes, the SZ3 shared library, the `pysz` wrapper directory, and the SPERR binary.
**Edit these to your environment before running.** The scientific data volumes
(`*.raw`, `*.f32`, `*.d64`, `*.sz`) and checkpoint directories are **not** part of this
repo (see `.gitignore`); point the paths at your local copies.

In particular, each experiment sets:
```python
sys.path.append("/home/sam/Halo_Finder/Final_design/base_script")   # the core library
PYSZ_PATH = "/home/sam/Data_Compression/SZ3/tools/pysz"             # pysz wrapper dir
sz_lib_path = "/home/sam/Data_Compression/SZ3/build/lib64/libSZ3c.so"
```

## Running an experiment

1. Install the Python deps and make SZ3/`pysz` (and SPERR for `SPERR/`) importable/available.
2. Edit the absolute paths at the top of the chosen notebook/script to your data + SZ3/SPERR locations.
3. Run one dataset of the paper pipeline, e.g. `cd SPERR && python SPERR_fft.py --task nyx_b`
   (results are cached in `SPERR/sperr_fft_cache/`), or the whole chain with
   `Reproduce/run_all.sh` followed by `python Reproduce/collect.py`. Ablation notebooks
   (Figs. 4, 6, 7, 8, 9) run cell-by-cell with a kernel that has `numpy`/`torch`/`pysz`;
   multi-GPU runs use `torchrun --nproc_per_node=4 MultiGPU_DDP/data_parallel_for_NYX.py`.
