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
| **`base_script/`** | The core library (imported by every experiment). `bg_stage.py` (train/inference), `experiment.py` (config builder + size budgeting), `bg_shard.py` (model-size selection + sharded training), `bg_normalize.py` / `bg_sampling.py` (helpers), `frequency_losses.py`, `config_io.py` (SZ3 I/O), `metrics.py`, `siren_fft_backbone_model.py` (`UNET_Model`), `train.py` (`TrainConfig`), `Patch_data.py` (samplers). |
| `BO_Adaptive/` | Two-phase BO of learning rate / slice direction (`lr_slice_direction_*` notebooks; `lr_*.py` DDP scripts). |
| `Model_parameter_Scaling/` | PSNR-vs-CR sweeps over **model size × error bound** (NYX, Miranda, Magnetic, S3D, Hurricane), with the per-`rel` Phase-1 → Phase-2 pipeline. |
| `frequency_head_loss/` | Ablation of the frequency head and frequency loss. |
| `Normalization/` | Input / residual normalization ablation (z-score vs min-max). |
| `MultiGPU_DDP/` | Multi-GPU data-parallel / DDP training and the sharded-expert (per-z-chunk) scheme. |
| `SPERR/` | SZ3+model vs **SPERR** (and SPERR+model) comparison across datasets. |

The experiment notebooks/scripts add `base_script/` to `sys.path` and import its modules
by bare name (`from bg_stage import ...`). `base_script/` is self-contained — it does not
depend on any other folder.

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
3. Open a notebook (e.g. `Model_parameter_Scaling/Miranda_parameters.ipynb`) and run all
   cells with the Python kernel that has `numpy`/`torch`/`pysz`, or run a DDP script
   (e.g. `torchrun --nproc_per_node=4 BO_Adaptive/lr_NYX.py`).
