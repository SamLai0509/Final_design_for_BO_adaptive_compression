# AdaMit: Adaptive, Frequency-Aware Neural Error Mitigation for Scientific Lossy Compression

AdaMit improves an error-bounded lossy compressor (**SZ3** or **SPERR**) by training a small
2-D CNN *online, at compression time*, to predict the compressor's residual. The compressed
output is the base bitstream plus the BF16 model weights; the decoder reconstructs the base
field, adds the predicted residual and clamps it to the error bound.

```
reconstruction = base_decompress(stream) + clamp(model(base_recon, auxiliary_fields))
effective CR   = original_bytes / (base_stream_bytes + model_bytes)
```

This branch (`multi-gpu`) is the `single-gpu` code plus the paper's multi-GPU data-parallel
experiments (§4.4, Table 4, Fig. 13) and the power-spectrum compliance study (Fig. 12, Table 3).
Everything from `single-gpu` runs unchanged here; the additions live under `sec_4_evaluation/`.

## Repository layout (one folder per paper section)

| Folder | Paper section | Contents |
|---|---|---|
| **`base_script/`** | §3 (method) | Core library. `bg_stage.py` (training / inference), `experiment.py` (config builder, model-size budgeting), `siren_fft_backbone_model.py` (`UNET_Model`: micro U-Net + low/mid/high heads), `frequency_losses.py` (dual-domain FFT loss), `bg_sampling.py` / `Patch_data.py` (GPU-resident slice samplers), `bg_normalize.py` (Z-score), `config_io.py` (SZ3 I/O, seeding, error-bound clamp), `bg_shard.py` (model-size selection), `train.py` (`TrainConfig`), `local_paths.py` (machine-specific path resolver, see Step 4). |
| `sec_3_1_auxiliary_fields/` | §3.1 matched-CR and enhanced auxiliary fields | `run_auxq_ablation.sh` + `plot_aux_quality.py` (auxiliary-quality ladder, Fig. 3), `run_cascade_6000.sh` (enhanced-auxiliary cascade DMD → temperature → baryon density), `run_novel_6000.sh` + `plot_velocity_ablation.py` (no-velocity siblings), `plot_cascade_orders.py`. |
| `sec_3_2_normalization/` | §3.2 Z-score normalization | `normalization.ipynb`, `norm_rerun.py` (Fig. 4), pinned results `norm_rerun_results*.json`, figures `norm_cr_time_*.pdf`. |
| `sec_3_3_frequency_loss/` | §3.3 frequency-split heads + dual-domain loss | `fft_err.ipynb` (Fig. 6). |
| `sec_3_4_model_scaling/` | §3.4 model size under a time budget | `nyx_miranda_isoepoch.ipynb`, `isoepoch_nyx_rerun.py` (Fig. 7), `isoepoch_nyx_crmatched.json`. |
| `sec_3_4_bf16_storage/` | §3.4 BF16 model storage | `bf_16.ipynb`, `bf16_rerun.py` (Fig. 8), `bf16_vs_fp32_results.json`. |
| `sec_3_5_bayesian_opt/` | §3.5 two-phase adaptive training (TPE + full-resolution) | `nyx_miranda.ipynb` (SZ3 / NYX), `nyx_miranda_sperr.ipynb` (SPERR / Miranda), `bo_combined_plot.ipynb` (Fig. 9), pinned `bo_results/*_final.pkl`. |
| **`sec_4_evaluation/`** | §4 evaluation | The paper pipeline. `SPERR_fft.py --task {nyx_b,nyx_t,nyx_d,miranda,mag,qmcpack}` runs SZ3, SPERR, AdaMit and NeurLZ for one dataset and caches the result in `sperr_fft_cache/`; `--task aux_prep` archives the matched-CR auxiliary streams; `--task neurlz_long` is the NeurLZ cost study (Table 2). `paper_numbers.py`, `plot_paper_figs.py`, `plot_paper_fft.py` turn the pinned caches (`sperr_fft_cache/PAPER_*.json`) into the paper's numbers and Figs. 10/11. |
| `sec_4_evaluation/multi_gpu/` | §4.4 one vs. four GPUs (Table 4, Fig. 13) | `bench_compress_table.py` (one dataset × codec at CR≈500 on 1 or 4 GPUs: SZ3/SPERR codec timing, Phase 1 + Phase 2 training under the wall-clock budget, per-epoch PSNR trace, one full-volume inference; `--latex` builds the matched-PSNR table), `aux_siblings.py` (decoder-reproducible NYX siblings: matched-CR archive streams or the enhanced output of an earlier cascade stage, order DMD → temperature → baryon density), `bench_infer_mgpu.py` (z-slab data-parallel inference timing), `run_cascade_nyx.sh` / `run_nyx_paper.sh` (drivers), `plot_paper_2panel.py` (Fig. 13), `bench_out/` (every run's JSON behind Table 4), `eval_parallel_section.tex`. |
| `sec_4_evaluation/power_spectrum/` | §4.3 power spectrum (Fig. 12, Table 3) | `power_spectrum_fig12.py` trains AdaMit (batch 1, enhanced siblings) and NeurLZ (its own recipe) on NYX baryon density at CR≈300 and records ε(k) after every epoch; `plot_fig12_2panel.py` draws ε(k) and best-so-far max ε(k) vs. training time; `figures/*/fig12_data.json` are the pinned traces. |
| `Reproduce/` | all | Whole-chain scripts and notes: `run_all.sh`, `collect.py` (pin results, regenerate figures and numbers), `make_reproduce_nb.py`, `REPORT.md` (every number in the paper and how it was produced), `HANDOFF.md`, `CLEANUP_PLAN.md`, `experiment/` (sibling-protocol, cascade and NeurLZ-cost experiments), `benchmarks/`. Result pickles, figures and logs are written here locally and are not tracked. |

Scripts and notebooks add `base_script/` to `sys.path` (relative to their own location) and import
its modules by bare name (`from bg_stage import ...`); `base_script/` depends on nothing else in the
repository. Every `sec_*` folder is self-contained: run its commands from inside that folder.

## Requirements

- **Hardware.** One NVIDIA GPU (the paper used an RTX PRO 6000 Blackwell, 96 GB; a 24 GB card is
  enough for the 512³ fields, Miranda 1024³ needs more) and ≥ 64 GB host RAM (the NeurLZ baseline on
  Miranda peaks around 50 GB). All results are wall-clock budgeted: run on an otherwise idle machine.
- **Software.** Linux, CUDA, Python ≥ 3.10 (tested with 3.13, PyTorch 2.10, NumPy 2.3, Optuna 4.9, MONAI 1.5).
  Python packages: `pip install -r requirements.txt`
  (`numpy torch matplotlib pandas optuna monai jupyter nbconvert`; MONAI provides the `BasicUNet`
  used by the NeurLZ baseline).
- **SZ3 + `pysz`** (error-bounded base compressor) and **SPERR** (`sperr3d` binary, wavelet base
  compressor), built from source, see Step 3.

## Step-by-step setup

### Step 1: clone and create the environment

```bash
git clone <this repository> AdaMit && cd AdaMit
conda create -n adamit python=3.11 -y && conda activate adamit
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the wheel for your CUDA
pip install -r requirements.txt
```

### Step 2: download the datasets (all public, SDRBench)

| Dataset | Files the scripts expect | Shape (z, y, x) |
|---|---|---|
| NYX (EXASKY, cosmology) | `baryon_density.f32`, `dark_matter_density.f32`, `temperature.f32`, `velocity_x.f32`, `velocity_y.f32`, `velocity_z.f32` in one directory | 512 × 512 × 512 |
| Miranda (hydrodynamics) | `miranda_1024x1024x1024_float32.raw` | 1024 × 1024 × 1024 |
| Magnetic Reconnection (plasma) | `magnetic_reconnection_512x512x512_float32.raw` | 512 × 512 × 512 |
| QMCPack (einspline coefficients) | `288x115x69x69/einspline_288_115_69_69.pre.f32` | folded to 33120 × 69 × 69 |

All files are raw little-endian float32 volumes in C order. The data are **not** part of the
repository (`*.f32`, `*.raw`, `*.d64`, `*.sz` are gitignored). The NYX notebooks write their SZ3
bitstreams (`<field>_rel<eps>.sz`) next to the NYX `.f32` files, so that directory must be writable.

### Step 3: build the base compressors

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

### Step 4: tell the code where things are (one file, no code edits)

All machine-specific locations go through `base_script/local_paths.py`. Every script and notebook
calls `P("ADAMIT_...")`, which resolves a key in this order: **environment variable** >
**`local_paths.env`** in the repository root (gitignored) > a `/path/to/...` placeholder.

```bash
cp local_paths.env.example local_paths.env    # then edit the right-hand sides
```

```ini
ADAMIT_SZ3_LIB=/path/to/SZ3/build/lib64/libSZ3c.so
ADAMIT_PYSZ=/path/to/SZ3/tools/pysz
ADAMIT_SPERR_BIN=/path/to/SPERR/build/bin/sperr3d
ADAMIT_NYX_DIR=/path/to/data/NYX/                 # directory with the six NYX .f32 files
ADAMIT_MIRANDA_FILE=/path/to/data/miranda_1024x1024x1024_float32.raw
ADAMIT_MAGNETIC_FILE=/path/to/data/magnetic_reconnection_512x512x512_float32.raw
ADAMIT_QMC_FILE=/path/to/data/SDRBENCH-QMCPack/288x115x69x69/einspline_288_115_69_69.pre.f32
```

Optional keys (defaults in parentheses): `ADAMIT_CACHE_DIR` (`sec_4_evaluation/sperr_fft_cache`,
result pickles + auxiliary streams), `ADAMIT_VIZ_DIR` (`Reproduce/viz`, volume exports for ParaView),
`ADAMIT_CASCADE_DIR` (`Reproduce/cascade`, enhanced-auxiliary exports), `ADAMIT_CKPT_DIR`
(checkpoints of the notebook ablations), `ADAMIT_IO_BENCH_TMP` (`/tmp/adamit_io_bench`, scratch for
the I/O benchmark). Shell scripts find the repository from their own location; override with
`ADAMIT_REPO`, and pick interpreters with `PYTHON` / `JUPYTER` (default: whatever is on `PATH`).

Check the configuration:

```bash
python -c "import sys; sys.path.append('base_script'); from local_paths import P, DEFAULTS; import os
[print(f'{k:22s} {P(k)}  exists={os.path.exists(P(k))}') for k in DEFAULTS]"
```

### Step 5: archive the matched-CR auxiliary streams (once)

```bash
cd sec_4_evaluation
python SPERR_fft.py --task aux_prep
```

This compresses every NYX field with SZ3 and SPERR at CR levels 100…500 (level 600 is built on
demand) and stores the streams in `sperr_fft_cache/aux_streams/` (~140 MB, CPU only, ~25 min).
Every later run reads the siblings from this archive, so training and inference see exactly the
copies a decoder would hold (§3.1). The §3 notebooks reuse the same archive.

### Step 6: run one dataset of the evaluation (§4)

```bash
cd sec_4_evaluation
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

### Step 7: reproduce the full paper chain

```bash
bash Reproduce/run_all.sh          # ~4.5 h: aux_prep -> qmcpack -> nyx_b/t/d -> mag -> miranda -> Fig. 9 BO halves
python Reproduce/collect.py        # pins the six result pickles, writes paper_numbers.txt, Figs. 9-11, REPRODUCE.ipynb
bash sec_3_1_auxiliary_fields/run_cascade_6000.sh    # enhanced-auxiliary cascade for the NYX fields (Table 2 / Figs. 10-11 NYX rows)
cd sec_4_evaluation
python plot_paper_figs.py PAPER_ENHANCED_MIXED_CACHES.json   # PSNR vs CR from a pin file
python plot_paper_fft.py  PAPER_ENHANCED_MIXED_CACHES.json   # FFT errors vs CR
python paper_numbers.py   PAPER_ENHANCED_MIXED_CACHES.json   # the numbers quoted in §4
```

`collect.py` pins results by the run's start/exit stamps in `Reproduce/logs/run_all.log`, so run the
chain uninterrupted. Outputs land in `Reproduce/results/` (pins, `paper_numbers.txt`) and
`Reproduce/figures/`. The pins the paper was built from are tracked in
`sec_4_evaluation/sperr_fft_cache/PAPER_*.json` together with the pickles they point to, so the
plotting commands above work without rerunning anything.

### Step 8: multi-GPU experiments (§4.4, Table 4 / Fig. 13) — this branch only

Hardware used in the paper: one node with four NVIDIA A100 80 GB **PCIe** GPUs (no NVLink; each GPU
on its own NUMA domain, so all-reduce crosses PCIe 4.0 and the socket interconnect), two 32-core
AMD EPYC 9334, 1 TB RAM. Runs are launched with `torch.distributed.run`; the volume is split into
contiguous z-shards (128 slices per GPU for 512³ fields, 256 for Miranda), every GPU samples from its
own shard and gradients are averaged before each update (DDP). Protocol: the single GPU trains for
the budget (10 s for 512³ fields, 60 s QMCPack, 80 s Miranda) with a batch of 4 slices per update
(1024 for QMCPack); the four-GPU run keeps the same batch per update (1 slice per GPU; 256 for
QMCPack) and is timed to the single-GPU PSNR. Decompression uses z-slab inference on four GPUs,
bit-identical to the single-GPU pass.

```bash
cd sec_4_evaluation/multi_gpu
# 1) enhanced NYX siblings for CR 500 (cascade DMD -> temperature -> baryon; ~10 min on one GPU)
CR=500 OUT=bench_out/table_cascade ENH=bench_out/enhanced_cr500 bash run_cascade_nyx.sh 1
# 2) one row of Table 4: single GPU, then four GPUs (both read the enhanced siblings)
python bench_compress_table.py --dataset nyx_b --codec sz3 --cr 500 --train-s 10 \
       --aux-mode enhanced --aux-enhanced-dir bench_out/enhanced_cr500 --out bench_out/table --tag nyx_b_sz3_n1
python -m torch.distributed.run --nproc_per_node=4 bench_compress_table.py --dataset nyx_b --codec sz3 --cr 500 \
       --train-s 10 --aux-mode enhanced --aux-enhanced-dir bench_out/enhanced_cr500 --out bench_out/table --tag nyx_b_sz3_n4
#    (non-NYX datasets: drop the --aux-* flags; QMCPack adds --tot-batch 1024; pin a configuration with --axis K --lr_abs LR)
# 3) the table and Fig. 13
python bench_compress_table.py --latex bench_out/table/*.json --iso budget
python plot_paper_2panel.py
# 4) four-GPU inference timing used in the Dec. columns
python -m torch.distributed.run --nproc_per_node=4 bench_infer_mgpu.py --dataset nyx_b
```

`bench_out/` already contains the JSON of every run behind the paper's Table 4, so step 3 works
without rerunning anything. Per-rank sampling seeds are decorrelated and the training-budget clock is
opened behind a barrier (`base_script/bg_stage.py`), both required for the four-GPU numbers to be
meaningful. SPERR is timed single-threaded in both configurations.

### Step 9: power-spectrum compliance (§4.3, Fig. 12 / Table 3) — this branch only

```bash
cd sec_4_evaluation/power_spectrum
# AdaMit (batch 1, three seeds) and NeurLZ (100 epochs of its own recipe) at CR~300, eps(k) after every epoch
python power_spectrum_fig12.py --cr 317.4 --budget 40 --seed 17 --batch 1 \
       --aux-mode enhanced --aux-enhanced-dir ../multi_gpu/bench_out/enhanced_cr300 --out figures/fig12_b1_s17
python plot_fig12_2panel.py --data figures/fig12_b1_s17 --neurlz-data figures/fig12_enh
```

Base CR 317.4 gives an effective CR of 306.7 once the 60 KB model is charged to the payload (SZ3's
CR is a step function of the error bound here: base ratios between 304 and 317 are unreachable).
The NeurLZ branch has no cascade and reads matched-CR siblings; it is reused from `figures/fig12_enh`.

## Ablations, section by section

Each ablation lives in the folder of its paper section. Run the commands **from inside that folder**
(the notebooks resolve `../base_script` from the working directory). Every notebook can also be
executed headless with `jupyter nbconvert --to notebook --execute --inplace <notebook>`.

### §3.1 Auxiliary fields (`sec_3_1_auxiliary_fields/`)

Prerequisite: Step 5 (`aux_prep`) and the Step 6 runs of `nyx_b`, `nyx_t`, `nyx_d` (the ladder
compares against those matched-CR results).

```bash
cd sec_3_1_auxiliary_fields
bash run_auxq_ablation.sh          # original-aux (SPERR_AUX_MODE=orig) and no-aux (SPERR_USE_AUX=0) runs of NYX baryon + temperature
python plot_aux_quality.py         # Fig. 3: no aux / decompressed aux / enhanced aux / original aux  -> Reproduce/figures/nyx_aux_quality_ladder.pdf
bash run_cascade_6000.sh           # enhanced-auxiliary cascade: stage 1 DMD -> stage 2 temperature -> stage 3 baryon density
                                   #   (exports each stage's enhanced field to $ADAMIT_CASCADE_DIR and feeds it to the next stage via SPERR_AUX_ENHANCED_DIR)
bash run_novel_6000.sh             # siblings without the velocity fields (robustness check in §3.1)
python plot_velocity_ablation.py   # figure for that check
python plot_cascade_orders.py      # compares the two cascade orders recorded in Reproduce/experiment/EXPERIMENT_PINS.json
```

The cascade script writes the merged NYX rows the paper uses (`NYX_*__cascA_ours_refNeurLZ.pkl`,
pinned in `PAPER_ENHANCED_MIXED_CACHES.json`). To evaluate one cascade stage by hand:

```bash
cd sec_4_evaluation
SPERR_SAVE_RECONS_DIR=$ADAMIT_CASCADE_DIR/stage1 python -u SPERR_fft.py --task nyx_d              # stage 1, exports enhanced DMD
SPERR_AUX_ENHANCED_DIR=$ADAMIT_CASCADE_DIR/stage1 python -u SPERR_fft.py --task nyx_t             # stage 2 uses enhanced DMD
```

### §3.2 Normalization (`sec_3_2_normalization/`)

```bash
cd sec_3_2_normalization
python norm_rerun.py                                   # temperature: Z-score vs Min-Max, PSNR vs CR and PSNR vs time  -> norm_cr_time_temperature.pdf
NORM_TARGET=baryon_density \
NORM_RELS=1e-6,1.4e-6,2.4e-6,4e-6,6.9e-6,1e-5,1.4e-5 \
python norm_rerun.py                                   # same for baryon density                                     -> norm_cr_time_baryon_density.pdf
```

`norm_rerun.py` executes the setup cells of `normalization.ipynb` (same trainer, CR-matched siblings
from `aux_streams/`) and writes `norm_rerun_results*.json`. Open the notebook for the interactive
version.

### §3.3 Frequency heads and dual-domain loss (`sec_3_3_frequency_loss/`)

```bash
cd sec_3_3_frequency_loss
jupyter nbconvert --to notebook --execute --inplace fft_err.ipynb    # Fig. 6: spatial-only vs frequency-split heads + FFT loss, plus the lambda_phase sweep
```

The rel-bound ladder and the target field are set at the top of the notebook (`TARGET_STEM`,
`REL_SETTINGS`); `BG_CKPT_ROOT = Path(P("ADAMIT_CKPT_DIR"))` keeps per-variant checkpoints.

### §3.4 Model size and BF16 storage (`sec_3_4_model_scaling/`, `sec_3_4_bf16_storage/`)

```bash
cd sec_3_4_model_scaling
python isoepoch_nyx_rerun.py       # Fig. 7: parameter budgets 8k-136k at fixed epochs, NYX baryon + temperature -> psnr_vs_cr_isoepoch_nyx_baryon_temp.pdf
                                   # Miranda panel: run nyx_miranda_isoepoch.ipynb (cells 6-7)

cd ../sec_3_4_bf16_storage
python bf16_rerun.py               # Fig. 8: FP32 vs BF16 weights, NYX (30k params, 10 s) and Miranda (240k, 80 s) -> bf16_vs_fp32_1x2.pdf
```

Both rerun scripts exec the engine cells of their notebook so the sweep uses exactly the notebook's
code path; `BO_SAMPLE_MODE=sequential` switches the slice sampler for comparison.

### §3.5 Two-phase adaptive training (`sec_3_5_bayesian_opt/`)

```bash
cd sec_3_5_bayesian_opt
jupyter nbconvert --to notebook --execute --inplace nyx_miranda.ipynb         # SZ3 side: NYX baryon 10 s and Miranda 80 s  -> bo_results/sz3_*_final.pkl
jupyter nbconvert --to notebook --execute --inplace nyx_miranda_sperr.ipynb   # SPERR side                                   -> bo_results/sperr_*_final.pkl
jupyter nbconvert --to notebook --execute --inplace bo_combined_plot.ipynb    # Fig. 9: Phase-1 proxy ranking + Phase-2 full-resolution curves
```

The notebooks show the Phase-1 TPE search over (learning rate, slice axis) on the half-resolution
proxy and the Phase-2 sweep at full resolution, with the Optuna studies stored in
`bo_results/optuna_studies.sqlite3` (`optuna-dashboard sqlite:///bo_results/optuna_studies.sqlite3`).
Knobs: `BO_AUX_MODE=orig` (original instead of matched-CR siblings), `BO_LR_MAX`, `BO_DEVICE`.

### §4 Evaluation extras (`sec_4_evaluation/`, `Reproduce/experiment/`)

```bash
cd sec_4_evaluation
SPERR_NLZ_EPOCHS=100 python -u SPERR_fft.py --task neurlz_long                 # NeurLZ 100-epoch cost study, SZ3 side (Table 2)
SPERR_NLZ_SIDE=sperr SPERR_NLZ_EPOCHS=100 python -u SPERR_fft.py --task neurlz_long   # SPERR side
python ../Reproduce/experiment/neurlz_long/plot_neurlz_long.py                 # time-to-match figure
SPERR_NYX_RELS=1.28e-5 SPERR_SAVE_RECONS_DIR=$ADAMIT_VIZ_DIR/NYX_baryon_density_sz3_cr530 python -u SPERR_fft.py --task nyx_b   # volume export for Figs. 1/12 (ParaView)
```

`Reproduce/experiment/neurlz_long/run_100ep.sh` and `run_100ep_sperr.sh` wrap the first two commands
with the paper's dataset list and caps.

## Use AdaMit on your own field

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

`sec_4_evaluation/SPERR_fft.py::bench_field_fft` shows the complete recipe including the Phase-1
search (`_phase1_best_fast`) and the model-size budgeting (`bg_shard.pick_bg_h_under_budget`).

## Notes

- The pipeline is deterministic up to cuDNN autotuning; Phase-1 picks can differ between runs by a
  few tenths of a dB on 512² slices (QMCPack runs are bit-reproducible). `SPERR_DET=1` forces
  deterministic kernels at some speed cost.
- Miranda's NeurLZ pass needs ~50 GB of host RAM; never run two training jobs on one machine, the
  wall-clock budgets would no longer be comparable.
- `Reproduce/REPORT.md` documents every number in the paper and how it was produced.
