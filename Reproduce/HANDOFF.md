# AdaMit — hand-off for the multi-GPU and power-spectrum tasks (2026-08-29)

> **Folder names (2026-09-12):** the repository folders were renamed after the paper sections: `Normalization/`→`sec_3_2_normalization/`, `frequency_head_loss/`→`sec_3_3_frequency_loss/`, `Model_parameter_Scaling/`→`sec_3_4_model_scaling/`, `bf_16_vs_32/`→`sec_3_4_bf16_storage/`, `BO_Adaptive/`→`sec_3_5_bayesian_opt/`, `SPERR/`→`sec_4_evaluation/`; the aux-quality / cascade scripts moved from `Reproduce/` to `sec_3_1_auxiliary_fields/`. Machine paths are now resolved by `base_script/local_paths.py`.

This is everything another machine needs to (1) re-run the multi-GPU / parallel experiment (paper Table 4) and
(2) produce the power-spectrum figure (paper Fig. 12: ε(k) of NYX baryon density at CR≈300 and "time to meet the
Nyx 1 % requirement") under the paper-final sibling protocol, optionally with **enhanced** siblings (cascade).
Repo: `<repo>` (git, branch main). Companion files: `Reproduce/REPORT.md` (full results
report, Chinese), `Reproduce/REPRODUCE.ipynb` (executed reproducibility notebook), `Reproduce/experiment/summary.md`.

## 1. What the project is

AdaMit: online neural error mitigation on top of an error-bounded base compressor (SZ3 or SPERR). A small 2-D CNN
(`base_script/siren_fft_backbone_model.py::UNET_Model`, ~30k params, bf16 weights charged to the CR) is trained per
field at compression time on the residual `X − X′` (X′ = decompressed base), with a split-band frequency loss
(`base_script/frequency_losses.py`). Decode = base decompress + model inference + error-bound clamp
(`base_script/bg_stage.py::run_bg_inference`). A two-phase search picks (learning rate, slice axis) per operating point:
Phase 1 = 10 Optuna-TPE trials on a strided proxy inside 10 % of the wall-clock budget, Phase 2 = full-resolution
training for the remaining 90 %. Baseline: NeurLZ (MONAI BasicUNet, min-max normalization, lr 1e-2, parameter- and
time-matched, same inputs) — `sec_4_evaluation/SPERR_fft.py::run_neurlz`.

## 2. Paper-final protocol (what was decided in the last two days)

| item | value | where |
|---|---|---|
| lr search window | [1e-3, 1e-2] (log), plain argmax, no trust gates | `sec_4_evaluation/SPERR_fft.py` BO_LR_MIN/BO_LR_MAX, BO_MIN_*_DB=-1 |
| slice sampling | `shuffled`: every epoch visits every depth slice once, random order (NeurLZ's own loop) | `base_script/bg_sampling.py::_shuffled_z`, cfg.bg_sample_mode |
| cosine schedule | epoch-level re-planning + step-level re-planning when one epoch would exceed 80 % of the budget (QMCPack) | `base_script/bg_stage.py` cfg.bg_sched_time_calibrate / bg_sched_step_calibrate |
| **sibling (aux) protocol** | each sibling field is **archived by the same base compressor at the CR level {100,…,600} closest (log-CR) to the target's CR; the SAME decompressed siblings feed Phase-1 proxies, training, inference and NeurLZ**. Training on originals and inferring on decompressed siblings collapses (NYX temperature → +0.00 dB), so never mix. | `SPERR_fft.py` AUX_MODE='cr_matched', `_aux_at_cr_level`, `_use_aux_for_cr`; streams in `sec_4_evaluation/sperr_fft_cache/aux_streams/` (`<sz3|sperr>_<field>_f32_cr<L>.bin/.json`, built by `python SPERR_fft.py --task aux_prep`) |
| **enhanced siblings (cascade, experiment)** | decode fields in order; a field enhanced by an earlier stage is fed ENHANCED to later ones. Order B: DMD → temperature → baryon gives baryon +6.4 dB mean (+1.7 over the paper protocol); order A: DMD → baryon → temperature gives temperature +3.5 (+0.7). Stage 1 is always DMD (insensitive to sibling quality). | `SPERR_AUX_ENHANCED_DIR=<dir>`: a sibling that has `<dir>/NYX_<field>_<sz3|sperr>_cr*/ours.f32` (exported by an earlier stage with `SPERR_SAVE_RECONS_DIR=<dir> SPERR_SAVE_RECONS_ONLY=ours`) is loaded enhanced (closest-CR export). See `sec_3_1_auxiliary_fields/run_cascade_6000.sh`. Exports of both orders: `<storage>/Final_visualization/cascade_2026-08-29/{stage1,A,B}` (26 GB, this node only). |
| budgets / model sizes | NYX 10 s / 30k params, Miranda 80 s, QMCPack 60 s / 35k, Magnetic 10 s; rel bands aligned to CR≈100/150/220/320/500 | `SPERR_fft.py` dataset blocks |
| platform | one RTX PRO 6000 Blackwell (cuda:0), 20 cores, 62 GB RAM; **never run two training jobs at once** (Miranda 1024³ peaks ~50 GB host RAM; a concurrent NYX job OOM-killed it) | — |

Final numbers (six-dataset means): SZ3 side +3.10 dB, SPERR side +2.36 dB; per-point tables in
`Reproduce/results/paper_numbers.txt`; old→new vs the draft in `paper_numbers_vs_v3.txt`. Pins:
`Reproduce/results/PAPER_FINAL_CACHES.json` (six pkls; Miranda has its CR≈500 band replaced by the median of 3 standalone
repeats — `PIN_OVERRIDES.json`), `CASCADE_{A,B}_CACHES.json` (NYX rows from the cascade). Every pkl holds, per series
(`sz3`, `pipe`=ours, `neurlz`, `sperr`, `sperr_pipe`, `sperr_neurlz`): lists `CR`, `PSNR`, `fft_mag`, `fft_phase`.

## 3. Code hooks you will need

* Run one dataset: `cd SPERR && python -u SPERR_fft.py --task nyx_b|nyx_t|nyx_d|miranda|qmcpack|mag` (cache:
  `sperr_fft_cache/<name>__<config-hash>.pkl`; the hash covers every constant, `SPERR_FORCE_RETRAIN=1` overrides a hit).
* Restrict to one operating point: `SPERR_ONLY_BANDS=<i>` (index into the dataset's rel list; the SPERR side is then
  bisected to the same CR). NYX baryon: band 3 = CR 330, band 4 = CR 496; for CR≈300 use rel 5.5e-6 ≈ CR 300 (the FAST
  benchmark used rel 5e-6 → CR 306; `benchmarks/cr300_1epoch_timing.py`).
* Export reconstructions: `SPERR_SAVE_RECONS_DIR=<dir>` writes, per operating point, `base/ours/neurlz.{f32,vtk}`,
  `err_*.f32`, `meta.json` (raw float32, (z,y,x) C-order; `.vtk` = legacy STRUCTURED_POINTS, opens in ParaView).
  `SPERR_SAVE_RECONS_ONLY=ours` keeps only `ours.f32`. Existing exports: `<storage>/Final_visualization/final_2026-08-29/`
  (NYX baryon CR≈496 and temperature CR≈312, SZ3 + SPERR sides; 19 GB).
* Other env knobs: `SPERR_SAMPLE_MODE`, `SPERR_LR_MIN/MAX`, `SPERR_AUX_MODE` (cr_matched|orig), `SPERR_AUX_FIELDS`
  (sibling subset), `SPERR_AUX_CR_LEVELS`, `SPERR_QMC_PARAMS`, `SPERR_STEP_CALIB`, `SPERR_DET` (deterministic cuDNN).
* Standalone use of the sibling archive (no SPERR_fft import needed) — the helper used by the BO / iso-epoch /
  normalization reruns (`sec_3_4_model_scaling/isoepoch_nyx_rerun.py::sibling`, `sec_3_2_normalization/norm_rerun.py`):
  ```python
  level = min((100,200,300,400,500,600), key=lambda L: abs(np.log(L) - np.log(target_cr)))
  stem  = f"{AUX_STREAM_DIR}/sz3_{field}_f32_cr{level}"           # or sperr_... for a SPERR pipeline
  dec   = sz_engine.decompress(np.fromfile(stem + ".bin", np.uint8), (512,512,512), np.float32)
  ```
* Per-epoch evaluation hook (for "time to reach X"): `train_bg_only(Xs, Xps, device, cfg, evaluator=fn)` calls
  `fn(model) -> (metric, aux)` at every epoch end and records `hist["time"]`, `hist["psnr"]`; `run_neurlz(...,
  return_history=True)` returns `{"time": [...], "psnr": [...]}` for NeurLZ. Replace the PSNR inside the evaluator by a
  power-spectrum error to get ε(k) trajectories (the evaluator receives the model; call `run_bg_inference(model, Xs, Xps, cfg, rel)`).
* Multi-GPU: `base_script/bg_stage.py` supports `cfg.bg_data_parallel=True` (torch.nn.DataParallel, one model,
  `bg_batch=4`, `cfg.bg_sample_mode="z_shard"` = the four slices of a step come from four z-shards) and `cfg.bg_ddp`
  (torchrun scripts `MultiGPU_DDP/data_parallel_for_{NYX,Miranda}.py`). The iso-time 1-GPU vs 4-GPU comparison that
  produced Table 4 is `MultiGPU_DDP/parallel_compute.ipynb` (NYX dark-matter density, **lossless siblings** — must be
  redone with the sibling archive; DMD is the least sibling-sensitive field, so numbers will move little).
  `MultiGPU_DDP/shard_expert.ipynb` = per-z-chunk experts ablation. Sampling mode for the multi-GPU run should be
  `z_shard` per GPU-batch (that is what the paper describes) — keep it, and state it.

## 4. Task A — multi-GPU (Table 4) under the new protocol

1. Copy `sec_4_evaluation/sperr_fft_cache/aux_streams/` (140 MB, in `handoff_aux_streams.tar.gz`) next to the NYX data, or rebuild
   with `python sec_4_evaluation/SPERR_fft.py --task aux_prep` (CPU, ~27 min; needs the six NYX `.f32` files and SZ3/SPERR builds).
2. In `MultiGPU_DDP/parallel_compute.ipynb` (or the `.py` DDP scripts) replace the sibling loading (`FIELD_FILES` →
   `load_multifield_from_disk` / memmaps of the originals) by the `sibling(field, target_cr)` helper above, using the
   target's SZ3 CR at the chosen rel; keep the same decompressed siblings for training and evaluation.
3. Keep the paper's framing: equal wall-clock budget, 1 GPU vs 4 GPUs, report PSNR reached in the same time (and/or
   time to reach the 1-GPU PSNR). Use `cfg.bg_sample_mode="z_shard"` for the 4-GPU DataParallel run and `"shuffled"`
   for the 1-GPU run (or z_shard for both — say which).
4. Optional "enhanced aux" variant: export enhanced DMD / temperature with the cascade (`sec_3_1_auxiliary_fields/run_cascade_6000.sh`
   pattern) and point `SPERR_AUX_ENHANCED_DIR` at it, or load `<export>/NYX_<field>_sz3_cr<L>/ours.f32` directly as the
   sibling volume.

## 5. Task B — power spectrum (Fig. 12) under the new protocol

Fig. 12 (draft): left = ε(k), the relative error of the 3-D power spectrum P(k) of the reconstruction vs the original,
per k-bin, for SZ3 / SZ3+NeurLZ (107 s) / SZ3+Ours (3 s) on NYX baryon density at effective CR≈300, with the Nyx 1 %
requirement line; right = best max_k ε(k) vs pure training wall time (time to meet 1 %). The code that made the draft
figure is NOT in this repo (it was produced elsewhere; only a per-slice radial FFT-error figure exists in
`_archive_2026-09-10/SPERR_visualization.ipynb`). Reproduce it as follows:

1. Volumes: run `SPERR_SAVE_RECONS_DIR=<dir> SPERR_ONLY_BANDS=3 SPERR_FORCE_RETRAIN=1 python -u SPERR_fft.py --task nyx_b`
   (CR≈330 band; or add rel 5.5e-6 to `NYX_REL["baryon_density"]` for CR≈300) → `base/ours/neurlz.f32` at the paper
   protocol. For the enhanced-aux variant add `SPERR_AUX_ENHANCED_DIR=<cascade export dir>` (order B exports of DMD and
   temperature are in `<storage>/Final_visualization/cascade_2026-08-29/B` on this node; regenerate elsewhere with
   `../sec_3_1_auxiliary_fields/run_cascade_6000.sh`).
2. ε(k): 3-D FFT of original and reconstruction (numpy `rfftn` on the 512³ float32 volume, double precision), P(k) =
   spherically binned |F|², ε(k) = |P̂(k) − P(k)| / P(k) per bin; k in units of the fundamental mode (bins 1…~9 as in the
   draft). Baryon density is heavy-tailed: use the field as is (that is what the draft did) and state it.
3. Time-to-1 %: re-train with an evaluator that computes max_k ε(k) at every epoch end (`train_bg_only(..., evaluator=)`),
   for ours at the paper budget and NeurLZ for 100 epochs (`run_neurlz(..., return_history=True)` gives per-epoch PSNR;
   add the spectrum inside its `_enhanced()` loop or evaluate from saved epoch outputs). Report pure training wall time.
4. Both variants (decompressed siblings = paper protocol; enhanced siblings = cascade) — the figure's story (ours meets
   1 % in seconds, NeurLZ in ~100 s) is expected to hold; the absolute ε(k) of ours will be somewhat higher than the draft
   because the draft used lossless siblings.

## 6. Data and environment

* NYX 512³ fields: `~/Halo_Finder/halo_finder_v1/SDRBENCH-EXASKY-NYX-512x512x512/origin/{baryon_density,temperature,
  dark_matter_density,velocity_x,velocity_y,velocity_z}.f32` (float32, (z,y,x)); Miranda `halo_finder_v1/miranda_1024x1024x1024_float32.raw`;
  QMCPack `<storage>/SDRBench/SDRBENCH-QMCPack/288x115x69x69/einspline_288_115_69_69.pre.f32` (33120×69×69);
  Magnetic `halo_finder_v1/magnetic_reconnection_512x512x512_float32.raw`. All SDRBench.
* SZ3: `~/Data_Compression/SZ3/build/lib64/libSZ3c.so` + `tools/pysz` (pysz wrapper; `compress(x,1,0,rel,0)` = REL
  mode); SPERR: `~/Halo_Finder/SPERR/build/bin/sperr3d` (`--psnr` target). Edit the paths at the top of
  `sec_4_evaluation/SPERR_fft.py` on the other node. Python: torch (CUDA), numpy, optuna, monai (NeurLZ's BasicUNet), matplotlib.
* NeurLZ reference code: `~/Halo_Finder/halo_finder_v1/neurlz/neurlz/train.py` (it decompresses every sibling at
  the target's rel, `rel_list=[rel]*n`, and feeds the same decompressed inputs at train and eval — same principle as ours).

## 7. Pitfalls (learned the hard way)

* Never overlap a Miranda-scale job with anything (OOM); serialize with one queue script (`Reproduce/run_after_chain.sh` pattern).
* `pgrep -f`/`pkill -f` on a script name matches the calling shell too; use a `[x]` regex trick or pidfiles.
* Results are wall-clock budgeted: run on an idle machine; numbers from a busy/slower GPU are not comparable.
* cuDNN autotune makes Phase-1 picks vary run to run (±0.008 dB proxy noise flips argmax when the lr spread is small);
  Miranda-SPERR CR≈500 picked a non-converging lr (6.7e-3) in 3/5 runs → report medians of repeats where it matters.
* Don't pin results by "newest pkl": several experiments share the same pkl prefixes; pin by the run's own start/exit stamps.
* Sibling protocol: same decompressed siblings for train and inference, always; the lossless-sibling numbers in the
  old draft (+12 dB on NYX) are not decoder-reproducible and must not come back.

## 8. Files in this hand-off

`handoff_2026-08-29.tar.gz`: this file, REPORT.md, `Reproduce/` (results pins + pkls, figures, experiment/, logs, scripts,
notebook), `sec_4_evaluation/` scripts + cache pins (no big pkl trees), `base_script/`, `sec_3_5_bayesian_opt/` notebooks + pickles,
`MultiGPU_DDP/`, `sec_3_4_model_scaling/isoepoch_nyx_rerun.py` (+json), `sec_3_4_bf16_storage/bf16_rerun.py` (+json),
`sec_3_2_normalization/norm_rerun.py` (+json), `benchmarks/`, README.md, requirements.txt.
`handoff_aux_streams.tar.gz`: the sibling archive (60 streams, 140 MB). Large volumes (viz/cascade exports, 45 GB) stay on
this node under `<storage>/Final_visualization/`.
