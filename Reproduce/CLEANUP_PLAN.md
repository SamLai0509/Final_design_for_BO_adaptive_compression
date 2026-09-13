# Code cleanup plan (2026-09-10) — proposal, nothing moved yet

> **Folder names (2026-09-12):** the repository folders were renamed after the paper sections: `Normalization/`→`sec_3_2_normalization/`, `frequency_head_loss/`→`sec_3_3_frequency_loss/`, `Model_parameter_Scaling/`→`sec_3_4_model_scaling/`, `bf_16_vs_32/`→`sec_3_4_bf16_storage/`, `BO_Adaptive/`→`sec_3_5_bayesian_opt/`, `SPERR/`→`sec_4_evaluation/`; the aux-quality / cascade scripts moved from `Reproduce/` to `sec_3_1_auxiliary_fields/`. Machine paths are now resolved by `base_script/local_paths.py`.

Legend: DELETE = safe to remove; ARCHIVE = move to `_archive_2026-09-10/` (reversible, not in git);
TRIM = edit inside a file; KEEP = part of the paper's final pipeline.

## KEEP (final pipeline)
- `base_script/` — all modules except `metrics.py` are in the import closure of
  `SPERR/SPERR_fft.py` (bg_stage -> config_io, frequency_losses, Patch_data, bg_normalize, bg_sampling;
  experiment -> train; SPERR_fft -> bg_shard.pick_bg_h_under_budget). DELETE `base_script/metrics.py` (unreferenced).
- `SPERR/SPERR_fft.py`, `paper_numbers.py`, `plot_paper_figs.py`, `plot_paper_fft.py`, `sperr_fft_cache/`
  (pinned pkls + `aux_streams/` + pin jsons), `overleaf_figures/`.
- `Reproduce/` (scripts, results, figures, experiment/, logs, REPORT.md, HANDOFF.md, REPRODUCE.ipynb).
- `BO_Adaptive/{nyx_miranda.ipynb, nyx_miranda_sperr.ipynb, bo_combined_plot.ipynb, bo_results/, NYX_SZ3_Miranda_SPERR_1x4.pdf}` (Fig 9).
- `Model_parameter_Scaling/{nyx_miranda_isoepoch.ipynb, isoepoch_nyx_rerun.py, isoepoch_nyx_crmatched.json, psnr_vs_cr_isoepoch_nyx_baryon_temp.pdf}` (Fig 7).
- `bf_16_vs_32/{bf_16.ipynb, bf16_rerun.py, bf16_vs_fp32_results.json, bf16_vs_fp32_1x2.pdf}` (Fig 8).
- `Normalization/{normalization.ipynb, norm_rerun.py, norm_rerun_results*.json, norm_cr_time_baryon_density.pdf, norm_cr_time_temperature.pdf}` (Fig 4).
- `frequency_head_loss/fft_err.ipynb` + `fft_err_psnr_vs_cr_baryon_density.pdf` (Fig 6 — still the July lossless-aux version; keep until rerun).
- `MultiGPU_DDP/{parallel_compute.ipynb, data_parallel_for_NYX.py, data_parallel_for_Miranda.py}` (Table 4 / Fig 14, other node).
- `figures/{pipeline.svg, unet.svg, unet_block.svg/.pdf, phase1_sampling.svg, panels/}` (Fig 2 / architecture sources).
- `SDRBench/` (29 GB data, gitignored), `README.md`, `LICENSE`, `requirements.txt`.

## TRIM inside SPERR/SPERR_fft.py (~550 of 2085 lines)
- Dataset blocks for datasets not in the paper: warpx (1415), hurricane (1498), cloud (1524), mir256 (1566),
  mir256_noaux (1592), density (1625), scale (1656), s3d (1681), scale_t (1705), gcldlwp (1728), cesm_cloud (1757)
  + their constants and `--task` choices.
- The combined-figure block `if TASK in ("all", "plot")` (1908–2085): superseded by plot_paper_figs.py / plot_paper_fft.py.
- Docstring references to "SPERR.py's bench_field" once SPERR.py is archived.

## ARCHIVE (superseded / exploratory, not cited by the paper)
- `SPERR/SPERR.py` (legacy 1389-line predecessor; only old notebooks import it), `SPERR/SPERR.ipynb`,
  `SPERR/SPERR_fft.ipynb` (notebook twin of SPERR_fft.py), `SPERR/SPERR_visualization.ipynb` + `slice_viz_nyx_temperature.npz` (9 MB),
  `SPERR/residual_band_comparison.png`, `SPERR/scale_vs_baryon_residual_bands.png` (9 MB), `SPERR/figures/` (07-29),
  `SPERR/sperr_cache/`, `SPERR/fft_err.sh`, `SPERR/sperr.sh` (SLURM launchers; keep only if you still use the cluster).
- `Model_parameter_Scaling/`: all_dataset, deep_model, hurricane, Magnetic_reconnection, Miranda_parameters, NYX,
  nyx_miranda_scale, nyx_scale_letkf, nyx_temp_qmcpack_isoepoch, s3d, warpx_parameters notebooks; relsweep_* pdfs,
  psnr_vs_depth_nyx.pdf, isoepoch_miranda / nyx_miranda500k / nyxtemp_qmcpack pdf+png, *_origaux.pdf.
- `BO_Adaptive/`: autotune/, lr_Miranda.py, lr_NYX.py, lr_slice_*.ipynb (06-17), NYX_Magnetic*.ipynb/.pdf,
  SPERR_NYX_Magnetic*, NYX_Miranda_1x4.pdf, SPERR_NYX_Miranda_1x4.pdf, run_logs/.
- `Discussion/` (NeurLZ trajectory + normalization histograms, exploratory), `25Dmodel/`, `Final_visualization/`
  (make_viz_data.ipynb depends on SPERR.py; superseded by SPERR_SAVE_RECONS_DIR exports),
  `frequency_head_loss/{frequency_head.ipynb, frequency_loss.ipynb, frequency_loss.py, lambda_phase_sweep*, psnr_vs_time_norm_temperature.pdf, fft_err_psnr_vs_cr_temperature.pdf}`,
  `Normalization/{ablation_norm_temperature.pdf, psnr_vs_time_norm_temperature.pdf, *_origaux.pdf}`,
  `bf_16_vs_32/{bf16_vs_fp32_miranda.pdf, bf16_vs_fp32_nyx.pdf, *_origaux.pdf}`,
  `MultiGPU_DDP/{4_gpus_1_model.ipynb, shard_expert.ipynb}`, `figures/visualization.{ipynb,png,svg}` (old Fig 1, 7 MB).

## DELETE
- `_trash_2026-08-28/` (73 files, already triaged on 08-28).
- `benchmarks/` at repo root (byte-identical copy of `Reproduce/benchmarks/`).
- 11 unpinned top-level pkls in `SPERR/sperr_fft_cache/` (move to `_stale/` first if unsure).
- all `__pycache__/`, `.ipynb_checkpoints/`.
- `db.sqlite3` (124 KB SQLite, tracked in git, unrelated to the pipeline) — `git rm --cached` + gitignore.

## MOVE OUT OF THE REPO (not code)
- `Reproduce/handoff_aux_streams.tar.gz` (137 MB) and `handoff_2026-08-29.tar.gz` (14 MB) -> /storage.

## DONE 2026-09-10
- 108 files (42 MB) moved to `_archive_2026-09-10/` (gitignored, reversible with `mv`).
- Deleted: `_trash_2026-08-28/`, root `benchmarks/`, `base_script/metrics.py`, `db.sqlite3` (untracked), all `__pycache__`.
- Data offloaded to `<storage>/repo_offload/`: `handoff_*.tar.gz` (150 MB), `aux_streams/` (145 MB, symlinked back into
  `SPERR/sperr_fft_cache/`), plus `SPERR_fft.py.bak_2026-09-10` (pre-trim backup).
- `SPERR/SPERR_fft.py` trimmed 2085 -> 1566 lines (WarpX, Hurricane, CLOUD, Miranda-256, density, SCALE, S3D, SCALE-T,
  GCLDLWP, CESM blocks and the combined-plot block removed; CLI choices = nyx_b/nyx_t/nyx_d/miranda/mag/qmcpack/all/aux_prep/neurlz_long).
  Verified: py_compile OK; `--task nyx_b` hits the pinned cache (hash unchanged).
- `.gitignore` += *.bin, *.tar.gz, *.npz, *.sqlite3, _trash_*/, _archive_*/, Reproduce/**/*.png (figures/panels kept).
- Reproduce/ pass: removed duplicate/superseded figures (raw-named copies, cascadeA/B/ab variants, figures/cascade/,
  non-NYX scaling pdfs, all .png previews, 2x3/3x2 cascade plots), plot_cascade_nyx_1x2.py, timing_table_updated.tex,
  smoke.json, stray .pid; archived NYX_before_vs_now_psnr_fft.pdf, plot_cascade_nyx_{2x3,3x2}.py, enhanced_aux_section.tex,
  run_temp_iso300.sh, run_miranda_cr500_rep2.sh. TODO before release: re-execute REPRODUCE.ipynb with the
  PAPER_ENHANCED_MIXED pin (Table 2 / Fig 10-11 now use it).
- base_script slimmed (2026-09-10): `siren_fft_backbone_model.py` 554 -> ~95 lines (only Micro_UNet_Backbone + UNET_Model,
  spatial arch, three-band heads; FiLM/SE/ResUNet/Depth/slab/adapters removed; legacy kwargs accepted and ignored);
  two-band split and slab2d/res3d sampling paths removed from bg_stage/bg_sampling/bg_normalize/Patch_data;
  config_io lost load_and_process_data_with_sz3/free_memory, experiment lost compute_param_budget_bytes.
  Verified: old vs new UNET_Model identical parameter counts, state_dict-compatible, identical outputs; pipeline cache hit;
  forced Magnetic band-0 training run reproduces the pinned gain. Pre-edit copies: <storage>/repo_offload/base_script_bak_2026-09-10/.
- `train_bg_only` refactor (2026-09-10, bg_stage.py 856 -> 515 lines): dropped unused warm-start args
  (init_state_dict / init_optimizer_state) and the optimizer-state export, the never-set sampling-mask /
  pixel-mask path, the positional z/y0/x0/rel_err plumbing through the DataParallel adapter, and the
  psnr-slope / loss-slope early-stop variants (only the paper's "loss improves < 2% for 2 consecutive
  epochs" rule remains, still opt-in and disabled in experiments). Epoch-end logging unified into one block.
  Kept: bf16 AMP, GPU-resident sampling, epoch/step schedule calibration, DDP stop-flag, bg_max_steps replay,
  bg_log_prefix. Verified: synthetic run with evaluator, forced Magnetic band-0 (+1.4 SZ3 / +0.6 SPERR vs
  pinned +1.51 / +0.62), pipeline cache hit. Pre-refactor copy: <storage>/repo_offload/base_script_bak_2026-09-10/bg_stage.pre_refactor.py
- End-to-end validation of the refactored code (2026-09-10, `<storage>/repo_offload/validation_2026-09-10/`):
  forced `--task nyx_b`: identical base CRs on both compressors; AdaMit gains within 0.13 dB on 8/10 points, the other two
  higher by +1.05/+0.61 dB (different Phase-1 pick, cuDNN autotune run-to-run); NeurLZ identical. Forced `--task qmcpack`
  (step-calibration path): SZ3 gains within 0.2 dB of the pin on 4/5 points; the CR≈500 SZ3 point and the CR≈210 SPERR point came out at base quality (Phase-1 lr pick; see the 2026-09-12 end-to-end rerun, which showed the same behaviour on two other SPERR points). BO notebook `nyx_miranda.ipynb`
  executed headless: same base CR, same Y-axis pick, per-config PSNRs within 0.16 dB, Phase-1 pick 9.4e-3 (was 7.6e-3;
  0.13 dB from the best full-res config). Pinned pkls restored untouched; SDRBench moved to <storage>/SDRBench (symlink).
