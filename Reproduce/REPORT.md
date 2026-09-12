# AdaMit — Final Rerun Report (2026-08-29, updated through 2026-09-10)

All runs were executed serially on one RTX PRO 6000 (nothing in parallel). Results live in `Reproduce/`,
experiments in `Reproduce/experiment/`. The original Chinese version of this report is archived at
`/storage/sam/repo_offload/REPORT_zh_2026-09-11.md`.

> Naming note. Sections 4, 7, 8 and 10 use the *internal* cascade labels from the run scripts
> (**A = DMD → baryon → temperature**, **B = DMD → temperature → baryon**, pins `CASCADE_A/B_CACHES.json`).
> The paper adopted a single cascade and calls it **cascade A = DMD → temperature → baryon density**,
> i.e. the internal *B*. Table 2 / Fig. 10 / Fig. 11 / Figs. 1, 3, 12 of the paper use that single cascade
> (pin `SPERR/sperr_fft_cache/PAPER_ENHANCED_MIXED_CACHES.json`).

## 0. One-paragraph summary

- Under the paper-final configuration (lr window [1e-3, 1e-2], shuffled slice sampling, CR-matched
  decompressed siblings, 10% Phase-1, no gates), **59 of the 60 operating points** across the six datasets are
  healthy; the only +0.00 was Miranda–SPERR CR≈500 (Phase-1 picked lr 6.7e-3 twice, see §3), later
  replaced by the median of three standalone repeats (§8.1).
- Six-dataset mean gain: **SZ3 +3.10 dB, SPERR +2.27 dB** (v3 draft: +4.44 / +2.65); Table-2 (CR≈500) mean
  +3.35 / +2.20. The whole drop comes from the three NYX fields moving from lossless to matched-CR lossy
  siblings; the other three datasets agree with v3 within noise.
- NYX experiments: siblings matter most for baryon density, less for temperature, hardly at all for DMD; the
  **cascade (decode DMD first, then feed the enhanced fields as auxiliaries) adds another +1.7 dB on baryon and
  +0.7 dB on temperature**; dropping the velocity siblings is slightly worse.

## 1. Final configuration and timeline

| Item | Value | Where |
|---|---|---|
| lr search window | [1e-3, 1e-2] (paper §3.5, unchanged) | `SPERR_fft.py` `BO_LR_MIN/MAX` |
| Slice sampling | `shuffled`: every depth slice exactly once per epoch, random order (= NeurLZ's own loop) | `bg_sampling.py` `_shuffled_z` |
| Sibling protocol | same base compressor, decompressed at the CR level in {100…600} closest to the target's CR; identical copies for training and inference | `SPERR_fft.py` `AUX_MODE='cr_matched'`; streams in `sperr_fft_cache/aux_streams/` |
| Phase 1 | 10% of the budget, 10 TPE trials, plain argmax, no gates | same |
| Cosine schedule | epoch-level re-planning + step-level re-planning (only QMCPack triggers it) | `bg_stage.py` `bg_sched_step_calibrate` |
| QMCPack | 35k parameters (bg_h=23) | — |

Timeline (08-28 22:53 → 08-29 08:22): aux archive check → QMCPack → NYX b/t/d → Magnetic → Miranda (OOM-killed
at 02:18 because a no-aux experiment was started on the 5090 at the same time; rerun alone from 03:11, done 05:21)
→ both halves of Fig. 8 → collect → viz exports → 5 cascade stages → 3 no-velocity fields → Fig. 6 NYX rerun →
bf16/fp32 rerun. Stage stamps: `Reproduce/logs/run_all.log`, `run_after_chain.log`, `run_figs_after.log`.

Note: `collect.py` originally pinned "the newest pkl after RUN_START", which mistook the overnight no-aux NYX pkls
for paper results; it now pins by the chain's start/exit stamps and was re-run (12:06). The six pkls in
`Reproduce/results/PAPER_FINAL_CACHES.json` are the correct ones.

## 2. Paper numbers (`Reproduce/results/paper_numbers.txt`, `paper_numbers_vs_v3.txt`)

### 2.1 Table 2 (the CR≈500 point), decompressed-sibling protocol

| Dataset | Base | CR ours | PSNR base / +NeurLZ / **+Ours** | Gain vs base | vs NeurLZ | Rel L2 base → ours (reduction) |
|---|---|---|---|---|---|---|
| NYX baryon | SZ3 | 470 | 111.34 / 111.30 / **117.43** | **+6.09** | +6.13 | 1.31e-2 → 6.49e-3 (−50.4%) |
| NYX baryon | SPERR | 476 | 109.46 / 100.31 / **113.56** | **+4.10** | +13.26 | 1.62e-2 → 1.01e-2 (−37.6%) |
| NYX temperature | SZ3 | 474 | 73.64 / 73.73 / **76.07** | **+2.43** | +2.34 | 5.58e-2 → 4.22e-2 (−24.4%) |
| NYX temperature | SPERR | 474 | 72.91 / 67.29 / **75.09** | **+2.17** | +7.80 | 6.06e-2 → 4.72e-2 (−22.1%) |
| NYX DMD | SZ3 | 474 | 79.37 / 79.29 / **83.86** | **+4.49** | +4.57 | 1.74e-1 → 1.04e-1 (−40.4%) |
| NYX DMD | SPERR | 468 | 78.50 / 69.61 / **80.63** | **+2.14** | +11.02 | 1.92e-1 → 1.50e-1 (−21.8%) |
| Miranda | SZ3 | 474 | 42.51 / 42.65 / **44.90** | **+2.39** | +2.25 | 7.95e-3 → 6.04e-3 (−24.1%) |
| Miranda | SPERR | 473 | 46.81 / 46.65 / **46.81** | **+0.00** | +0.16 | 4.84e-3 → 4.84e-3 (0%) |
| QMCPack | SZ3 | 474 | 62.26 / 64.26 / **64.80** | **+2.53** | +0.54 | 3.39e-2 → 2.53e-2 (−25.3%) |
| QMCPack | SPERR | 473 | 65.10 / 66.05 / **68.61** | **+3.51** | +2.56 | 2.45e-2 → 1.64e-2 (−33.2%) |
| Magnetic | SZ3 | 475 | 48.61 / 49.67 / **50.77** | **+2.16** | +1.10 | 1.72e-1 → 1.34e-1 (−22.0%) |
| Magnetic | SPERR | 475 | 52.88 / 52.42 / **54.19** | **+1.31** | +1.77 | 1.05e-1 → 9.03e-2 (−14.0%) |

(The paper's final Table 2 replaces the NYX baryon and temperature rows with the cascade values, §8.2/§10:
baryon +7.49 / +4.38, temperature +3.11 / +2.34; Miranda–SPERR uses the §8.1 override, +2.61.)

### 2.2 v3 draft → final (up-to gain | mean gain | Table-2 gain | iso-PSNR CR gain up-to | mean FFT-magnitude reduction)

| Dataset | Base | up-to | mean | Table-2 | iso-PSNR CR | FFT-mag |
|---|---|---|---|---|---|---|
| NYX baryon | SZ3 | +11.67 → **+6.28** | +7.79 → **+4.69** | +11.67 → **+6.09** | +224% → **+88%** | 42% → 31% |
| NYX baryon | SPERR | +4.84 → +4.10 | +2.98 → +2.94 | +4.84 → +4.10 | +60% → +48% | 22% → 26% |
| NYX temperature | SZ3 | +11.46 → **+3.36** | +7.15 → **+2.79** | +11.46 → **+2.43** | +351% → **+44%** | 40% → 30% |
| NYX temperature | SPERR | +5.44 → +2.28 | +3.52 → +1.99 | +5.44 → +2.17 | +136% → +40% | 27% → 19% |
| NYX DMD | SZ3 | +5.27 → +4.49 | +4.46 → +3.83 | +5.27 → +4.49 | +166% → +133% | 32% → 28% |
| NYX DMD | SPERR | +3.20 → +2.35 | +2.62 → +2.17 | +3.20 → +2.14 | +89% → +52% | 23% → 20% |
| Miranda | SZ3 | +3.43 → +3.20 | +2.93 → +2.80 | +2.39 → +2.39 | +45% → +42% | 45% → 44% |
| Miranda | SPERR | +3.14 → +3.20 | +2.95 → +2.41 | +2.64 → **+0.00** | +36% → +36% | 37% → 31% |
| QMCPack | SZ3 | +2.51 → +2.58 | +2.36 → +2.52 | +2.28 → +2.53 | +16% → +17% | 39% → 40% |
| QMCPack | SPERR | +3.14 → +3.51 | +2.85 → +3.08 | +3.14 → +3.51 | +23% → +26% | 44% → 45% |
| Magnetic | SZ3 | +2.21 → +2.18 | +1.94 → +1.94 | +2.21 → +2.16 | +39% → +38% | 18% → 18% |
| Magnetic | SPERR | +1.40 → +1.31 | +0.99 → +1.01 | +1.40 → +1.31 | +23% → +21% | 6% → 6% |

Six-dataset summary: SZ3 mean **+3.10 dB** (max +6.28, baryon), NeurLZ mean +0.52; SPERR mean **+2.27 dB**
(max +4.10), NeurLZ mean −3.07. Table-2 means +3.35 / +2.20.

### 2.3 Other quantities quoted in the paper (decompressed protocol)

- iso-PSNR CR gain, up-to: baryon SZ3 +88% (mean +61%), temperature SZ3 +44%, DMD SZ3 +133%, Miranda +42%,
  QMCPack +17% (SPERR +26%), Magnetic +38%.
- FFT-magnitude reduction vs base (up-to / mean): baryon 41/31%, temperature 33/30%, DMD 31/28%, Miranda 48/44%,
  QMCPack 43/40% (SPERR 53/45%), Magnetic 23/18%. Mean phase reduction: NYX 21–30%, Miranda 26%, QMCPack ≈0,
  Magnetic 2%.
- RMSE vs FFT-magnitude reduction: "spectral > spatial" still holds on Miranda (48% vs 31%) and QMCPack
  (43% vs 26%; SPERR 53% vs 33%); on the NYX fields the two are close (baryon 41% vs 51%).
- NeurLZ's own FFT-magnitude change: SZ3 side NYX ≈0, Miranda +7%, QMCPack +28%; SPERR side NYX −17 to −38% (worse).

## 3. Figures (`Reproduce/figures/`)

| Figure | File | Notes |
|---|---|---|
| Fig. 8 BO example | `Fig8_bo_phase1_phase2.pdf` | NYX half (SZ3, CR-matched aux): pick Y@7.6e-3 → 118.32 dB, best 118.33 (gap 0.01); Miranda half (SPERR): pick X@3.9e-3 → 60.66, best 60.68 (gap 0.02). Mirrored to `BO_Adaptive/NYX_SZ3_Miranda_SPERR_1x4.pdf`; legend/height compacted 09-03 |
| Fig. 9 PSNR vs CR | `Fig9_psnr_vs_cr.pdf` (decompressed) / `Fig9_psnr_vs_cr_enhanced_mixed.pdf` (paper: NYX rows from the cascade) | mirrored to `SPERR/overleaf_figures/` |
| Fig. 10 FFT vs CR | `Fig10_fft_error_vs_cr.pdf` / `Fig10_fft_error_vs_cr_enhanced_mixed.pdf` | same |
| Fig. 6 iso-epoch (NYX panels rerun) | `figures/scaling/psnr_vs_cr_isoepoch_nyx_baryon_temp.pdf` | 30k-parameter mean gain: baryon +4.34, temperature +2.35 (3k: +3.09 / +1.22); the capacity effect stays monotonic (+0.2–0.4 dB per step). Data `isoepoch_nyx_crmatched.json` |
| bf16 vs fp32 | `bf16_vs_fp32_1x2.pdf` | NYX: base 117.34, fp32 121.49, bf16 121.34 (Δ +0.16 dB); Miranda: 52.38 / 55.39 / 55.40 (Δ −0.01). The old figure's NYX 124.89 / 125.22 were lossless-aux numbers |
| NYX before/after | archived (`_archive_2026-09-10/Reproduce/figures/NYX_before_vs_now_psnr_fft.pdf`) | v3 vs final, 3 fields × (PSNR, FFT-mag, FFT-phase) |

**Points to keep in mind**

1. **Miranda–SPERR CR≈500 = +0.00** in the chain: Phase 1 picked lr 6.7e-3 both times, Phase 2 never beat the
   baseline and the guard returned the base reconstruction. This is genuine behaviour of the paper configuration
   (the only collapse in 120 points); §8.1 replaces the point by the median of standalone repeats.
2. The Fig. 8 NYX ladder sits ~5 dB below the lossless-aux version (118 vs 123 dB); its structure is unchanged.
3. `Reproduce/REPRODUCE.ipynb` was re-executed against the final pin without errors (TODO: re-execute against
   `PAPER_ENHANCED_MIXED_CACHES.json`, which the paper now uses for the NYX rows).

## 4. NYX sibling experiments (`Reproduce/experiment/summary.md`, `nyx_sibling_experiments.pdf`)

SZ3 side, 5 operating points / mean (SPERR side in summary.md):

| Experiment | Baryon | Temperature | DMD |
|---|---|---|---|
| No siblings (5090, core-limited; indicative) | +1.6/+1.5/+2.1/+2.4/+2.7 (2.04) | +1.6/+1.6/+1.8/+1.7/+1.1 (1.53) | +2.7/+3.1/+3.3/+3.4/+3.5 (3.20) |
| **CR-matched siblings (paper protocol)** | +2.3/+3.0/+5.9/+6.3/+6.1 (4.69) | +2.6/+2.5/+3.4/+3.0/+2.4 (2.79) | +2.9/+3.7/+4.3/+3.8/+4.5 (3.83) |
| Without velocity siblings | +2.0/+3.7/+5.6/+5.8/+5.9 (4.59) | +2.2/+2.0/+2.6/+2.6/+1.9 (2.25) | +3.2/+3.3/+3.9/+4.1/+4.0 (3.69) |
| Cascade A (internal): DMD → baryon → temperature | +2.3/+3.0/+6.2/+6.5/+6.4 (4.86) | +2.0/+3.2/+4.4/+4.4/+3.6 (**3.50**) | +3.1/+3.2/+3.9/+4.5/+4.4 (3.81) |
| Cascade B (internal): DMD → temperature → baryon | +4.3/+5.3/+7.7/+7.4/+7.5 (**6.43**) | +2.6/+2.6/+3.9/+3.6/+3.1 (3.16) | same as A (shared stage 1) |
| Lossless siblings (v3 draft) | +4.2/+6.0/+7.7/+9.4/+11.7 (7.79) | +2.9/+5.6/+7.2/+8.6/+11.5 (7.15) | +3.4/+4.1/+4.6/+4.9/+5.3 (4.46) |

Reading:
- Value of siblings: largest for baryon (none → matched +2.7 dB), medium for temperature (+1.3), small for DMD (+0.6)
  — hence the cascade starts with DMD.
- **The cascade works**: baryon fed with enhanced DMD + enhanced temperature (internal B) gains another **+1.7 dB**
  over the paper protocol (mean 6.43, +7.5 at CR 500), +0.3 on SPERR; temperature fed with enhanced DMD + enhanced
  baryon (internal A) gains **+0.7 dB**. The last field in the chain benefits most. The logs record
  "ENHANCED by an earlier stage" for every band (16 times each in A_nyx_t / B_nyx_b), confirming stages 2/3 saw
  enhanced siblings.
- Dropping velocity: baryon/DMD unchanged (−0.1), temperature −0.5 → the velocity siblings carry signal; keep them.
- More than half of the lossless-sibling +12 dB comes from leaked information and is not decoder-reproducible.

The cascade costs no storage (each field's model is already charged to its own CR); the decoder simply
decodes → enhances in order. Enhanced volumes: `/storage/sam/Final_visualization/cascade_2026-08-29/` (26 GB).

## 5. Miscellaneous

- **ParaView volumes**: `/storage/sam/Final_visualization/final_2026-08-29/` — `NYX_{baryon_density,temperature}_{sz3,sperr}_cr*/`
  with `base/ours/neurlz.{f32,vtk}`, `err_*.f32`, `meta.json`. Later iso-CR exports for Figs. 1/12: `NYX_baryon_density_sz3_cr503/`
  (SZ3 503.4 / NeurLZ 500.2), `enhanced/NYX_baryon_density_sz3_cr530/ours` (cascade, 500.6× / 118.7 dB),
  `NYX_temperature_sz3_cr501/` (SZ3 501.5 / NeurLZ 498.3), `enhanced_b/NYX_temperature_sz3_cr531/ours` (stage-2 enhanced, 501.1× / 76.28 dB).
- **NeurLZ code check**: it trains and infers on decompressed auxiliaries with `rel_list = [rel]*n` (same rel for
  every field); the paper's "1E-5 for DMD" in its Fig. 8 is illustrative. Our protocol matches "same decompressed
  siblings for train and inference" and differs only in matching by CR level instead of rel.
- Reproduction: `Reproduce/run_all.sh` (six datasets + Fig. 8) → `collect.py`; experiments `run_cascade_6000.sh`,
  `run_novel_6000.sh`, `experiment/collect_experiments.py`; figures `Model_parameter_Scaling/isoepoch_nyx_rerun.py`,
  `bf_16_vs_32/bf16_rerun.py`.

## 6. Paper edits implied by the rerun (checklist)

1. §3.1 sampling wording: sequential → "each epoch visits every depth slice once in a random order".
2. §4.1 sibling protocol: siblings enter the model only as the decoder holds them — archived by the same base
   compressor at the CR level (100–600, step 100) closest to the target's; the same decompressed siblings are used
   for training and inference (NeurLZ decompresses at the same rel and does not charge aux storage).
3. Table 2 numbers from §2.1 (later the cascade rows); six-dataset means +4.44/+2.65 → **+3.10/+2.27**.
4. §4.2/§4.3 iso-PSNR CR gains, FFT/RMSE reductions and the NeurLZ spectral sentences per §2.3.
5. Fig. 6 / bf16 / Fig. 8 / Fig. 9 / Fig. 10 from `Reproduce/figures/`; bf16 text: NYX 121.49 / 121.34 (Δ 0.16 dB).
6. Abstract/conclusion headlines synchronised; guard sentence for the Miranda–SPERR point.
7. Cascade: new methodology paragraph + NYX rows from the cascade (see §8.2, §10 and the paper naming note above).

## 7. Addendum (08-29 afternoon): Miranda CR≈500 repeats and two experiment figures

### 7.1 Miranda CR≈500 (`Reproduce/experiment/miranda_cr500/`)

The same operating point (SZ3 band 4 = CR 500; SPERR bisected to the same CR) run five times under the paper
configuration (2 in the chain + 3 standalone; standalone rep2 first hit a driver-level `cudaErrorLaunchTimeout`
and was rerun):

| Run | SZ3 pick (axis, lr) | lr spread | SZ3 gain | SPERR pick (axis, lr) | lr spread | SPERR gain |
|---|---|---|---|---|---|---|
| paper chain (08-28 16:28, shuffled) | X, 2.2e-3 | 0.15 dB | +2.40 | X, 6.7e-3 | 0.54 dB | **+0.00** |
| paper chain (08-29 03:11 rerun) | X, 2.7e-3 | 0.30 dB | +2.40 | X, 6.7e-3 | 4.88 dB | **+0.00** |
| standalone rep1 | X, 2.2e-3 | 0.52 dB | +2.40 | X, 1.8e-3 | 0.21 dB | +2.60 |
| standalone rep2 | Y, 2.7e-3 | 0.23 dB | +2.50 | X, 6.7e-3 | 0.27 dB | +1.30 |
| standalone rep3 | X, 2.0e-3 | 0.21 dB | +2.40 | X, 2.2e-3 | 0.14 dB | +2.60 |

- SZ3 side: 5/5 healthy, +2.4–2.5 dB, fully stable.
- SPERR side: 3/5 healthy. When the proxy picks lr ≈ 2e-3 the gain is +2.60 (twice); with 6.7e-3 it is +1.30 once
  and +0.00 twice. Phase 1 has almost no lr resolution here (lr spread 0.14–0.54 dB), so the argmax is a coin flip
  between 2e-3 and 6.7e-3, and 6.7e-3 is a marginal lr within the 72 s budget.
- Figure: `miranda_cr500_repeats.pdf` (lr vs gain; hollow markers = collapsed).

Write-up options: (a) keep the chain's +0.00 and explain the guard; (b) report the median of the repeats and say so;
(c) leave as is. Option (b) was adopted (§8.1).

### 7.2 With vs without velocity siblings

`Reproduce/experiment/nyx_with_vs_without_velocity.pdf`: baryon / DMD curves coincide (≤0.3 dB); temperature loses
0.4–0.7 dB on SZ3 without velocity and its FFT error rises slightly → small positive contribution, keep them.

### 7.3 Cascade orders

`Reproduce/experiment/nyx_cascade_orders.pdf`: in the baryon row internal cascade B (DMD → temperature → baryon) lies
above the paper protocol along the whole curve (CR 100: 131.1 vs 129.0; CR 500: 118.8 vs 117.4) with lower FFT
magnitude/phase; in the temperature row internal cascade A is ~1 dB higher; the DMD row (stage 1) coincides with the
paper protocol; SPERR-side curves nearly coincide. The field decoded last benefits most; A and B are mutually exclusive.

## 8. Addendum (08-29 afternoon): Miranda point update and cascade versions of Figs. 9/10

### 8.1 Miranda CR≈500 override (`Reproduce/results/PIN_OVERRIDES.json`)

Following §7.1, band 4 (CR≈500, both compressors) of the chain's Miranda pkl is replaced by **standalone rep3**
(the median SPERR gain of the three standalone repeats, +2.61 dB; SZ3 +2.35 vs the chain's +2.39), producing
`Miranda__e8f59c75dabf_band4rep3.pkl`; `collect.py` picks it up through `PIN_OVERRIDES.json` and regenerated
Figs. 9/10, the number tables and the notebook.

- Miranda SPERR: Table-2 +0.00 → **+2.61**, mean +2.41 → **+2.94**, mean FFT-mag reduction 31% → 37%.
- Six-dataset means: SZ3 +3.10 (unchanged), **SPERR +2.27 → +2.36**; Table-2 SPERR mean +2.20 → **+2.63**.
- Paper sentence: "the Miranda–SPERR CR≈500 point is the median of three standalone runs, because the plain-argmax
  proxy picks a non-converging lr there in some runs."

### 8.2 Cascade (enhanced-sibling) versions of Fig. 9 / Fig. 10

NYX rows replaced by cascade results (other rows = final pin). Pins: `Reproduce/results/CASCADE_{A,B}_CACHES.json`.
The paper's final figures (`Fig9/Fig10_*_enhanced_mixed.pdf`) take both baryon (stage 3) and temperature (stage 2)
from the *single* cascade DMD → temperature → baryon (internal B); the NeurLZ series come from the decompressed-aux
runs so that "NeurLZ uses decompressed auxiliaries only" holds. Baryon row at CR≈500: SZ3 +7.49 dB, FFT-mag ≈ −50%;
temperature (stage 2) +3.11 dB.

## 9. Addendum (08-29 17:35): normalization ablation rerun (`Normalization/norm_cr_time_temperature.pdf`)

`Normalization/norm_rerun.py` executes the notebook's own BasicUNet trainer (NeurLZ-style: (4,)×6, lr 1e-2,
batch 10, 100 epochs, pure MSE); the only change is that the 5 siblings are the CR-matched decompressed copies
(CR 73 → level 100 … CR 560 → level 600, the 600 level built on the fly). Data: `norm_rerun_results.json`.

| rel | CR | SZ3 | Z-score | Min-Max | old figure (lossless siblings) Z-score / Min-Max |
|---|---|---|---|---|---|
| 1e-4 | 73 | 88.5 | 90.1 | 88.4 | ~94 / ~88.5 |
| 2e-4 | 128 | 83.4 | 85.9 | 84.6 | ~90.5 / ~85 |
| 3e-4 | 193 | 80.6 | 82.8 | 82.5 | ~87.5 / ~83 |
| 4e-4 | 240 | 78.5 | 81.3 | 80.6 | ~86.5 / ~83 |
| 5e-4 | 296 | 76.9 | 79.4 | 79.0 | ~85.5 / ~79.5 |
| 6e-4 | 369 | 75.8 | 77.9 | 77.4 | ~84.5 / ~79.5 |
| 1e-3 | 559 | 72.0 | 74.5 | 73.6 | ~82 / ~77 |

Same conclusion, smaller margin: Z-score beats Min-Max at every point (+0.4 to +1.7 dB; the old figure showed
+3 to +6). In the time panel Min-Max still sits at the base level for the first ~55 s while Z-score rises within 5 s
(this slow start is caused by the normalization itself, not by the siblings).

### 9b. Same ablation on baryon density (08-29 18:35, `Normalization/norm_cr_time_baryon_density.pdf`)

`NORM_TARGET=baryon_density NORM_RELS=1e-6,…,1.4e-5 python norm_rerun.py` (siblings = CR-matched decompressed
temperature/DMD/velocity; data `norm_rerun_results_baryon_density.json`).

| rel | CR | SZ3 | Z-score | Min-Max | Z − MM |
|---|---|---|---|---|---|
| 1.0e-06 | 82 | 129.13 | 131.81 | 129.15 | +2.67 |
| 1.4e-06 | 103 | 126.68 | 130.19 | 128.41 | +1.78 |
| 2.4e-06 | 149 | 122.73 | 126.17 | 122.76 | +3.41 |
| 4.0e-06 | 230 | 118.97 | 123.80 | 120.16 | +3.64 |
| 6.9e-06 | 331 | 115.02 | 120.32 | 116.33 | +4.00 |
| 1.0e-05 | 440 | 112.40 | 117.91 | 116.18 | +1.73 |
| 1.4e-05 | 568 | 110.13 | 115.51 | 110.25 | +5.26 |

The effect is much stronger on baryon: Min-Max is essentially at the SZ3 base at 4 of 7 points (+0.0…+0.1 dB, it does
not learn), Z-score is consistently +2.7…+5.4 dB; in the time panel (CR 331) Min-Max hugs the base line for ~100 s,
Z-score gains +3 dB within 5 s. Baryon density is heavy-tailed (range 1.16e5, median 0.5): after Min-Max squeezes the
volume into [0,1] the residual is ~1e-6 and the MSE gradient vanishes, whereas Z-score keeps a learnable scale.
This panel is the one used in the paper (Fig. 4).

## 10. Aux-quality ladder (08-31, figure for the new methodology paragraph)

`Reproduce/run_auxq_ablation.sh` (RTX 6000, paper protocol) added original-aux and no-aux runs for NYX baryon and
temperature; enhanced = cascade pins, decompressed = PAPER_FINAL pin. Figure `Reproduce/figures/nyx_aux_quality_ladder.pdf`
(final version: single cascade DMD → temperature → baryon; temperature panel = stage 2, baryon panel = stage 3;
temperature left, baryon right), pins `results/AUXQ_LADDER_PINS.json`. SZ3 gains over the 5 bands vs SZ3 base:

- baryon: no-aux +1.75…+2.80 | decompressed +2.27…+6.09 | enhanced +4.34…+7.49 | original +3.68…+11.90
- temperature (stage 2, enhanced DMD only): no-aux +1.23…+1.94 | decompressed +2.43…+3.36 | enhanced +2.57…+3.87 | original +8.58…+12.20

The ladder is monotonic in both panels. The 5090 no-aux pkls (indicative only) are backed up in
`sperr_fft_cache/_experiments/2026-08-31_noaux_5090_backup/` (their hashes were overwritten by the RTX 6000 runs).

## 11. How long does NeurLZ need to catch up with AdaMit? (09-06/07, `Reproduce/experiment/neurlz_long/`)

`SPERR_fft.py --task neurlz_long` (new task): at the Table-2 SZ3 operating point, NeurLZ in its default configuration
(Z slicing, 3,415 parameters, matched-CR decompressed siblings) trains up to a cap (NYX/Magnetic 300 s, QMCPack 600 s,
Miranda 800 s) with a PSNR evaluation after every epoch; the target is AdaMit's PSNR at that point (`PAPER_ENHANCED_MIXED`
pin). `run_neurlz` gained an inference timer. Data `neurlz_long.json`, figure `neurlz_time_to_match.pdf` (2×3 trajectories),
table `table_neurlz_time_to_match.tex`.

| Dataset | Base | AdaMit gain (train+infer) | NeurLZ @ AdaMit budget | NeurLZ reaches half the gain | NeurLZ best within cap | NeurLZ infer |
|---|---|---|---|---|---|---|
| NYX baryon | 111.34 | +7.49 (12.2 s) | −0.28 | never | +3.64 @300 s | 1.86 s |
| NYX temperature | 73.64 | +3.11 (12.2 s) | −0.13 | 194 s | +1.96 @300 s | 1.34 s |
| NYX DMD | 79.37 | +4.49 (12.2 s) | −0.62 | 138 s | +3.61 @300 s | 1.58 s |
| Miranda | 42.51 | +2.35 (104.1 s) | +0.33 | 155 s | +1.87 @800 s | 8.80 s |
| QMCPack | 62.26 | +2.53 (71.4 s) | +1.22 | 81 s | +2.39 @600 s | 17.33 s |
| Magnetic | 48.61 | +2.16 (11.6 s) | +0.13 | 48 s | +1.94 @300 s | 0.84 s |

Conclusion: within 10–30× the budget NeurLZ reaches AdaMit's PSNR on **none** of the six datasets; recovering half of
the gain alone takes 48–194 s (baryon does not get there in 300 s). On the three NYX fields NeurLZ is still below the
base at AdaMit's 10 s budget (Min-Max origin shift, paper §3.2). Note: NeurLZ uses its default Z slicing here
(44 s/epoch on QMCPack), slower than the Table-2 run that borrowed AdaMit's axis (+2.0 dB) — say "default
configuration" when quoting. SPERR side not run at this stage (NeurLZ is below the base there anyway).

### 11b. Long-cap version (09-07, `neurlz_long_2000.json`; stop at target — never triggered)

| Dataset | AdaMit gain (T+I) | NeurLZ 100 ep gain (T+I) | NeurLZ best within cap (time / cap) | t_half |
|---|---|---|---|---|
| NYX baryon | +7.49 (12.2 s) | +0.06 (184 s) | +5.37 (1965 s / 2000) | 322 s |
| NYX temperature | +3.11 (12.2 s) | +1.71 (216 s) | +2.56 (2002 s / 2000) | 162 s |
| NYX DMD | +4.49 (12.2 s) | +3.09 (220 s) | +4.20 (1960 s / 2000) | 140 s |
| Miranda | +2.35 (104.1 s) | +1.88 (649 s) | +1.95 (1546 s / 2000) | 165 s |
| QMCPack | +2.53 (71.4 s) | see §11c | +2.48 (1567 s / 1500) | 94 s |
| Magnetic | +2.16 (11.6 s) | +1.61 (108 s) | +2.01 (1488 s / 1500) | 55 s |

The 100-epoch numbers here are read at epoch 100 of the long trajectories (the training path is identical to a fixed
100-epoch run; NeurLZ's cosine schedule has T_max = 1500 steps regardless of length). Baryon convergence: at the base
until ~200 s, then a jump from +0.5 to +4 dB between 206 and 345 s, then a plateau from ~600 s at +4.7 → +5.4 dB
(the 3,415-parameter capacity limit); +7.49 is out of reach within 2000 s. `SPERR_NLZ_EPOCHS` adds a fixed-epoch mode.

### 11c. Clean 100-epoch reruns (09-07, `neurlz_100ep_all.json`; fixed 100 epochs, evaluation every 10)

| Dataset | AdaMit gain (T+I) | NeurLZ 100 ep gain (T+I) | Time ratio | Share of AdaMit's gain |
|---|---|---|---|---|
| NYX baryon | +7.49 (12.2 s) | +2.25 (167 s) | 13.6× | 30% |
| NYX temperature | +3.11 (12.2 s) | +1.55 (180 s) | 14.7× | 50% |
| NYX DMD | +4.49 (12.2 s) | +2.96 (187 s) | 15.3× | 66% |
| Miranda | +2.35 (104 s) | +1.87 (612 s) | 5.9× | 79% |
| QMCPack | +2.53 (71 s) | +2.51 (3,960 s) | 55.5× | 99% |
| Magnetic | +2.16 (11.6 s) | +1.55 (103 s) | 8.8× | 72% |

Versus the epoch-100 readings of §11b only baryon differs materially (+0.06 → +2.25): NeurLZ sits at a lr-cycle
boundary there and run-to-run non-determinism decides whether the jump happens before or after epoch 100; the other
fields agree within ±0.2 dB. Times are ~10% lower than the trajectory version (no per-epoch evaluation).
QMCPack at 100 epochs = 3,933 s training + 27 s inference, essentially matching AdaMit (64.78 vs 64.80 dB).
Paper Table 2 uses this version (NeurLZ block).

### 11d. SPERR + NeurLZ, 100 epochs (09-07, `neurlz_100ep_sperr.json`; best checkpoint within 100 epochs)

| Dataset | NeurLZ time (s) | Overhead | Gain | Rel L2 | AdaMit gain (time) |
|---|---|---|---|---|---|
| NYX baryon | 164 | 32× | −1.07 | 16.2 → 18.3 | +4.38 (12.2 s) |
| NYX temperature | 173 | 34× | −0.50 | 60.6 → 64.2 | +2.34 (12.2 s) |
| NYX DMD | 183 | 36× | −0.84 | 192 → 212 | +2.14 (12.2 s) |
| Miranda | 643 | 15× | +1.71 | 4.8 → 3.9 | +2.61 (104 s) |
| QMCPack | 3778 | 592× | +3.06 | 24.5 → 17.2 | +3.51 (71 s) |
| Magnetic | 88 | 17× | +0.03 | 105 → 105 | +1.31 (11.6 s) |

The three NYX fields remain below the SPERR base after 100 epochs. NeurLZ has no best-checkpoint guard and a periodic
cosine lr, so the epoch-100 state often lands in a bad phase (baryon final 106.78 vs best 108.39); the table reports the
best checkpoint within 100 epochs for both compressors (hence SZ3 DMD = +3.14 / 174 → 121 in the paper table).

## 12. Code cleanup and validation (09-10)

See `Reproduce/CLEANUP_PLAN.md`. The repository was slimmed for the GitHub push (`_archive_2026-09-10/`, data on
`/storage/sam/repo_offload/`, `SDRBench` moved to `/storage/sam/SDRBench`), `SPERR_fft.py` trimmed to the six paper
datasets, and `base_script/` reduced to the paper model (spatial micro U-Net + three-band heads; `train_bg_only`
856 → 515 lines). Validation: old vs new model bit-identical; forced `--task nyx_b` reproduces the pinned base CRs and
gains within Phase-1 noise; forced `--task qmcpack` reproduces the pinned pkl **byte-for-byte**; the BO notebook
executes headless with the same axis pick and PSNRs within 0.16 dB.
