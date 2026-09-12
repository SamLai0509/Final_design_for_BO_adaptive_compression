# NYX sibling-protocol experiments

All runs: shuffled slice order, lr in [1e-3, 1e-2], 10 % Phase-1, no trust gates, 30k-param model, same NeurLZ
protocol; RTX PRO 6000 unless marked. Gains are PSNR(base + Ours) - PSNR(base) in dB; FFT columns are the
reductions of the per-slice FFT magnitude / phase L1 error vs the base. Siblings, where used, are the decoder's
copies (CR level closest to the target); 'enhanced' = the output of an earlier cascade stage.

## Baryon density

| experiment | base | CR≈100 | CR≈150 | CR≈220 | CR≈320 | CR≈500 | mean gain | mean FFT-mag red. | mean FFT-pha red. |
|---|---|---|---|---|---|---|---|---|---|
| reference (CR-matched siblings) | SZ3 | +2.27 | +2.96 | +5.88 | +6.28 | +6.09 | +4.69 | +31% | +21% |
| reference (CR-matched siblings) | SPERR | +1.57 | +2.56 | +3.26 | +3.23 | +4.10 | +2.94 | +26% | +15% |
| v3 draft (lossless siblings) | SZ3 | +4.16 | +6.03 | +7.72 | +9.38 | +11.67 | +7.79 | +42% | +20% |
| v3 draft (lossless siblings) | SPERR | +2.43 | +0.02 | +3.49 | +4.13 | +4.84 | +2.98 | +22% | +10% |
| no siblings (5090, indicative) | SZ3 | +1.58 | +1.49 | +2.07 | +2.38 | +2.66 | +2.04 | +25% | +25% |
| no siblings (5090, indicative) | SPERR | +0.92 | +1.23 | +1.52 | +1.95 | +2.30 | +1.58 | +18% | +11% |
| no velocity siblings | SZ3 | +2.03 | +3.66 | +5.57 | +5.80 | +5.87 | +4.59 | +33% | +25% |
| no velocity siblings | SPERR | +1.18 | +2.30 | +3.09 | +3.48 | +3.76 | +2.76 | +26% | +17% |
| cascade A: DMD -> baryon -> temperature (enhanced siblings) | SZ3 | +2.27 | +2.95 | +6.23 | +6.51 | +6.36 | +4.86 | +33% | +22% |
| cascade A: DMD -> baryon -> temperature (enhanced siblings) | SPERR | +1.67 | +2.57 | +3.45 | +3.78 | +4.18 | +3.13 | +28% | +17% |
| cascade B: DMD -> temperature -> baryon (enhanced siblings) | SZ3 | +4.34 | +5.29 | +7.66 | +7.38 | +7.49 | +6.43 | +42% | +30% |
| cascade B: DMD -> temperature -> baryon (enhanced siblings) | SPERR | +2.30 | +2.14 | +3.74 | +3.84 | +4.38 | +3.28 | +28% | +16% |

## Temperature

| experiment | base | CR≈100 | CR≈150 | CR≈220 | CR≈320 | CR≈500 | mean gain | mean FFT-mag red. | mean FFT-pha red. |
|---|---|---|---|---|---|---|---|---|---|
| reference (CR-matched siblings) | SZ3 | +2.63 | +2.52 | +3.36 | +3.02 | +2.43 | +2.79 | +30% | +30% |
| reference (CR-matched siblings) | SPERR | +2.07 | +1.43 | +2.02 | +2.28 | +2.17 | +1.99 | +19% | +13% |
| v3 draft (lossless siblings) | SZ3 | +2.88 | +5.62 | +7.20 | +8.60 | +11.46 | +7.15 | +40% | +24% |
| v3 draft (lossless siblings) | SPERR | +2.11 | +2.43 | +3.61 | +4.00 | +5.44 | +3.52 | +27% | +22% |
| no siblings (5090, indicative) | SZ3 | +1.55 | +1.57 | +1.76 | +1.71 | +1.09 | +1.53 | +22% | +24% |
| no siblings (5090, indicative) | SPERR | +1.08 | +0.96 | +1.30 | +1.38 | +1.16 | +1.17 | +12% | +9% |
| no velocity siblings | SZ3 | +2.20 | +1.95 | +2.57 | +2.59 | +1.93 | +2.25 | +26% | +27% |
| no velocity siblings | SPERR | +1.83 | +1.91 | +1.50 | +2.13 | +1.73 | +1.82 | +17% | +13% |
| cascade A: DMD -> baryon -> temperature (enhanced siblings) | SZ3 | +1.95 | +3.20 | +4.37 | +4.35 | +3.64 | +3.50 | +32% | +30% |
| cascade A: DMD -> baryon -> temperature (enhanced siblings) | SPERR | +1.65 | +2.46 | +2.57 | +2.51 | +2.42 | +2.32 | +21% | +14% |
| cascade B: DMD -> temperature -> baryon (enhanced siblings) | SZ3 | +2.63 | +2.57 | +3.87 | +3.61 | +3.11 | +3.16 | +31% | +30% |
| cascade B: DMD -> temperature -> baryon (enhanced siblings) | SPERR | +2.07 | +2.00 | +2.16 | +2.15 | +2.34 | +2.14 | +20% | +14% |

## Dark matter density

| experiment | base | CR≈100 | CR≈150 | CR≈220 | CR≈320 | CR≈500 | mean gain | mean FFT-mag red. | mean FFT-pha red. |
|---|---|---|---|---|---|---|---|---|---|
| reference (CR-matched siblings) | SZ3 | +2.86 | +3.71 | +4.27 | +3.80 | +4.49 | +3.83 | +28% | +26% |
| reference (CR-matched siblings) | SPERR | +2.18 | +2.11 | +2.35 | +2.09 | +2.14 | +2.17 | +20% | +19% |
| v3 draft (lossless siblings) | SZ3 | +3.37 | +4.14 | +4.59 | +4.92 | +5.27 | +4.46 | +32% | +28% |
| v3 draft (lossless siblings) | SPERR | +2.00 | +2.17 | +2.75 | +2.96 | +3.20 | +2.62 | +23% | +21% |
| no siblings (5090, indicative) | SZ3 | +2.72 | +3.05 | +3.32 | +3.41 | +3.50 | +3.20 | +26% | +25% |
| no siblings (5090, indicative) | SPERR | +1.79 | +1.94 | +1.96 | +1.96 | +1.92 | +1.91 | +19% | +19% |
| no velocity siblings | SZ3 | +3.15 | +3.30 | +3.94 | +4.06 | +3.99 | +3.69 | +28% | +26% |
| no velocity siblings | SPERR | +2.05 | +2.14 | +2.29 | +2.36 | +2.35 | +2.24 | +21% | +20% |
| cascade A: DMD -> baryon -> temperature (enhanced siblings) | SZ3 | +3.06 | +3.20 | +3.93 | +4.46 | +4.42 | +3.81 | +28% | +25% |
| cascade A: DMD -> baryon -> temperature (enhanced siblings) | SPERR | +2.16 | +1.81 | +2.19 | +2.02 | +2.44 | +2.13 | +19% | +19% |
| cascade B: DMD -> temperature -> baryon (enhanced siblings) | SZ3 | +3.06 | +3.20 | +3.93 | +4.46 | +4.42 | +3.81 | +28% | +25% |
| cascade B: DMD -> temperature -> baryon (enhanced siblings) | SPERR | +2.16 | +1.81 | +2.19 | +2.02 | +2.44 | +2.13 | +19% | +19% |
