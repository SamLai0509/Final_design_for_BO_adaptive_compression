# Miranda CR≈500: repeat analysis

Paper configuration (shuffled, lr ∈ [1e-3, 1e-2], plain argmax). Each row is an independent run of this one
operating point (SZ3 band 4; SPERR aligned to the same CR). 'gain' = PSNR(base+Ours) − PSNR(base).

| run | SZ3: pick (axis, lr) | lr spread | SZ3 gain | SPERR: pick (axis, lr) | lr spread | SPERR gain |
|---|---|---|---|---|---|---|
| paper chain (08-28 16:28, shuffled) | X, 2.2e-03 | 0.15 dB | **+2.40** | X, 6.7e-03 | 0.54 dB | **+0.00** |
| paper chain (08-29 03:11 rerun) | X, 2.7e-03 | 0.30 dB | **+2.40** | X, 6.7e-03 | 4.88 dB | **+0.00** |
| standalone rep1 | X, 2.2e-03 | 0.52 dB | **+2.40** | X, 1.8e-03 | 0.21 dB | **+2.60** |
| standalone rep2 | Y, 2.7e-03 | 0.23 dB | **+2.50** | X, 6.7e-03 | 0.27 dB | **+1.30** |
| standalone rep3 | X, 2.0e-03 | 0.21 dB | **+2.40** | X, 2.2e-03 | 0.14 dB | **+2.60** |

- SZ3: 5/5 runs healthy; healthy gains +2.40, +2.40, +2.40, +2.50, +2.40 (median +2.40); collapsed picks: none.
- SPERR: 3/5 runs healthy; healthy gains +2.60, +1.30, +2.60 (median +2.60); collapsed picks: 6.7e-03, 6.7e-03.
