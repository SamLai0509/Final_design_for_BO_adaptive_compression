# Evaluation-section text updates (2026-09-01)
Source of truth: `Reproduce/results/paper_numbers.txt` (PAPER_FINAL pins, Miranda override included).
Tables 2/3/4 and Figs 10/11/13/14 in the draft are already the new ones; only the prose below changes.

## 4.3 Rate-Distortion (PSNR)
- "up to **+11.70 dB** over SZ3 and **+11.8 dB** over SZ3 + NeurL at matched CRs"
  -> "up to **+6.28 dB** over SZ3 and **+6.32 dB** over SZ3 + NeurLZ at matched CRs"
- "raises the effective CR by up to **224%** on baryon density (123 dB at CR 470 versus CR ~145 ...) and by **351%** on temperature (85 dB at CR 474 versus 104)"
  -> "raises the effective CR by up to **88%** on baryon density and by **133%** on dark-matter density
      (83.9 dB at CR 474, which SZ3 alone only reaches near CR 200)"
- "With SPERR, AdaMit gains up to **+5.4 dB** over SPERR and up to **15.1 dB** over SPERR + NeurLZ"
  -> "up to **+4.10 dB** over SPERR and up to **+13.26 dB** over SPERR + NeurLZ"
- "NuerLZ lowers SPERR's own PSNR by **5.0–6.9 dB** on the three NYX fields"
  -> "by **5.4–6.8 dB** on average on the three NYX fields"
- "Average over CR 100–500, the largest SPERR + AdaMit gain is on NYX **temperature (+3.52 dB)**"
  -> "the largest SPERR + AdaMit gain is on **QMCPack (+3.08 dB)**"

## Table-2 paragraph
- "(+**11.67** dB over SZ3 and +**5.44** dB over SPERR), and it decreases the relative L2 error by up to **73.9%** on SZ3 and up to **46.5%** on SPERR"
  -> "(+**6.09** dB over SZ3 and +**4.10** dB over SPERR), and it decreases the relative L2 error by up to **50.4%** on SZ3 and up to **37.6%** on SPERR"
- (timing sentences unchanged: 5.6x / 2.5x / 26.5x / 11.2x still correct)

## 4.3 Rate-Distortion (FFT Error)
SZ3 paragraph:
- "reduces the magnitude error by up to **59.6%** and the phase error by up to **43.6%** ..., and by up to **59.8%** and **43.6%** relative to SZ3 + NeurLZ"
  -> "by up to **48%** and **32%** ...; and by up to **46%** and **32%** relative to SZ3 + NeurLZ"
- "The largest magnitude reduction is on Miranda (up to **50.1%**), and the largest phase reduction is on NYX temperature (up to **43.5%**); the other datasets with substantial PSNR gains reduce the magnitude error by **32–42%** on average."
  -> "The largest magnitude reduction is on Miranda (up to **48%**), and the largest phase reductions are on NYX temperature and Miranda (up to **32%**); the other datasets with substantial PSNR gains reduce the magnitude error by **28–40%** on average."
SPERR paragraph:
- "removes up to **49.4%** of the magnitude error and up to **36.7%** of the phase error, and it outperforms SPERR + NeurLZ by up to **66.1%** and **49.0%**"
  -> "removes up to **53%** of the magnitude error and up to **37%** of the phase error, and it outperforms SPERR + NeurLZ by up to **61%** and **34%**"
- "The largest magnitude error reductions are on QMCPack (up to **43.9%**) and on Miranda for phase (up to **36.4%**)."
  -> "... on QMCPack (up to **53%**) and on Miranda for phase (up to **37%**)."
  (the follow-up sentence "the only two datasets whose spectral gains over SPERR exceed those over SZ3" still holds for exactly these two.)
NeurLZ paragraph:
- "On SZ3, its magnitude error is only **6.3%** below base's, and its phase error is within **0.3%** of it. On SPERR, its magnitude and phase errors are **14.4%** and **7.9%** above the base."
  -> "On SZ3, its magnitude error stays within **7%** of the base's on average (unchanged on the NYX fields) and its phase error within **2%**. On SPERR, its magnitude and phase errors rise up to **38%** and **27%** above the base (**13%** and **7%** on average)."
Miranda/QMCPack spectral-vs-spatial paragraph:
- "RMSEs of Miranda and QMCPack are reduced by at most **32.6%** and **25.1%** over SZ3, their FFT magnitude errors fall by up to **50.1%** and **40.7%**"
  -> "... by at most **31%** and **26%** over SZ3, their FFT magnitude errors fall by up to **48%** and **43%**"
- "Miranda's phase error falls by up to **36.4%** against an RMSE reduction of at most **30.3%**"
  -> "... up to **37%** against an RMSE reduction of at most **31%**"

## Power Spectrum paragraph (match Table 3)
- "a mean error of **0.094%** and a maximum error of **0.269%**, in just **6.85s** on a single GPU (and **4.33s** on 4 GPUs). In contrast, NeurLZ requires **106s** ..., demonstrating that AdaMit satisfies downstream analysis targets **25x** faster."
  -> "a mean error of **0.277%** and a maximum error of **0.570%**, after just **8.3 s** of training on a single GPU (**2.4 s** on 4 GPUs). In contrast, NeurLZ requires **55.2 s** ..., so AdaMit satisfies the downstream requirement **6.7x** faster on the same GPU and **23x** faster with 4 GPUs."

## 4.4 Multi-GPU (match Table 4 / Fig 14)
- "Four GPUs reach single-GPU quality **2.14–4.08x** faster."
  -> "**2.20–4.39x** faster."
- "End-to-end compression of the 5123 fields drop from about **11–12 s to 3.4–5.3 s**, QMCPack from about **61 s to 23–29 s**, and Miranda from about **87–95 s to 35–39 s**."
  -> "... from about **11.5–13.3 s to 4.0–5.9 s**, QMCPack from about **62–65 s to 21–26 s**, and Miranda from about **95–105 s to 50–56 s**."
- "The largest is on Miranda (**3.80x**), whose 1024^2 slices best amortize gradient synchronization, and the largest SPERR speedup is on NYX temperature (**4.08x**)."
  -> "The largest speedup is on NYX baryon density with SPERR (**4.39x**), and the largest SZ3 speedup is on NYX dark-matter density (**3.49x**)."  (drop the Miranda-amortization rationale -- Miranda now shows the smallest speedups, 2.26x/2.51x.)
- "four GPUs reach the single-GPU target of **121.67 dB** after **2.9 s** of training instead of 10 s, while ... Miranda reaches the single-GPU target of **31.7 s** instead of 80.0 s."
  -> "... target of **117.05 dB** after **2.8 s** ... Miranda reaches the single-GPU target (**49.31 dB**) after **31.4 s** instead of 80.0 s."

## Headlines to sync elsewhere (abstract / intro bullets / conclusion / Sec.1)
- "improves the effective compression ratio by up to **351%**" -> "up to **133%**" (NYX dark-matter density, SZ3)
- "reduces the FFT error by up to **67%**" (abstract) / "**66%**" (conclusion) -> "up to **61%**" (vs NeurLZ, SPERR baryon magnitude)
- "reaches the scientific requirement ... ~**25x** faster" -> "~**23x** faster" (55.2 s vs 2.4 s on 4 GPUs; 6.7x on one GPU)
- "Multi-GPU training further accelerates online learning by up to **4.1x**" -> "up to **4.4x**"
- Sec 4.4 intro sentence "Four GPUs reach single-GPU quality 2.14–4.08x" is covered above.

# ============ FINAL (2026-09-02): cascade protocol numbers ============
# Convention: baryon = cascade A (decoded last), temperature = cascade B (decoded last),
# DMD = shared stage 1; Miranda/QMC/Magnetic unchanged. Pins: PAPER_ENHANCED_MIXED_CACHES.json.

## Rate-Distortion (PSNR)
- "up to +6.28 dB over SZ3 and +6.32 dB over SZ3 + NeurLZ" -> "up to **+7.66** dB over SZ3 and **+7.68** dB over SZ3 + NeurLZ"
- "by up to 88% on baryon density (123 dB at CR 470 versus CR ~145 ...) and by 133% on temperature (83.9 dB ...)"
  -> "by up to **126%** on baryon density and by **133%** on dark-matter density (83.9 dB at CR 474 versus ~205)"
- "up to +4.1 dB over SPERR and up to 13.26 dB over SPERR + NeurLZ" -> "up to **+4.38** dB over SPERR and up to **+13.05** dB over SPERR + NeurLZ"
- "NeurLZ lowers SPERR's own PSNR by 5.4-6.8 dB" -> "by **4.8-6.8 dB** on average"
- "largest SPERR + AdaMit gain is on QMCPack (+3.08 dB)" -> "on **NYX baryon density (+3.28 dB)**"

## Table-2 paragraph
- "(+6.09 dB over SZ3 and +4.10 dB over SPERR) ... relative L2 error by up to 50.4% on SZ3 and up to 37.6% on SPERR"
  -> "(+**7.49** dB over SZ3 and +**4.38** dB over SPERR) ... by up to **58.0%** on SZ3 and up to **39.5%** on SPERR"
- Table rows: baryon +7.49 / 13.1->5.5 | +4.38 / 16.2->9.8 ; temperature +3.64 / 55.8->36.7 | +2.42 / 60.6->45.9 ; DMD unchanged.

## FFT Error
SZ3: "reduces the magnitude error by up to 48% and the phase error by up to 32% ..., and by up to 46% and 32% relative to SZ3 + NeurLZ"
  -> "by up to **48%** and **34%** ..., and by up to **49%** and **34%** relative to SZ3 + NeurLZ"
- "The largest magnitude reduction is on Miranda (up to 48%), and the largest phase reduction is on NYX temperature (up to 32%); the other datasets ... 28-40% on average."
  -> "The largest magnitude reductions are on Miranda and baryon density (up to **48%**), and the largest phase reduction is on baryon density (up to **34%**); the other datasets with substantial PSNR gains reduce the magnitude error by **28-44%** on average."
SPERR: "removes up to 53% ... up to 37% ... outperforms SPERR + NeurLZ by up to 61% and 34%"
  -> "... by up to **62%** and 34%"  (largest-reduction sentence: QMCPack 53% / Miranda phase 37% unchanged; the "only two datasets" sentence still holds)
NeurLZ paragraph: SZ3 side unchanged ("within 7% ... within 2%"); SPERR side "rise up to 38% and 27%" -> "rise up to **63%** and **27%** above the base (13% and 7% on average)"
Miranda/QMCPack spectral-vs-spatial paragraph: 31%/26%, 48%/43%, 37% vs 31% (as previously listed).

## Headlines (abstract / intro bullet / conclusion)
- 133% unchanged; "61%" -> "**62%**"; "~23x" unchanged; "4.4x" unchanged.
