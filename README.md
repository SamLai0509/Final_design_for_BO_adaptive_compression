# AdaMit: Adaptive and Frequency-Aware Neural-Enhanced Compression for Scientific Workflows

AdaMit is an online neural error-mitigation framework for error-bounded lossy compression of
scientific data. It works on top of an existing compressor (**SZ3** or **SPERR**): at compression
time a small 2-D CNN is trained, within a fixed wall-clock budget, to predict the compressor's
residual. The compressed output is the base bitstream plus the BF16 model weights, and the decoder
adds the predicted residual to the base reconstruction, clamped to the original error bound.

```
reconstruction = base_decompress(stream) + clamp(model(base_recon, auxiliary_fields))
effective CR   = original_bytes / (base_stream_bytes + model_bytes)
```

Compared with the previous online approach (NeurLZ), AdaMit trains and stores the model online
per instance, needs no pretrained model, and

- **enhanced auxiliary fields**: sibling fields enter the model as the decoder holds them, and a
  field enhanced earlier in the cascade is fed to later fields in its enhanced form (§3.1);
- **residual-aware normalization**: a reversible per-field Z-score scheme that keeps the
  zero-centred residual distribution and is not dominated by extreme values (§3.2);
- **frequency-aware reconstruction**: a frequency-split three-head architecture trained with a
  dual-domain (spatial + spectral) loss (§3.3);
- **model scaling and BF16 storage**: the parameter budget follows the data volume, and the weights
  are stored in BF16 (§3.4);
- **adaptive two-phase training**: a TPE search on a half-resolution proxy picks the learning rate
  and slice orientation in the first 10% of the budget, then full-resolution training (§3.5);
- **high-throughput online training**: a GPU-resident slicing pipeline and multi-GPU data-parallel
  training (§3.1, §4.4).

On six fields from four simulations (NYX, Miranda, Magnetic Reconnection, QMCPack) and two
compressors, AdaMit reduces the storage footprint by up to 57% at matched PSNR under the same
training budget, reduces the FFT error by up to 62%, and meets the NYX power-spectrum requirement
6.7× faster on one GPU and 23× faster on four.

## Branches

This `main` branch only holds this overview. The code lives on two branches:

| Branch | Use it for | Contents |
|---|---|---|
| **`single-gpu`** | everything that runs on one GPU | The complete pipeline (`sec_4_evaluation/SPERR_fft.py`), the §3 ablations, one folder per paper section, pinned results behind Table 2 and the rate-distortion figures, and a step-by-step README (environment, data, compressors, path configuration, per-section ablation commands). |
| **`multi-gpu`** | the four-GPU experiments and the power-spectrum study | A superset of `single-gpu`: the same code plus `sec_4_evaluation/multi_gpu/` (one vs. four A100s, §4.4, Table 4) and `sec_4_evaluation/power_spectrum/` (Gimlet power-spectrum compliance, §4.3, Table 3), with the run JSON behind every table row. Needs a node with four GPUs for the multi-GPU rows; the power-spectrum study runs on one GPU. |

Start with `single-gpu` unless you need the multi-GPU or power-spectrum results:

```bash
git clone -b single-gpu <this repository> AdaMit && cd AdaMit    # or -b multi-gpu
cat README.md                                                    # step-by-step setup and reproduction
```

Both branches share the same layout (`base_script/` core library, `sec_3_*` ablations,
`sec_4_evaluation/` evaluation, `Reproduce/` whole-chain scripts and the results report) and the
same configuration mechanism: machine-specific paths (SZ3, SPERR, datasets) go into a gitignored
`local_paths.env`, never into the code.

## Requirements in brief

Linux, CUDA, Python ≥ 3.10 with PyTorch, NumPy, Optuna, MONAI (for the NeurLZ baseline) and
matplotlib; SZ3 with its `pysz` wrapper and the SPERR `sperr3d` binary built from source; the public
SDRBench NYX, Miranda and Magnetic Reconnection volumes and the QMCPack einspline table. The
branch READMEs give the exact steps.

## License

MIT, see `LICENSE`.
