"""Frequency-domain (Fourier) loss terms for the background (BG) residual network.

These losses supervise the network in the 2-D spatial-frequency domain, complementing
the pixel-space loss. Matching the target's Fourier content encourages the
reconstruction to reproduce both the energy distribution across frequency bands
(magnitude) and the spatial structure/edges (phase), and lets us emphasise either the
low- or high-frequency band.

All transforms use the real 2-D FFT (`rfft2`) with orthonormal normalisation. The FFT
section is computed in float32 because `torch.fft.*` does not support bf16/fp16 on some
backends; results are cast back to the caller's dtype so they can be summed with an
AMP (autocast) spatial loss.

Optional optimisation (not used here for simplicity): the Hann window and the radial
band-weight maps depend only on (H, W, boost), so they could be precomputed and cached
across calls instead of being rebuilt every forward pass.
"""

import torch
import torch.nn.functional as F


def fft_mag_phase_loss_bg_t(
    pred,
    tgt,
    focus="low",
    boost=1.0,
    mag_weight=1.0,
    phase_weight=1.0,
    eps=1e-6,
):
    """Windowed magnitude + phase loss between predicted and target BG patches.

    Steps:
      1. Apply a separable 2-D Hann window (outer product of two 1-D Hann windows) to
         suppress spectral leakage caused by the implicit periodic boundary of the FFT.
      2. Take the orthonormal real 2-D FFT of both patches.
      3. Magnitude term: MSE between log-magnitudes, weighted by a radial band weight.
      4. Phase term: squared *wrapped* phase difference, weighted by the band weight and
         gated by the target magnitude (phase is meaningless where there is little
         energy, so low-magnitude frequencies are down-weighted).

    Args:
        pred, tgt: (B, C, H, W) predicted and target patches.
        focus: "low" up-weights low frequencies, "high" up-weights high frequencies,
            anything else gives uniform weighting.
        boost: strength of the low/high emphasis (0 = uniform).
        mag_weight, phase_weight: relative weights of the two terms.
        eps: numerical floor for logs / divisions.

    Returns:
        (loss_fft, loss_mag, loss_phase), each cast back to ``pred.dtype``.
    """
    batch, channels, height, width = pred.shape
    device = pred.device
    out_dtype = pred.dtype

    # --- Separable 2-D Hann window (float32) ---------------------------------------
    # Tapering the patch to ~0 at the borders avoids the high-frequency artefacts that
    # the FFT would otherwise see from the discontinuity between opposite edges.
    wy = torch.hann_window(height, periodic=False, device=device, dtype=torch.float32)
    wx = torch.hann_window(width, periodic=False, device=device, dtype=torch.float32)
    window = torch.outer(wy, wx).view(1, 1, height, width)

    # --- Window + forward FFT (float32 for backend compatibility) ------------------
    pred_w = pred.float() * window
    tgt_w = tgt.float() * window
    pred_f = torch.fft.rfft2(pred_w, norm="ortho")
    tgt_f = torch.fft.rfft2(tgt_w, norm="ortho")

    # Log-magnitude compresses the large dynamic range of the spectrum so the loss is
    # not dominated by a few very-high-energy (low-frequency) coefficients.
    pred_mag = torch.abs(pred_f)
    tgt_mag = torch.abs(tgt_f)
    pred_logmag = torch.log1p(pred_mag + eps)
    tgt_logmag = torch.log1p(tgt_mag + eps)

    # --- Radial band weight on the rfft2 grid (H x (W//2+1)) -----------------------
    # rr is the normalised radial frequency (distance from DC), in [0, 1].
    fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=torch.float32)
    fx = torch.fft.rfftfreq(width, d=1.0, device=device, dtype=torch.float32)
    rr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    rr = rr / (rr.max() + 1e-8)
    focus = str(focus).lower()
    if focus == "low":
        freq_weight = 1.0 + float(boost) * (1.0 - rr)   # weight ~ (1+boost) at DC -> 1 at Nyquist
    elif focus == "high":
        freq_weight = 1.0 + float(boost) * rr           # weight ~ 1 at DC -> (1+boost) at Nyquist
    else:
        freq_weight = torch.ones_like(rr)
    freq_weight = freq_weight.view(1, 1, height, width // 2 + 1)

    # --- Magnitude loss ------------------------------------------------------------
    loss_mag = (((pred_logmag - tgt_logmag) ** 2) * freq_weight).mean()

    # --- Phase loss ----------------------------------------------------------------
    if float(phase_weight) > 0.0:
        pred_phase = torch.angle(pred_f)
        tgt_phase = torch.angle(tgt_f)
        # Wrap the phase difference into (-pi, pi] via atan2(sin, cos) so that e.g.
        # +pi and -pi are treated as identical rather than maximally different.
        phase_diff = torch.atan2(
            torch.sin(pred_phase - tgt_phase),
            torch.cos(pred_phase - tgt_phase),
        )
        # Magnitude gate: normalise target magnitude by its per-patch mean and clamp to
        # [0, 1]. Frequencies with little target energy (where phase is ill-defined)
        # contribute almost nothing to the phase loss.
        gate = tgt_mag / (tgt_mag.mean(dim=(-2, -1), keepdim=True) + eps)
        gate = gate.clamp(0.0, 1.0)
        phase_weight_map = freq_weight * gate
        # Weighted mean (normalised by the total weight) of the squared wrapped phase error.
        loss_phase = ((phase_diff ** 2) * phase_weight_map).sum() / (phase_weight_map.sum() + eps)
    else:
        loss_phase = torch.zeros((), device=device, dtype=torch.float32)

    loss_fft = float(mag_weight) * loss_mag + float(phase_weight) * loss_phase

    # Cast back to the autocast dtype so this term can be added to the spatial loss.
    return (
        loss_fft.to(out_dtype),
        loss_mag.to(out_dtype),
        loss_phase.to(out_dtype),
    )


def fft_mag_l1_t(pred, tgt):
    """L1 distance between the FFT magnitudes of ``pred`` and ``tgt`` (no windowing)."""
    pred_f = torch.fft.rfft2(pred.float(), dim=(-2, -1), norm="ortho")
    tgt_f = torch.fft.rfft2(tgt.float(), dim=(-2, -1), norm="ortho")
    return F.l1_loss(torch.abs(pred_f), torch.abs(tgt_f))


def fft_phase_l1_t(pred, tgt):
    """L1 distance between FFT phases, using the wrapped phase difference.

    The atan2(sin, cos) form maps the angular error into (-pi, pi] so phase wrap-around
    (e.g. +pi vs -pi) is not counted as a large error.
    """
    pred_f = torch.fft.rfft2(pred.float(), dim=(-2, -1), norm="ortho")
    tgt_f = torch.fft.rfft2(tgt.float(), dim=(-2, -1), norm="ortho")

    pred_phase = torch.angle(pred_f)
    tgt_phase = torch.angle(tgt_f)
    phase_diff = torch.atan2(
        torch.sin(pred_phase - tgt_phase),
        torch.cos(pred_phase - tgt_phase),
    )
    return torch.mean(torch.abs(phase_diff))


def masked_fft_mag_l1_t(pred, tgt, mask=None):
    """``fft_mag_l1_t`` restricted to a spatial region (both inputs multiplied by mask)."""
    if mask is not None:
        pred = pred * mask
        tgt = tgt * mask
    return fft_mag_l1_t(pred, tgt)


def masked_fft_phase_l1_t(pred, tgt, mask=None):
    """``fft_phase_l1_t`` restricted to a spatial region (both inputs multiplied by mask)."""
    if mask is not None:
        pred = pred * mask
        tgt = tgt * mask
    return fft_phase_l1_t(pred, tgt)
