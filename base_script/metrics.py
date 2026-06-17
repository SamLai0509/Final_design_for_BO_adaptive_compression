import numpy as np


def compute_psnr(x_true, x_hat):
    mse = float(np.mean((x_true - x_hat) ** 2))
    data_range = float(np.max(x_true) - np.min(x_true))
    if mse <= 0.0:
        return 999.0
    if data_range <= 0.0:
        data_range = 1.0
    return 20.0 * np.log10(data_range / np.sqrt(mse))


def compute_masked_psnr(x_true, x_hat, mask):
    if mask is None or np.count_nonzero(mask) == 0:
        return np.nan
    diff = x_true[mask] - x_hat[mask]
    mse = float(np.mean(diff ** 2))
    data_range = float(np.max(x_true) - np.min(x_true))
    if mse <= 0.0:
        return 999.0
    if data_range <= 0.0:
        data_range = 1.0
    return 20.0 * np.log10(data_range / np.sqrt(mse))


def evaluate_reconstruction(x_true, x_hat, roi_mask=None):
    abs_err = np.abs(x_true - x_hat)
    out = {
        "psnr": compute_psnr(x_true, x_hat),
        "global_mae": float(np.mean(abs_err)),
        "max_err": float(np.max(abs_err)),
    }

    if roi_mask is not None and np.count_nonzero(roi_mask) > 0:
        out["roi_mae"] = float(np.mean(abs_err[roi_mask]))
        out["nonroi_mae"] = float(np.mean(abs_err[~roi_mask]))
        out["roi_psnr"] = compute_masked_psnr(x_true, x_hat, roi_mask)
    else:
        out["roi_mae"] = np.nan
        out["nonroi_mae"] = np.nan
        out["roi_psnr"] = np.nan

    return out


def compute_roi_fft_metrics_from_hat(x_true, x_hat, roi_list_zyx, patch=32, K=7):
    if roi_list_zyx is None or len(roi_list_zyx) == 0:
        return np.nan, np.nan

    depth, height, width = x_true.shape
    mag_losses = []
    phase_losses = []

    for z0, y0, x0 in roi_list_zyx:
        z0, y0, x0 = int(z0), int(y0), int(x0)
        if z0 + K > depth or y0 + patch > height or x0 + patch > width:
            continue

        true_patch = x_true[z0:z0 + K, y0:y0 + patch, x0:x0 + patch]
        hat_patch = x_hat[z0:z0 + K, y0:y0 + patch, x0:x0 + patch]

        for k in range(K):
            ft_true = np.fft.rfft2(true_patch[k], norm="ortho")
            ft_hat = np.fft.rfft2(hat_patch[k], norm="ortho")

            mag_true = np.abs(ft_true)
            mag_hat = np.abs(ft_hat)
            mag_losses.append(float(np.mean(np.abs(mag_hat - mag_true))))

            phase_true = np.angle(ft_true)
            phase_hat = np.angle(ft_hat)
            phase_diff = np.angle(np.exp(1j * (phase_hat - phase_true)))
            phase_losses.append(float(np.mean(np.abs(phase_diff))))

    if len(mag_losses) == 0:
        return np.nan, np.nan

    return float(np.mean(mag_losses)), float(np.mean(phase_losses))


def compute_roi_diagnostics(x_true, x_hat, roi_mask, roi_list_zyx, patch=32, K=7):
    metrics = evaluate_reconstruction(x_true, x_hat, roi_mask=roi_mask)
    fft_mag_err, fft_phase_err = compute_roi_fft_metrics_from_hat(
        x_true=x_true,
        x_hat=x_hat,
        roi_list_zyx=roi_list_zyx,
        patch=patch,
        K=K,
    )
    metrics["fft_mag_err"] = fft_mag_err
    metrics["fft_phase_err"] = fft_phase_err
    return metrics
