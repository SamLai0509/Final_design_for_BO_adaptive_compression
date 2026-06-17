# Patch_data.py
import numpy as np
from typing import List, Optional, Union

Array = np.ndarray

# ============================================================
# Multi-field synchronous patch / slab samplers for BG training.
# Every field is cropped at the same (z, y, x) location per sample so the
# model input and the ground-truth target stay spatially aligned.
# ============================================================


def sample_bg_patches_multifield(
    Xs: List[Array],    # ground-truth fields:  [target_X,  aux1_X,  aux2_X,  ...]
    Xps: List[Array],   # decompressed fields:  [target_Xp, aux1_Xp, aux2_Xp, ...]
    n: int,
    patch: int = 64,
    seed: Optional[int] = None,
    mask: Optional[Array] = None,
    min_mask_frac: float = 0.0,
    max_retries: int = 20,
) -> dict:
    """Sample ``n`` random 2D patches synchronously across all physical fields.

    Every field is cropped at the same (z, y, x) location per sample, so the model
    input and the ground-truth target stay physically aligned.

    Returns a dict with:
        xp: [n, n_fields, patch, patch]  -- model input (decompressed fields)
        x : [n, 1, patch, patch]         -- target (Xs[0]; the residual ground truth)
        z, y0, x0: [n]                   -- chosen patch coordinates
        pixel_mask: [n, 1, patch, patch] -- only when ``mask`` is supplied
    """
    if len(Xps) < 1:
        raise ValueError("Xps must be non-empty")
    if Xs[0].shape != Xps[0].shape:
        raise ValueError(f"Xs[0] shape {Xs[0].shape} != Xps[0] shape {Xps[0].shape}")
    n_fields = len(Xps)
    D, H, W = Xs[0].shape
    rng = np.random.default_rng(seed)

    y_max = max(H - patch + 1, 1)
    x_max = max(W - patch + 1, 1)

    use_mask = mask is not None and float(min_mask_frac) > 0.0
    if use_mask:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (D, H, W):
            raise ValueError(
                f"sample_bg_patches_multifield mask shape mismatch: got {mask.shape}, expected {(D, H, W)}"
            )
        min_mask_frac = float(np.clip(min_mask_frac, 0.0, 1.0))
        max_retries = max(1, int(max_retries))

    z = np.zeros((n,), dtype=np.int64)
    y0 = np.zeros((n,), dtype=np.int64)
    x0 = np.zeros((n,), dtype=np.int64)
    xp = np.zeros((n, n_fields, patch, patch), dtype=np.float32)
    x_target = np.zeros((n, 1, patch, patch), dtype=np.float32)
    pixel_mask_out = np.zeros((n, 1, patch, patch), dtype=np.float32) if use_mask else None

    for i in range(n):
        zi = int(rng.integers(0, D))
        yi = int(rng.integers(0, y_max))
        xi = int(rng.integers(0, x_max))

        # With a mask, reject-sample until the patch covers at least `min_mask_frac`
        # of masked voxels; otherwise keep the first random coordinate.
        if use_mask:
            for _ in range(max_retries):
                zt = int(rng.integers(0, D))
                yt = int(rng.integers(0, y_max))
                xt = int(rng.integers(0, x_max))
                if float(np.mean(mask[zt, yt:yt + patch, xt:xt + patch])) >= min_mask_frac:
                    zi, yi, xi = zt, yt, xt
                    break

        z[i], y0[i], x0[i] = zi, yi, xi
        for f in range(n_fields):
            xp[i, f] = Xps[f][zi, yi:yi + patch, xi:xi + patch]
        x_target[i, 0] = Xs[0][zi, yi:yi + patch, xi:xi + patch]
        if use_mask:
            pixel_mask_out[i, 0] = mask[zi, yi:yi + patch, xi:xi + patch]

    # Scientific volumes often contain NaNs; zero them so they do not poison training.
    xp = np.nan_to_num(xp, nan=0.0)
    x_target = np.nan_to_num(x_target, nan=0.0)

    out_dict = {"xp": xp, "x": x_target, "z": z, "y0": y0, "x0": x0}
    if use_mask:
        out_dict["pixel_mask"] = pixel_mask_out
    return out_dict


def sample_bg_slices_at_indices_multifield(
    Xs: List[Array],
    Xps: List[Array],
    z_indices: Union[Array, List[int]],
    patch: int,
    y0: Optional[Union[Array, List[int]]] = None,
    x0: Optional[Union[Array, List[int]]] = None,
    mask: Optional[Array] = None,
) -> dict:
    """
    Deterministic BG sampling: train on explicit z-slice indices (full or partial xy patch).
    xp: [n, n_fields, patch, patch], x: [n, 1, patch, patch]
    """
    if len(Xps) < 1:
        raise ValueError("Xps must be non-empty")
    if Xs[0].shape != Xps[0].shape:
        raise ValueError(f"Xs[0] shape {Xs[0].shape} != Xps[0] shape {Xps[0].shape}")
    n_fields = len(Xps)
    D, H, W = Xs[0].shape
    z = np.asarray(z_indices, dtype=np.int64).reshape(-1)
    n = int(z.size)
    if n <= 0:
        raise ValueError("z_indices must be non-empty")

    y_max = max(H - patch + 1, 1)
    x_max = max(W - patch + 1, 1)

    if y0 is None:
        y0_arr = np.zeros((n,), dtype=np.int64)
    else:
        y0_arr = np.asarray(y0, dtype=np.int64).reshape(-1)
    if x0 is None:
        x0_arr = np.zeros((n,), dtype=np.int64)
    else:
        x0_arr = np.asarray(x0, dtype=np.int64).reshape(-1)

    if y0_arr.size != n or x0_arr.size != n:
        raise ValueError("y0/x0 length must match z_indices")

    xp = np.zeros((n, n_fields, patch, patch), dtype=np.float32)
    x_target = np.zeros((n, 1, patch, patch), dtype=np.float32)
    pixel_mask_out = None
    if mask is not None:
        pixel_mask_out = np.zeros((n, 1, patch, patch), dtype=np.float32)

    for i in range(n):
        zi = int(z[i]) % D
        yi = int(y0_arr[i]) % y_max
        xi = int(x0_arr[i]) % x_max
        z[i] = zi
        y0_arr[i] = yi
        x0_arr[i] = xi
        for f in range(n_fields):
            xp[i, f] = Xps[f][zi, yi : yi + patch, xi : xi + patch]
        x_target[i, 0] = Xs[0][zi, yi : yi + patch, xi : xi + patch]
        if mask is not None:
            pixel_mask_out[i, 0] = mask[zi, yi : yi + patch, xi : xi + patch]

    xp = np.nan_to_num(xp, nan=0.0)
    x_target = np.nan_to_num(x_target, nan=0.0)
    
    out_dict = {"xp": xp, "x": x_target, "z": z, "y0": y0_arr, "x0": x0_arr}
    if mask is not None:
        out_dict["pixel_mask"] = pixel_mask_out
    return out_dict


def sample_bg_center_slabs_multifield(
    Xs: List[Array],
    Xps: List[Array],
    n: int,
    K: int = 7,
    patch: int = 64,
    seed: Optional[int] = None,
) -> dict:
    """
    Sample 2.5D slabs for BG training.
    Input:
        xp: [n, n_fields * K, patch, patch]
    Target:
        x : [n, 1, patch, patch]   # center-slice residual only
    """
    n_fields = len(Xps)
    D, H, W = Xps[0].shape
    rng = np.random.default_rng(seed)

    half = K // 2
    if D < K:
        return {
            "xp": np.zeros((0, n_fields * K, patch, patch), dtype=np.float32),
            "x": np.zeros((0, 1, patch, patch), dtype=np.float32),
            "z0": np.zeros((0,), dtype=np.int64),
            "zc": np.zeros((0,), dtype=np.int64),
            "y0": np.zeros((0,), dtype=np.int64),
            "x0": np.zeros((0,), dtype=np.int64),
        }

    zc = rng.integers(half, D - half, size=n, dtype=np.int64)
    z0 = zc - half
    y0 = rng.integers(0, max(H - patch + 1, 1), size=n, dtype=np.int64)
    x0 = rng.integers(0, max(W - patch + 1, 1), size=n, dtype=np.int64)

    xp = np.zeros((n, n_fields * K, patch, patch), dtype=np.float32)
    x_target = np.zeros((n, 1, patch, patch), dtype=np.float32)

    for i in range(n):
        zi0, zic, yi, xi = int(z0[i]), int(zc[i]), int(y0[i]), int(x0[i])

        for f in range(n_fields):
            slab = Xps[f][zi0:zi0 + K, yi:yi + patch, xi:xi + patch]
            xp[i, f * K:(f + 1) * K] = slab

        x_target[i, 0] = Xs[0][zic, yi:yi + patch, xi:xi + patch]

    xp = np.nan_to_num(xp, nan=0.0)
    x_target = np.nan_to_num(x_target, nan=0.0)

    return {
        "xp": xp,
        "x": x_target,
        "z0": z0,
        "zc": zc,
        "y0": y0,
        "x0": x0,
    }


def sample_bg_center_slab_at_z_multifield(
    Xs: List[Array],
    Xps: List[Array],
    z_center: int,
    K: int = 7,
    patch: int = 64,
    y0: int = 0,
    x0: int = 0,
    n: int = 1,
) -> dict:
    """Deterministic 2.5D slab centered at z_center; xp [n, n_fields*K, patch, patch]."""
    n_fields = len(Xps)
    D, H, W = Xps[0].shape
    half = int(K) // 2
    if D < K:
        raise ValueError(f"depth {D} < slab K={K}")

    zc = int(np.clip(z_center, half, D - half - 1))
    zi0 = zc - half
    yi = int(y0) % max(H - patch + 1, 1)
    xi = int(x0) % max(W - patch + 1, 1)

    xp = np.zeros((n, n_fields * K, patch, patch), dtype=np.float32)
    x_target = np.zeros((n, 1, patch, patch), dtype=np.float32)
    zc_arr = np.full((n,), zc, dtype=np.int64)
    y0_arr = np.full((n,), yi, dtype=np.int64)
    x0_arr = np.full((n,), xi, dtype=np.int64)

    for i in range(n):
        for f in range(n_fields):
            xp[i, f * K:(f + 1) * K] = Xps[f][zi0:zi0 + K, yi:yi + patch, xi:xi + patch]
        x_target[i, 0] = Xs[0][zc, yi:yi + patch, xi:xi + patch]

    xp = np.nan_to_num(xp, nan=0.0)
    x_target = np.nan_to_num(x_target, nan=0.0)
    return {
        "xp": xp,
        "x": x_target,
        "z": zc_arr,
        "zc": zc_arr,
        "z0": np.full((n,), zi0, dtype=np.int64),
        "y0": y0_arr,
        "x0": x0_arr,
    }


def sample_bg_volume_slab_at_z_multifield(
    Xs: List[Array],
    Xps: List[Array],
    z_center: int,
    K: int = 7,
    patch: int = 64,
    y0: int = 0,
    x0: int = 0,
    n: int = 1,
) -> dict:
    """Deterministic 3D slab for Res3D UNet; xp [n, n_fields, K, patch, patch]."""
    slab = sample_bg_center_slab_at_z_multifield(
        Xs, Xps, z_center, K=K, patch=patch, y0=y0, x0=x0, n=n
    )
    n_fields = len(Xps)
    K = int(K)
    n = int(n)
    patch = int(patch)
    xp2d = slab["xp"]
    xp3d = np.zeros((n, n_fields, K, patch, patch), dtype=np.float32)
    for f in range(n_fields):
        xp3d[:, f] = xp2d[:, f * K:(f + 1) * K]
    slab["xp"] = xp3d
    return slab
