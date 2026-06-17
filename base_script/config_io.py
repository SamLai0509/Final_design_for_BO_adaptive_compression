import gc
import os
import random
import sys

import numpy as np
import torch


def ensure_pysz_path(pysz_path):
    if pysz_path and pysz_path not in sys.path:
        sys.path.append(pysz_path)


def set_seed(seed_value=17):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)


def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_and_process_data_with_sz3(
    gt_path,
    aux_paths,
    sz_lib_path,
    pysz_path,
    rel_err,
    data_shape,
    sz_bin_path=None,
):
    ensure_pysz_path(pysz_path)
    from pysz import SZ

    dtype = np.float32
    sz = SZ(sz_lib_path)

    print(f"[*] Loading target field: {os.path.basename(gt_path)}")
    gt_target = np.fromfile(gt_path, dtype=dtype).reshape(data_shape)

    print(f"[*] Running SZ3 compression (REL = {rel_err})...")
    sz_bytes, _ = sz.compress(gt_target, 1, 0, rel_err, 0)
    if sz_bin_path is not None:
        with open(sz_bin_path, "wb") as f:
            f.write(sz_bytes)
        print("Saved SZ bitstream to:", sz_bin_path)
    lq_target = sz.decompress(sz_bytes, data_shape, dtype)

    aux_data = []
    for idx, path in enumerate(aux_paths):
        print(f"[*] Loading auxiliary field {idx + 1}/{len(aux_paths)}: {os.path.basename(path)}")
        aux_data.append(np.fromfile(path, dtype=dtype).reshape(data_shape))

    print("[Info] Running...")
    gt_data = [gt_target] + aux_data
    lq_data = [lq_target] + aux_data
    return gt_data, lq_data


def _error_bounded_post_process(
    x_enhanced,
    x_prime,
    absolute_error_bound,
    relative_error_bound=0.0,
    verbose=False,
    a=1.0,
):
    if relative_error_bound > 0:
        data_range = float(np.max(x_prime) - np.min(x_prime))
        effective_bound = relative_error_bound * data_range
    else:
        effective_bound = absolute_error_bound

    upper = x_prime + a * effective_bound
    lower = x_prime - a * effective_bound
    d_prime = np.maximum(np.minimum(x_enhanced, upper), lower)

    if verbose:
        max_delta = np.max(np.abs(d_prime - x_enhanced))
        print("  Post-processing applied (full volume):")
        print(f"    a: {a}")
        print(f"    Effective bound: {effective_bound:.3e}")
        print(f"    Max delta: {max_delta:.3e}")

    return d_prime


def free_memory(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_multifield_from_disk(
    gt_path,
    aux_paths,
    sz_bin_path,
    data_shape,
    dtype=np.float32,
    pysz_path=None,
    sz_lib_path=None,
    rel_err_create_if_missing=None,
):
    ensure_pysz_path(pysz_path)
    from pysz import SZ

    if sz_lib_path is None:
        sz_lib_path = os.environ.get("SZ3_LIB_PATH", None)
    sz = SZ(sz_lib_path) if sz_lib_path else SZ()

    if not os.path.isfile(sz_bin_path):
        if rel_err_create_if_missing is None:
            raise FileNotFoundError(
                f"SZ bitstream not found: {sz_bin_path!r}. "
                "Either place a precomputed .sz next to the .f32, or pass "
                "`rel_err_create_if_missing=<float>` to compress the target once from disk."
            )
        print(
            f"[load_multifield_from_disk] Missing {sz_bin_path!r}; "
            f"compressing GT with REL={rel_err_create_if_missing:g} ..."
        )
        gt_full = np.fromfile(gt_path, dtype=dtype).reshape(data_shape)
        sz_bytes, cr = sz.compress(gt_full, 1, 0, rel_err_create_if_missing, 0)
        del gt_full
        parent = os.path.dirname(os.path.abspath(sz_bin_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(sz_bin_path, "wb") as f:
            f.write(sz_bytes)
        print(f"[load_multifield_from_disk] Wrote bitstream (CR≈{cr:.4f}).")

    gt_target = np.memmap(gt_path, dtype=dtype, mode="r", shape=data_shape)
    aux_data = [np.memmap(path, dtype=dtype, mode="r", shape=data_shape) for path in aux_paths]

    with open(sz_bin_path, "rb") as f:
        sz_bytes = np.frombuffer(f.read(), dtype=np.uint8)

    # NOTE: pysz loads SZ3 shared library via ctypes. If the library is not on the
    # system dynamic loader path, we must pass an explicit path here.
    # You can supply it via `sz_lib_path` or the env var `SZ3_LIB_PATH`.
    lq_target = sz.decompress(sz_bytes, data_shape, dtype)

    gt_data = [gt_target] + aux_data
    lq_data = [lq_target] + aux_data
    return gt_data, lq_data
