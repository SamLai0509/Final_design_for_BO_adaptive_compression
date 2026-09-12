"""Auxiliary (sibling) field loading for the NYX runs.

A NYX target field is compressed with the other five fields supplied as extra
input channels. Those siblings must be *decoder-reproducible*: at decode time
only the archived, lossily compressed siblings exist, so training on the
originals and inferring on decompressed ones is a protocol violation that
collapses the gain (NYX temperature: +0.00 dB). This module supplies the
siblings the paper protocol calls for.

Three modes, in order of increasing fidelity:

  originals  -- read <field>.f32 directly. NOT decoder-reproducible; kept only
                so an old result can be re-derived.
  cr_matched -- decompress the archived stream whose CR level is closest in
                log-CR to the target's own CR (levels 100..600). This is the
                paper protocol: the SAME decompressed volume feeds the Phase-1
                proxy, training, inference and the NeurLZ baseline.
  enhanced   -- a sibling that an earlier cascade stage already corrected is
                fed in corrected. Falls back to cr_matched per field, so a
                partially populated export directory is fine.

The cascade order that the hand-off measured as best is
dark_matter_density -> temperature -> baryon_density: stage 1 is always DMD
because it is the field least sensitive to sibling quality, and baryon density
runs last because it gains the most (+6.4 dB mean, +1.7 over cr_matched).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]                # repository root
sys.path.append(str(ROOT / "base_script"))
from local_paths import P  # noqa: E402
# the matched-CR archive written by `SPERR_fft.py --task aux_prep` (Step 5)
AUX_STREAM_DIR = Path(P("ADAMIT_CACHE_DIR")) / "aux_streams"
NYX_DIR = Path(P("ADAMIT_NYX_DIR"))

NYX_ALL = ["baryon_density", "dark_matter_density", "temperature",
           "velocity_x", "velocity_y", "velocity_z"]

# Cascade A, the paper's final chain: a field is decoded with whatever enhanced
# siblings the stages before it produced. Stage 1 is dark matter density (the
# field least sensitive to sibling quality); temperature reads DMD enhanced;
# baryon density runs last and reads both. Cascade B (DMD -> baryon ->
# temperature) was measured too: temperature gains +0.15..0.4 dB but baryon
# density loses 0.6..0.9 dB, so A is kept.
CASCADE_ORDER = ["dark_matter_density", "temperature", "baryon_density"]

_cache: dict[tuple, np.ndarray] = {}


def available_levels(field: str, codec: str = "sz3") -> list[int]:
    """CR levels actually present in the archive for this field.

    Not every field has every level -- temperature stops at 500 -- so the
    nearest-level search has to be over what exists, not over the nominal grid.
    """
    out = []
    for p in AUX_STREAM_DIR.glob(f"{codec}_{field}_f32_cr*.bin"):
        try:
            out.append(int(p.stem.rsplit("cr", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def closest_level(target_cr: float, levels) -> int:
    """The archived level nearest the target CR in log space."""
    lv = list(levels)
    if not lv:
        raise FileNotFoundError("no archived sibling levels")
    return min(lv, key=lambda L: abs(np.log(L) - np.log(max(target_cr, 1e-9))))


def _enhanced_path(enhanced_dir, field: str, level: int, codec: str):
    """Locate an earlier stage's export for this field.

    Exports are named NYX_<field>_<codec>_cr<L>; the level an earlier stage ran
    at need not equal the one we want, so take the closest available.
    """
    d = Path(enhanced_dir)
    cands = sorted(d.glob(f"NYX_{field}_{codec}_cr*"))
    cands = [c for c in cands if (c / "ours.f32").is_file()]
    if not cands:
        return None
    def lv(p):
        try:
            return int(p.name.rsplit("cr", 1)[1])
        except (IndexError, ValueError):
            return 10 ** 9
    return min(cands, key=lambda c: abs(np.log(max(lv(c), 1)) - np.log(level))) / "ours.f32"


def load_sibling(field, target_cr, shape, sz=None, codec="sz3",
                 mode="cr_matched", enhanced_dir=None, sperr_decode=None,
                 log=print):
    """One sibling volume, as the decoder would see it.

    Siblings are archived by the same base compressor as the pipeline, so a
    SZ3 run reads sz3_* streams and a SPERR run reads sperr_* ones. `sz` is a
    pysz engine (SZ3 streams); `sperr_decode(bin_path, shape) -> ndarray`
    decodes SPERR streams, since sperr3d is a CLI over files.
    """
    key = (field, round(float(target_cr), 3), tuple(shape), codec, mode,
           str(enhanced_dir or ""))
    if key in _cache:
        return _cache[key]

    if mode == "originals":
        vol = np.fromfile(NYX_DIR / f"{field}.f32", dtype=np.float32).reshape(shape)
        log(f"[aux] {field}: ORIGINAL (not decoder-reproducible)")
        _cache[key] = vol
        return vol

    levels = available_levels(field, codec)
    level = closest_level(target_cr, levels)

    if mode == "enhanced" and enhanced_dir:
        p = _enhanced_path(enhanced_dir, field, level, codec)
        if p is not None:
            vol = np.fromfile(p, dtype=np.float32).reshape(shape)
            log(f"[aux] {field}: ENHANCED {p.parent.name}")
            _cache[key] = vol
            return vol

    stem = AUX_STREAM_DIR / f"{codec}_{field}_f32_cr{level}"
    if codec == "sperr":
        if sperr_decode is None:
            raise ValueError("sperr siblings need a sperr_decode callback")
        vol = sperr_decode(str(stem.with_suffix(".bin")), tuple(shape))
    else:
        if sz is None:
            raise ValueError("sz3 siblings need a pysz SZ engine")
        bits = np.fromfile(stem.with_suffix(".bin"), dtype=np.uint8)
        vol = sz.decompress(bits, tuple(shape), np.float32)
    meta = {}
    if stem.with_suffix(".json").is_file():
        meta = json.loads(stem.with_suffix(".json").read_text())
    log(f"[aux] {field}: cr{level} (actual CR {meta.get('cr', float('nan')):.1f}) "
        f"for target CR {target_cr:.1f}")
    _cache[key] = vol
    return vol


def load_siblings(target_field, target_cr, shape, sz=None, codec="sz3",
                  mode="cr_matched", enhanced_dir=None, sperr_decode=None,
                  fields=None, log=print):
    """Every sibling of `target_field`, in NYX_ALL order."""
    names = [f for f in (fields or NYX_ALL) if f != target_field]
    return [load_sibling(f, target_cr, shape, sz=sz, codec=codec, mode=mode,
                         enhanced_dir=enhanced_dir, sperr_decode=sperr_decode,
                         log=log) for f in names], names


def export_enhanced(out_dir, field, target_cr, rec, codec="sz3", levels=None,
                    log=print):
    """Write a corrected volume where a later cascade stage will find it."""
    level = closest_level(target_cr, levels or available_levels(field, codec)
                          or [100, 200, 300, 400, 500, 600])
    d = Path(out_dir) / f"NYX_{field}_{codec}_cr{level}"
    d.mkdir(parents=True, exist_ok=True)
    np.asarray(rec, dtype=np.float32).tofile(d / "ours.f32")
    (d / "meta.json").write_text(json.dumps(
        dict(field=field, codec=codec, level=level, target_cr=float(target_cr),
             shape=list(np.shape(rec))), indent=1))
    log(f"[aux] exported enhanced {field} -> {d}/ours.f32")
    return d / "ours.f32"
