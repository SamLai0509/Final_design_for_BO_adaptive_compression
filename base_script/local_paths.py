"""Resolve machine-specific paths (compressors, data, caches).

Lookup order for every key: environment variable  >  ``<repo>/local_paths.env``
(a gitignored KEY=VALUE file, see ``local_paths.env.example``)  >  a ``/path/to/...``
placeholder. Import as ``from local_paths import P`` and call ``P("ADAMIT_SZ3_LIB")``.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "ADAMIT_SZ3_LIB":       "/path/to/SZ3/build/lib64/libSZ3c.so",
    "ADAMIT_PYSZ":          "/path/to/SZ3/tools/pysz",
    "ADAMIT_SPERR_BIN":     "/path/to/SPERR/build/bin/sperr3d",
    "ADAMIT_NYX_DIR":       "/path/to/data/NYX/",                      # six NYX .f32 files
    "ADAMIT_MIRANDA_FILE":  "/path/to/data/miranda_1024x1024x1024_float32.raw",
    "ADAMIT_MAGNETIC_FILE": "/path/to/data/magnetic_reconnection_512x512x512_float32.raw",
    "ADAMIT_QMC_FILE":      "/path/to/data/SDRBENCH-QMCPack/288x115x69x69/einspline_288_115_69_69.pre.f32",
    "ADAMIT_CACHE_DIR":     os.path.join(REPO_ROOT, "sec_4_evaluation", "sperr_fft_cache"),
    "ADAMIT_VIZ_DIR":       os.path.join(REPO_ROOT, "Reproduce", "viz"),
    "ADAMIT_CASCADE_DIR":   os.path.join(REPO_ROOT, "Reproduce", "cascade"),
    "ADAMIT_CKPT_DIR":      os.path.join(REPO_ROOT, "ablation_ckpts"),
    "ADAMIT_IO_BENCH_TMP":  "/tmp/adamit_io_bench",
}

def _load_env_file():
    cfg = {}
    f = os.path.join(REPO_ROOT, "local_paths.env")
    if os.path.isfile(f):
        for line in open(f):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = os.path.expanduser(v.strip().strip('"').strip("'"))
    return cfg

_FILE_CFG = _load_env_file()

def P(key, default=None):
    """Path for ``key`` (env var > local_paths.env > DEFAULTS > ``default``)."""
    v = os.environ.get(key) or _FILE_CFG.get(key) or DEFAULTS.get(key) or default
    if v is None:
        raise KeyError(f"unknown path key {key!r}")
    if key.endswith("_DIR") and key != "ADAMIT_IO_BENCH_TMP":
        v = v.rstrip("/") + "/" if key == "ADAMIT_NYX_DIR" else v   # callers do NYX_DIR + "field.f32"
    return v
