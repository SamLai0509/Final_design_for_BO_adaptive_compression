"""Training configuration for the BG residual pipeline.

Only ``TrainConfig`` is consumed by the base scripts (via
``experiment.build_bg_only_cfg``). The full training / proposal machinery lives in
the research repo and is not needed for the released base pipeline.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 100
    steps_per_epoch: int = 30
    bg_patch_size: int = 64
    roi_patch: int = 32
    bg_batch: int = 64
    roi_batch: int = 64
    lr: float = 1e-3
    res_mean: float = 0.0
    res_std: float = 1.0
    abs_err: float = 1.0
    input_means: list = None
    input_stds: list = None
    seed: int = 42
    bg_h: int = 4
    roi_h: int = 4
    max_train_time: float = 300.0
    roi_backprop_to_bg: bool = False
    roi_warmup_fraction: float = 0.35
    roi_switch_gain_window: int = 6
    roi_switch_gain_threshold: float = 1e-3
    roi_refresh_every: int = 10
    roi_refresh_min_frac: float = 0.10
