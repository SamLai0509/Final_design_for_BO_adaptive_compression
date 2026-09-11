"""AdaMit residual network.

A micro 2-D U-Net backbone (three GELU conv levels, bilinear up-sampling, skip
connections) shared by four 3x3 projection heads: one full-residual head and the
three frequency-band heads (low / mid / high) supervised in ``bg_stage.py``.
With ``bg_split_mode="three"`` (the paper configuration) the predicted residual is
the sum of the three band heads; otherwise the single ``bg_out_head`` is used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Micro_UNet_Backbone(nn.Module):
    """Two-level micro U-Net: in -> down x2 -> up x2 with skips. Returns the last
    decoder feature map (``hidden`` channels) alongside its own projection."""

    def __init__(self, in_channels, out_channels, hidden=8):
        super().__init__()
        h = int(hidden)
        self.inc = nn.Conv2d(in_channels, h, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(h, h, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(h, h, kernel_size=3, stride=2, padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv1 = nn.Conv2d(h * 2, h, kernel_size=3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv2 = nn.Conv2d(h * 2, h, kernel_size=3, padding=1)
        self.outc = nn.Conv2d(h, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        nn.init.zeros_(self.outc.weight)
        if self.outc.bias is not None:
            nn.init.zeros_(self.outc.bias)

    def forward(self, x, return_feat=False):
        x1 = self.act(self.inc(x))
        x2 = self.act(self.down1(x1))
        x3 = self.act(self.down2(x2))
        u1 = self.up1(x3)
        if u1.shape[-2:] != x2.shape[-2:]:      # odd spatial sizes: exact-size upsample
            u1 = F.interpolate(u1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.act(self.up_conv1(torch.cat([u1, x2], dim=1)))
        u2 = self.up2(u1)
        if u2.shape[-2:] != x1.shape[-2:]:
            u2 = F.interpolate(u2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.act(self.up_conv2(torch.cat([u2, x1], dim=1)))
        out = self.outc(u2)
        return (out, u2) if return_feat else out


class UNET_Model(nn.Module):
    """Backbone + heads. ``n_fields`` input channels (decompressed target + auxiliaries),
    ``bg_hidden`` channels everywhere. Legacy keyword arguments from older configs
    (K, D, H, W, bg_arch, bg_use_se, adapters, ...) are accepted and ignored; only the
    ``spatial`` architecture and the ``three``-band split remain."""

    def __init__(self, n_fields=1, bg_hidden=4, bg_split_bands=True, bg_split_mode="three", **legacy_kwargs):
        super().__init__()
        arch = str(legacy_kwargs.get("bg_arch", "spatial")).lower()
        if arch != "spatial":
            raise ValueError(f"Unsupported bg_arch {arch!r}: only 'spatial' is supported")
        if bg_split_bands and bg_split_mode not in (None, "three"):
            raise ValueError(f"Unsupported bg_split_mode {bg_split_mode!r}: only 'three' is supported")
        self.n_fields = int(n_fields)
        self.bg_arch = "spatial"
        self.bg_split_bands = bool(bg_split_bands)
        self.bg_split_mode = "three" if self.bg_split_bands else None

        h = int(bg_hidden)
        self.bg_net = Micro_UNet_Backbone(in_channels=self.n_fields, out_channels=h, hidden=h)
        self.bg_out_head = nn.Conv2d(h, 1, kernel_size=3, padding=1)
        self.bg_low_head = nn.Conv2d(h, 1, kernel_size=3, padding=1)
        self.bg_mid_head = nn.Conv2d(h, 1, kernel_size=3, padding=1)
        self.bg_high_head = nn.Conv2d(h, 1, kernel_size=3, padding=1)
        for head in (self.bg_out_head, self.bg_low_head, self.bg_mid_head, self.bg_high_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self._print_param_count()

    def _print_param_count(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[Model: {self.bg_arch}] Total Params: {n:,}")
        print(f" [Params] Main (BG) Network : {n:,} parameters")

    # ``z_idx, y0, x0, rel_err`` are kept for call-site compatibility; the spatial model
    # is position- and error-bound-agnostic.
    def bg_forward_split(self, xp_all_fields, z_idx=None, y0=None, x0=None, rel_err=None):
        _, feat = self.bg_net(xp_all_fields, return_feat=True)
        pred_low = self.bg_low_head(feat)
        pred_mid = self.bg_mid_head(feat)
        pred_high = self.bg_high_head(feat)
        return pred_low, pred_mid, pred_high, pred_low + pred_mid + pred_high

    def bg_forward_with_feat(self, xp_all_fields, z_idx=None, y0=None, x0=None, rel_err=None):
        _, feat = self.bg_net(xp_all_fields, return_feat=True)
        if self.bg_split_bands:
            pred = self.bg_low_head(feat) + self.bg_mid_head(feat) + self.bg_high_head(feat)
        else:
            pred = self.bg_out_head(feat)
        return pred, feat

    def bg_forward(self, xp_all_fields, z_idx=None, y0=None, x0=None, rel_err=None):
        return self.bg_forward_with_feat(xp_all_fields, z_idx, y0, x0, rel_err=rel_err)[0]
