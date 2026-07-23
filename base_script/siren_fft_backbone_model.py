import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from monai.networks.nets import BasicUNet
import numpy as np


class FiLMHead(nn.Module):
    """Tiny FiLM modulation conditioned on log10(rel_err).

    Predicts (gamma, beta) for a feature map [B, C, H, W].
    """

    def __init__(self, channels, hidden=8):
        super().__init__()
        h = max(4, int(hidden))
        self.mlp = nn.Sequential(
            nn.Linear(1, h),
            nn.GELU(),
            nn.Linear(h, 2 * int(channels)),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _to_log10_tensor(rel_err, device, dtype):
        if rel_err is None:
            v = 1e-4
        elif torch.is_tensor(rel_err):
            v = float(rel_err.detach().float().mean().item())
        else:
            v = float(rel_err)
        v = max(v, 1e-8)
        return torch.tensor([[math.log10(v)]], device=device, dtype=dtype)

    def forward(self, feat, rel_err=None):
        cond = self._to_log10_tensor(rel_err, feat.device, feat.dtype)  # [1,1]
        ab = self.mlp(cond)                                             # [1, 2C]
        gamma, beta = ab.chunk(2, dim=-1)
        gamma = (1.0 + gamma).view(1, -1, 1, 1)
        beta = beta.view(1, -1, 1, 1)
        return gamma * feat + beta


class SE2D(nn.Module):
    """Squeeze-and-Excitation style channel attention for NCHW feature maps."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        c = int(channels)
        mid = max(1, c // int(max(1, reduction)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class MicroUNetFiLMBackbone(nn.Module):
    """Micro-UNet backbone with two FiLM hooks conditioned on rel_err.

    Designed to stay within a small parameter budget (<5K when hidden<=7).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden=7,
        film_hidden=8,
        use_se=False,
        se_reduction=4,
    ):
        super().__init__()
        h = int(hidden)
        self.act = nn.GELU()

        self.inc = nn.Conv2d(in_channels, h, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(h, h, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(h, h, kernel_size=3, stride=2, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv1 = nn.Conv2d(h * 2, h, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv2 = nn.Conv2d(h * 2, h, kernel_size=3, padding=1)

        use_se = bool(use_se)
        red = int(se_reduction)
        self.se_bottleneck = SE2D(h, red) if use_se else nn.Identity()
        self.se_decoder = SE2D(h, red) if use_se else nn.Identity()

        # Two FiLM hooks: bottleneck and last decoder feature
        self.film_bottleneck = FiLMHead(channels=h, hidden=film_hidden)
        self.film_decoder = FiLMHead(channels=h, hidden=film_hidden)

        self.outc = nn.Conv2d(h, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.outc.weight)
        if self.outc.bias is not None:
            nn.init.zeros_(self.outc.bias)

    def forward(self, x, rel_err=None, return_feat=False):
        x1 = self.act(self.inc(x))
        x2 = self.act(self.down1(x1))
        x3 = self.act(self.down2(x2))
        x3 = self.se_bottleneck(x3)
        x3 = self.film_bottleneck(x3, rel_err=rel_err)

        u1 = self.up1(x3)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.act(self.up_conv1(u1))

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        feat = self.act(self.up_conv2(u2))
        feat = self.se_decoder(feat)
        feat = self.film_decoder(feat, rel_err=rel_err)

        out = self.outc(feat)
        if return_feat:
            return out, feat
        return out


class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.GELU()

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.act(x + identity)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ResBlock2D(in_ch, out_ch)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        feat = self.block(x)
        down = self.pool(feat)
        return feat, down


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResBlock2D(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class SmallResUNetBackbone(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=16, use_se=False, se_reduction=4):
        super().__init__()
        h1 = hidden
        h2 = hidden * 2
        h3 = hidden * 4

        self.stem = ResBlock2D(in_channels, h1)
        self.down1 = DownBlock(h1, h2)
        self.down2 = DownBlock(h2, h3)
        self.bottleneck = ResBlock2D(h3, h3)
        self.se_bottleneck = SE2D(h3, int(se_reduction)) if use_se else nn.Identity()
        self.up1 = UpBlock(h3, h3, h2)
        self.up2 = UpBlock(h2, h2, h1)
        self.out_proj = nn.Conv2d(h1, out_channels, kernel_size=1)

    def forward(self, x, return_feat=False):
        x0 = self.stem(x)
        s1, x1 = self.down1(x0)
        s2, x2 = self.down2(x1)
        xb = self.bottleneck(x2)
        xb = self.se_bottleneck(xb)
        u1 = self.up1(xb, s2)
        u2 = self.up2(u1, s1)
        out = self.out_proj(u2)

        if return_feat:
            return out, u2
        return out
class HeadAdapter(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Micro_UNet_Backbone(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=8, use_se=False, se_reduction=4):
        super().__init__()
        h = int(hidden)
        self.inc = nn.Conv2d(in_channels, h, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(h, h, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(h, h, kernel_size=3, stride=2, padding=1)

        self.se_bottleneck = SE2D(h, int(se_reduction)) if use_se else nn.Identity()

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_conv1 = nn.Conv2d(h * 2, h, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
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
        x3 = self.se_bottleneck(x3)

        u1 = self.up1(x3)
        if u1.shape[-2:] != x2.shape[-2:]:      # odd/non-power-of-2 spatial dims can break
            u1 = F.interpolate(u1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.act(self.up_conv1(u1))

        u2 = self.up2(u1)
        if u2.shape[-2:] != x1.shape[-2:]:      # exact scale_factor=2 doubling
            u2 = F.interpolate(u2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.act(self.up_conv2(u2))

        out = self.outc(u2)
        if return_feat:
            return out, u2
        return out


        

class UNET_Model(nn.Module):
    def __init__(
        self,
        n_fields=1,
        K=7,
        D=None,
        H=512,
        W=512,
        bg_hidden=4,
        bg_gate_hidden=8,
        bg_latent_channels=None,
        bg_arch="spatial",              # "spatial" / "resunet_small"
        bg_split_bands=False,
        bg_split_mode=None,             # None / "two" / "three"
        bg_use_se=False,
        bg_se_reduction=4,
        bg_feat_attn=False,
        bg_low_adapter=False,
        bg_mid_adapter=False,
        bg_high_adapter=False,
        bg_slab_k=7,
    ):
        super().__init__()
        self.n_fields = n_fields
        self.K = K
        self.D = H if D is None else D
        self.H = H
        self.W = W
        self.bg_arch = str(bg_arch).lower()
        self.bg_slab_k = int(bg_slab_k)
        self.bg_is_slab = self.bg_arch in ("slab2d", "slab2d_unet", "slab_2d")
        self.bg_split_bands = bool(bg_split_bands)
        self.bg_split_mode = bg_split_mode
        self.bg_use_se = bool(bg_use_se)
        self.bg_se_reduction = int(max(1, bg_se_reduction))
        self.bg_feat_attn = bool(bg_feat_attn)
        self.bg_low_adapter = bool(bg_low_adapter)
        self.bg_mid_adapter = bool(bg_mid_adapter)
        self.bg_high_adapter = bool(bg_high_adapter)

        self.bg_latent_channels = bg_hidden if bg_latent_channels is None else int(bg_latent_channels)

        self.bg_gate_hidden = int(bg_gate_hidden)

        # raw input: [Xp_main, aux...]
        bg_in = n_fields

        if self.bg_is_slab:
            bg_in_slab = int(n_fields) * int(self.bg_slab_k)
            self.bg_net = Micro_UNet_Backbone(
                in_channels=bg_in_slab,
                out_channels=bg_hidden,
                hidden=bg_hidden,
                use_se=self.bg_use_se,
                se_reduction=self.bg_se_reduction,
            )
        elif self.bg_arch == "spatial":
            self.bg_net = Micro_UNet_Backbone(
                in_channels=bg_in,
                out_channels=bg_hidden,
                hidden=bg_hidden,
                use_se=self.bg_use_se,
                se_reduction=self.bg_se_reduction,
            )
        elif self.bg_arch == "resunet_small":
            self.bg_net = SmallResUNetBackbone(
                in_channels=bg_in,
                out_channels=bg_hidden,
                hidden=bg_hidden,
                use_se=self.bg_use_se,
                se_reduction=self.bg_se_reduction,
            )
        elif self.bg_arch == "micro_film":
            self.bg_net = MicroUNetFiLMBackbone(
                in_channels=bg_in,
                out_channels=bg_hidden,
                hidden=bg_hidden,
                film_hidden=max(4, int(self.bg_gate_hidden)),
                use_se=self.bg_use_se,
                se_reduction=self.bg_se_reduction,
            )
        else:
            raise ValueError(f"Unsupported bg_arch: {self.bg_arch}")

        self.bg_out_head = nn.Conv2d(bg_hidden, 1, kernel_size=3, padding=1)
        self.bg_low_head = nn.Conv2d(bg_hidden, 1, kernel_size=3, padding=1)
        self.bg_mid_head = nn.Conv2d(bg_hidden, 1, kernel_size=3, padding=1)
        self.bg_high_head = nn.Conv2d(bg_hidden, 1, kernel_size=3, padding=1)

        nn.init.zeros_(self.bg_out_head.weight)
        nn.init.zeros_(self.bg_out_head.bias)
        nn.init.zeros_(self.bg_low_head.weight)
        nn.init.zeros_(self.bg_low_head.bias)
        nn.init.zeros_(self.bg_mid_head.weight)
        nn.init.zeros_(self.bg_mid_head.bias)
        nn.init.zeros_(self.bg_high_head.weight)
        nn.init.zeros_(self.bg_high_head.bias)

        if self.bg_feat_attn:
            self.bg_feat_se = SE2D(bg_hidden, self.bg_se_reduction)
        else:
            self.bg_feat_se = None

        if self.bg_low_adapter:
            self.bg_low_adapt = HeadAdapter(bg_hidden)
        else:
            self.bg_low_adapt = None

        if self.bg_mid_adapter:
            self.bg_mid_adapt = HeadAdapter(bg_hidden)
        else:
            self.bg_mid_adapt = None

        if self.bg_high_adapter:
            self.bg_high_adapt = HeadAdapter(bg_hidden)
        else:
            self.bg_high_adapt = None

        self._print_param_count()

    def _print_param_count(self):
        bg_modules = [
            self.bg_net,
            self.bg_out_head,
            self.bg_low_head,
            self.bg_mid_head,
            self.bg_high_head,
            self.bg_feat_se,
            self.bg_low_adapt,
            self.bg_mid_adapt,
            self.bg_high_adapt,
        ]
        bg_params = 0
        for m in bg_modules:
            if m is not None:
                bg_params += sum(p.numel() for p in m.parameters() if p.requires_grad)

        print(f"\n[Model: {self.bg_arch}] Total Params: {bg_params:,}")
        print(f" [Params] Main (BG) Network : {bg_params:,} parameters")

    def _bg_net_forward_feat(self, bg_inp, rel_err=None):
        if self.bg_arch == "micro_film":
            return self.bg_net(bg_inp, return_feat=True, rel_err=rel_err)
        return self.bg_net(bg_inp, return_feat=True)

    def _postprocess_bg_feat(self, feat):
        if self.bg_feat_se is not None:
            feat = self.bg_feat_se(feat)
        return feat

    def _apply_band_adapter(self, feat, adapter):
        if adapter is not None:
            return adapter(feat)
        return feat

    def _bg_three_band_preds(self, feat):
        feat = self._postprocess_bg_feat(feat)
        feat_low = self._apply_band_adapter(feat, self.bg_low_adapt)
        feat_mid = self._apply_band_adapter(feat, self.bg_mid_adapt)
        feat_high = self._apply_band_adapter(feat, self.bg_high_adapt)
        pred_low = self.bg_low_head(feat_low)
        pred_mid = self.bg_mid_head(feat_mid)
        pred_high = self.bg_high_head(feat_high)
        pred = pred_low + pred_mid + pred_high
        return pred_low, pred_mid, pred_high, pred

    def bg_forward_with_feat(self, xp_all_fields, z_idx, y0, x0, rel_err=None):
        _, feat = self._bg_net_forward_feat(xp_all_fields, rel_err=rel_err)

        if self.bg_split_bands:
            if self.bg_split_mode == "three":
                _, _, _, pred = self._bg_three_band_preds(feat)
            else:
                feat = self._postprocess_bg_feat(feat)
                pred = self.bg_low_head(feat) + self.bg_high_head(feat)
        else:
            feat = self._postprocess_bg_feat(feat)
            pred = self.bg_out_head(feat)

        return pred, feat

    def bg_forward(self, xp_all_fields, z_idx, y0, x0, rel_err=None):
        if self.bg_split_bands:
            if self.bg_split_mode == "three":
                _, _, _, pred = self.bg_forward_split(
                    xp_all_fields, z_idx, y0, x0, rel_err=rel_err
                )
            else:
                _, _, pred = self.bg_forward_split(
                    xp_all_fields, z_idx, y0, x0, rel_err=rel_err
                )
            return pred

        pred, _ = self.bg_forward_with_feat(
            xp_all_fields, z_idx, y0, x0, rel_err=rel_err
        )
        return pred

    def bg_forward_split(self, xp_all_fields, z_idx, y0, x0, rel_err=None):
        _, feat = self._bg_net_forward_feat(xp_all_fields, rel_err=rel_err)

        if self.bg_split_mode == "three":
            pred_low, pred_mid, pred_high, pred = self._bg_three_band_preds(feat)
            return pred_low, pred_mid, pred_high, pred

        feat = self._postprocess_bg_feat(feat)
        pred_low = self.bg_low_head(feat)
        pred_high = self.bg_high_head(feat)
        pred = pred_low + pred_high
        return pred_low, pred_high, pred
