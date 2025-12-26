# efficientnet_v1.py
# EfficientNet-B0..B7 in "MobileNetV2-style" (explicit layers, explicit padding, explicit init),
# with output_stride-controlled stride/dilation (for DeepLabV3 / DeepLabV3+ backbones).
#
# Notes:
# - Architecture template follows TorchVision EfficientNet-V1 (MBConv + SE + SiLU). :contentReference[oaicite:0]{index=0}
# - Repeats per stage are computed like TorchVision: ceil(base_repeats * depth_mult). :contentReference[oaicite:1]{index=1}
# - Channel endpoints (stem..head) are taken from your provided EFFICIENTNET_CHANNELS.
#
# If you want to use it as DeepLab backbone:
# - set `output_stride=8` or `output_stride=16`
# - use `model.extract_backbone_features(x)` or `model.features` slices as you prefer.

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

try:  # torchvision<0.4
    from torchvision.models.utils import load_state_dict_from_url
except Exception:  # torchvision>=0.4
    from torch.hub import load_state_dict_from_url

__all__ = [
    "EfficientNetV1",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
]

# TorchVision weights URLs (EfficientNet-V1). :contentReference[oaicite:2]{index=2}
model_urls = {
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-bac287d4.pth",
    "efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth",
    "efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-b3899882.pth",
    "efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth",
    "efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-1a07897c.pth",
    "efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-24a108a5.pth",
    "efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-c5b4e57e.pth",
}

# Your requested channels:
# [stem, stage1, stage2, stage3, stage4, stage5, stage6, stage7, head]
# Strides per stage (stem included):
# stem stride 2
# stage strides: [1,2,2,2,1,2,1]
EFFICIENTNET_CHANNELS: Dict[str, List[int]] = {
    "efficientnet_b0": [32, 16, 24, 40, 80, 112, 192, 320, 1280],
    "efficientnet_b1": [32, 16, 24, 40, 80, 112, 192, 320, 1280],
    "efficientnet_b2": [32, 16, 24, 48, 88, 120, 208, 352, 1408],
    "efficientnet_b3": [32, 24, 32, 48, 96, 136, 232, 384, 1536],
    "efficientnet_b4": [32, 24, 32, 56, 112, 160, 272, 448, 1792],
    "efficientnet_b5": [32, 24, 40, 64, 128, 176, 304, 512, 2048],
    "efficientnet_b6": [32, 32, 40, 72, 144, 200, 344, 576, 2304],
    "efficientnet_b7": [32, 32, 48, 80, 160, 224, 384, 640, 2560],
}

# TorchVision EfficientNet-V1 uses the same stage template for all B0..B7,
# with width_mult/depth_mult scaling. :contentReference[oaicite:3]{index=3}
# We keep the template (expand,kernel,stride,base_repeats) fixed,
# but we force the out_channels endpoints exactly as you provided.
_BASE_STAGE_SPECS = [
    # (expand_ratio, kernel, stride, base_repeats)
    (1, 3, 1, 1),  # stage1
    (6, 3, 2, 2),  # stage2
    (6, 5, 2, 2),  # stage3
    (6, 3, 2, 3),  # stage4
    (6, 5, 1, 3),  # stage5
    (6, 5, 2, 4),  # stage6
    (6, 3, 1, 1),  # stage7
]

# Standard EfficientNet-V1 compound scaling multipliers used by TorchVision. :contentReference[oaicite:4]{index=4}
_EFFICIENTNET_MULTS = {
    "efficientnet_b0": (1.0, 1.0),
    "efficientnet_b1": (1.0, 1.1),
    "efficientnet_b2": (1.1, 1.2),
    "efficientnet_b3": (1.2, 1.4),
    "efficientnet_b4": (1.4, 1.8),
    "efficientnet_b5": (1.6, 2.2),
    "efficientnet_b6": (1.8, 2.6),
    "efficientnet_b7": (2.0, 3.1),
}

# Dropout used by TorchVision EfficientNet-V1 factory functions. :contentReference[oaicite:5]{index=5}
_EFFICIENTNET_DROPOUT = {
    "efficientnet_b0": 0.2,
    "efficientnet_b1": 0.2,
    "efficientnet_b2": 0.3,
    "efficientnet_b3": 0.3,
    "efficientnet_b4": 0.4,
    "efficientnet_b5": 0.4,
    "efficientnet_b6": 0.5,
    "efficientnet_b7": 0.5,
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Same helper style as MobileNetV2 script."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def fixed_padding(kernel_size: int, dilation: int) -> Tuple[int, int, int, int]:
    """Returns (left,right,top,bottom) padding for 'same' conv with dilation."""
    k_eff = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = k_eff - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return (pad_beg, pad_end, pad_beg, pad_end)


def drop_connect(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Stochastic depth (a.k.a. drop connect)."""
    if (not training) or p <= 0.0:
        return x
    keep_prob = 1.0 - p
    # broadcast over (N, C, H, W)
    rand = keep_prob + torch.rand((x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device)
    mask = torch.floor(rand)
    return x / keep_prob * mask


class ConvBNAct(nn.Module):
    """Conv2d + BN + activation, with explicit 'same' padding via F.pad (MobileNetV2 style)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
        groups: int = 1,
        act_layer: Optional[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.stride = int(stride)
        self.input_padding = fixed_padding(self.kernel_size, self.dilation)

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,  # we pad manually
            dilation=self.dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.01)  # matches torchvision defaults :contentReference[oaicite:6]{index=6}
        self.act = act_layer(inplace=True) if act_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.input_padding)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_ch: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        # EfficientNet-V1 uses se_ratio=0.25; channel is made divisible. (common implementations)
        squeezed = _make_divisible(in_ch * se_ratio, 8)
        self.reduce = nn.Conv2d(in_ch, squeezed, kernel_size=1, bias=True)
        self.expand = nn.Conv2d(squeezed, in_ch, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean((2, 3), keepdim=True)
        s = self.reduce(s)
        s = self.act(s)
        s = self.expand(s)
        s = torch.sigmoid(s)
        return x * s


class MBConv(nn.Module):
    """
    EfficientNet-V1 MBConv:
    - optional expand 1x1
    - depthwise kxk (with stride/dilation)
    - SE
    - project 1x1
    - skip + stochastic depth when stride=1 and in==out
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        dilation: int,
        expand_ratio: int,
        kernel_size: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        assert stride in (1, 2)

        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.expand_ratio = int(expand_ratio)
        self.kernel_size = int(kernel_size)
        self.drop_rate = float(drop_rate)

        mid_ch = int(round(in_ch * expand_ratio))
        self.use_residual = (self.stride == 1) and (in_ch == out_ch)

        layers: List[nn.Module] = []

        # 1) expand
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, mid_ch, kernel_size=1, stride=1, dilation=1, groups=1, act_layer=nn.SiLU))

        # 2) depthwise
        layers.append(
            ConvBNAct(
                mid_ch,
                mid_ch,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                groups=mid_ch,
                act_layer=nn.SiLU,
            )
        )

        self.pre_se = nn.Sequential(*layers)

        # 3) SE
        self.se = SqueezeExcite(mid_ch, se_ratio=0.25)

        # 4) project
        self.project_conv = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pre_se(x)
        out = self.se(out)
        out = self.project_conv(out)
        out = self.project_bn(out)

        if self.use_residual:
            out = drop_connect(out, self.drop_rate, self.training)
            out = out + x
        return out


@dataclass(frozen=True)
class EfficientNetConfig:
    name: str
    channels: List[int]  # [stem, s1..s7, head]
    depth_mult: float
    dropout: float
    drop_connect_rate: float = 0.2  # typical EfficientNet default


def _round_repeats(base: int, depth_mult: float) -> int:
    return int(ceil(base * depth_mult))


class EfficientNetV1(nn.Module):
    """
    EfficientNet-V1 B0..B7 with:
    - explicit sequential `features` like MobileNetV2 script
    - explicit init like your MobileNetV2 script
    - output_stride-driven stride/dilation (DeepLab-friendly)

    channels are forced to EFFICIENTNET_CHANNELS[...] (no width rounding),
    repeats follow torchvision-style ceil(base_repeats * depth_mult). :contentReference[oaicite:7]{index=7}
    """

    def __init__(
        self,
        cfg: EfficientNetConfig,
        num_classes: int = 1000,
        output_stride: int = 32,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        assert output_stride in (8, 16, 32), "output_stride must be 8, 16, or 32"
        assert len(cfg.channels) == 9, "channels must be [stem, s1..s7, head]"

        self.cfg = cfg
        self.output_stride = int(output_stride)
        self.num_classes = int(num_classes)

        stem_ch = cfg.channels[0]
        head_ch = cfg.channels[-1]
        stage_out = cfg.channels[1:-1]  # 7 stages

        # We build like your MobileNetV2: a single nn.Sequential of all conv/blocks.
        features: List[nn.Module] = []

        # Track stride/dilation like in MobileNetV2 script
        current_stride = 1
        dilation = 1

        # ---- Stem (3x3 stride2) ----
        features.append(ConvBNAct(in_chans, stem_ch, kernel_size=3, stride=2, dilation=1, act_layer=nn.SiLU))
        current_stride *= 2  # now /2

        # ---- Stages (MBConv) ----
        # Base specs are fixed; repeats are depth-scaled; output channels are forced from your table.
        base_in = stem_ch
        total_blocks = 0
        per_stage_repeats: List[int] = []
        for (t, k, s, n_base) in _BASE_STAGE_SPECS:
            n = _round_repeats(n_base, cfg.depth_mult)
            per_stage_repeats.append(n)
            total_blocks += n

        block_id = 0
        for stage_idx, ((t, k, s, n_base), out_ch) in enumerate(zip(_BASE_STAGE_SPECS, stage_out)):
            repeats = per_stage_repeats[stage_idx]

            for i in range(repeats):
                # First block of the stage uses stride s, others stride 1
                intended_stride = s if i == 0 else 1

                prev_dilation = dilation
                if current_stride == self.output_stride:
                    stride = 1
                    # if we would have downsampled, convert it to dilation increase
                    dilation *= intended_stride
                else:
                    stride = intended_stride
                    current_stride *= intended_stride

                # stochastic depth linearly increased across blocks
                sd = cfg.drop_connect_rate * float(block_id) / float(max(1, total_blocks - 1))
                block_id += 1

                # For the first block in stage: use previous_dilation (like your MobileNetV2)
                # For subsequent blocks: use current dilation
                use_dil = prev_dilation if i == 0 else dilation

                features.append(
                    MBConv(
                        in_ch=base_in,
                        out_ch=out_ch,
                        stride=stride,
                        dilation=use_dil,
                        expand_ratio=t,
                        kernel_size=k,
                        drop_rate=sd,
                    )
                )
                base_in = out_ch

        # ---- Head (1x1 conv to cfg.channels[-1]) ----
        features.append(ConvBNAct(base_in, head_ch, kernel_size=1, stride=1, dilation=1, act_layer=nn.SiLU))

        self.features = nn.Sequential(*features)

        # ---- Classifier (dropout + linear) ----
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(head_ch, num_classes),
        )

        # ---- Weight init (match your MobileNetV2-style) ----
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean((2, 3))
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience for segmentation backbones: returns the final feature map before global pooling.
        (i.e., output of self.features, shape [N, head_ch, H', W']).
        """
        return self.features(x)


def _build_cfg(name: str) -> EfficientNetConfig:
    if name not in EFFICIENTNET_CHANNELS:
        raise ValueError(f"Unknown EfficientNet variant: {name}")

    _, depth = _EFFICIENTNET_MULTS[name]
    dropout = _EFFICIENTNET_DROPOUT[name]
    channels = EFFICIENTNET_CHANNELS[name]
    return EfficientNetConfig(name=name, channels=channels, depth_mult=depth, dropout=dropout)


def _efficientnet(
    arch: str,
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 1000,
    output_stride: int = 32,
    in_chans: int = 3,
) -> EfficientNetV1:
    cfg = _build_cfg(arch)
    model = EfficientNetV1(cfg=cfg, num_classes=num_classes, output_stride=output_stride, in_chans=in_chans)

    if pretrained:
        if arch not in model_urls:
            raise ValueError(f"No URL available for pretrained weights for {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        # If you use pretrained weights but change num_classes, this will fail unless you handle it.
        # Typically for segmentation backbones you set num_classes=1000, load, then drop classifier.
        model.load_state_dict(state_dict)

    return model


def efficientnet_b0(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return _efficientnet("efficientnet_b0", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b1(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return _efficientnet("efficientnet_b1", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b2(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return _efficientnet("efficientnet_b2", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b3(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return _efficientnet("efficientnet_b3", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b4(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return _efficientnet("efficientnet_b4", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b5(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return _efficientnet("efficientnet_b5", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b6(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return _efficientnet("efficientnet_b6", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b7(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return _efficientnet("efficientnet_b7", pretrained=pretrained, progress=progress, **kwargs)


if __name__ == "__main__":
    # quick sanity check
    m = efficientnet_b0(pretrained=False, num_classes=1000, output_stride=8)
    x = torch.randn(1, 3, 224, 224)
    y = m(x)
    print("logits:", y.shape)
    f = m.extract_backbone_features(x)
    print("features:", f.shape, "head_ch:", m.cfg.channels[-1])
