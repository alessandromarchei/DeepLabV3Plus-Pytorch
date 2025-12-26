#!/usr/bin/env python3
"""
efficientnetv1.py

EfficientNet-V1 family (B0..B7) implemented in a MobileNetV2-like style:
- explicit stages: stem, stage1..stage7, head, classifier
- output_stride control (8/16/32) via stride->dilation conversion (DeepLab-friendly)
- pretrained loader via load_state_dict_from_url
- keeps original classification head (Dropout + Linear)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import math
import torch
from torch import nn
import torch.nn.functional as F

try:  # torchvision<0.4
    from torchvision.models.utils import load_state_dict_from_url
except Exception:  # torchvision>=0.4
    from torch.hub import load_state_dict_from_url


__all__ = [
    "EfficientNetV1",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "get_efficientnet",
    "EFFICIENTNET_CHANNELS",
]

# -------------------------------------------------------------------------
# Channels per stage (provided by you)
# [stem, stage1, stage2, stage3, stage4, stage5, stage6, stage7, head]
# stride pattern: [2, 1, 2, 2, 2, 1, 2, 1, 1]
# -------------------------------------------------------------------------
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

# EfficientNet-V1 baseline (B0) block definition (paper)
# Each tuple: (expansion, kernel, repeats_base, stride_stage, se_ratio)
# Stages correspond to out channels: stage1..stage7
_EFF_B0_BLOCKS = [
    (1, 3, 1, 1, 0.25),  # stage1
    (6, 3, 2, 2, 0.25),  # stage2
    (6, 5, 2, 2, 0.25),  # stage3
    (6, 3, 3, 2, 0.25),  # stage4
    (6, 5, 3, 1, 0.25),  # stage5
    (6, 5, 4, 2, 0.25),  # stage6
    (6, 3, 1, 1, 0.25),  # stage7
]

# Depth multipliers (V1 family)
_DEPTH_MULT = {
    "efficientnet_b0": 1.0,
    "efficientnet_b1": 1.1,
    "efficientnet_b2": 1.2,
    "efficientnet_b3": 1.4,
    "efficientnet_b4": 1.8,
    "efficientnet_b5": 2.2,
    "efficientnet_b6": 2.6,
    "efficientnet_b7": 3.1,
}

# Dropout rates (common defaults)
_DROPOUT = {
    "efficientnet_b0": 0.2,
    "efficientnet_b1": 0.2,
    "efficientnet_b2": 0.3,
    "efficientnet_b3": 0.3,
    "efficientnet_b4": 0.4,
    "efficientnet_b5": 0.4,
    "efficientnet_b6": 0.5,
    "efficientnet_b7": 0.5,
}

# Stochastic depth base (drop connect) – typical defaults
_DROP_CONNECT = {
    "efficientnet_b0": 0.2,
    "efficientnet_b1": 0.2,
    "efficientnet_b2": 0.2,
    "efficientnet_b3": 0.2,
    "efficientnet_b4": 0.2,
    "efficientnet_b5": 0.2,
    "efficientnet_b6": 0.2,
    "efficientnet_b7": 0.2,
}

# Pretrained URLs (torchvision-style; update if your env differs)
model_urls = {
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
    "efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
    "efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
    "efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_rwightman-b6417697.pth",
    "efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_rwightman-c76e70fd.pth",
    "efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_rwightman-1c0d9f9a.pth",
}


# -------------------------------------------------------------------------
# Utils: "same-ish" padding like your MobileNetV2 fixed_padding
# -------------------------------------------------------------------------
def fixed_padding(kernel_size: int, dilation: int) -> Tuple[int, int, int, int]:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return (pad_beg, pad_end, pad_beg, pad_end)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        act: str = "silu",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride

        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0,
            dilation=dilation, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)

        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "relu6":
            self.act = nn.ReLU6(inplace=True)
        elif act == "identity":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown act='{act}'")

        self._pad = fixed_padding(kernel_size, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self._pad)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        reduced = max(1, int(in_ch * se_ratio))
        self.fc1 = nn.Conv2d(in_ch, reduced, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced, in_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean((2, 3), keepdim=True)
        s = F.silu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


def drop_connect(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob <= 0.0 or (not training):
        return x
    keep_prob = 1.0 - drop_prob
    # per-sample mask
    mask = torch.rand((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype) < keep_prob
    x = x / keep_prob
    x = x * mask
    return x


class MBConv(nn.Module):
    """
    EfficientNet-V1 MBConv:
    - optional expand (1x1)
    - depthwise kxk
    - squeeze-excite
    - project (1x1)
    - residual if stride==1 and in==out
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        dilation: int,
        expand_ratio: int,
        kernel_size: int,
        se_ratio: float,
        drop_connect_rate: float,
    ):
        super().__init__()
        assert stride in [1, 2]

        self.use_res = (stride == 1 and in_ch == out_ch)
        self.drop_connect_rate = float(drop_connect_rate)

        mid_ch = in_ch * expand_ratio

        layers: List[nn.Module] = []

        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, mid_ch, kernel_size=1, stride=1, dilation=1, act="silu"))

        # depthwise
        layers.append(
            ConvBNAct(
                mid_ch, mid_ch, kernel_size=kernel_size, stride=stride,
                dilation=dilation, groups=mid_ch, act="silu"
            )
        )

        self.pre_se = nn.Sequential(*layers)
        self.se = SqueezeExcite(mid_ch, se_ratio=se_ratio) if se_ratio and se_ratio > 0 else nn.Identity()

        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pre_se(x)
        out = self.se(out)
        out = self.project(out)
        if self.use_res:
            out = drop_connect(out, self.drop_connect_rate, self.training)
            out = x + out
        return out


def _round_repeats(repeats: int, depth_mult: float) -> int:
    return int(math.ceil(repeats * depth_mult))


class EfficientNetV1(nn.Module):
    def __init__(
        self,
        arch: str,
        num_classes: int = 1000,
        output_stride: int = 32,
        drop_rate: float = 0.2,
        drop_connect_rate: float = 0.2,
    ):
        super().__init__()
        assert arch in EFFICIENTNET_CHANNELS, f"Unknown arch {arch}"
        assert output_stride in (8, 16, 32), "output_stride must be 8,16,32"

        self.arch = arch
        self.num_classes = int(num_classes)
        self.output_stride = int(output_stride)

        ch = EFFICIENTNET_CHANNELS[arch]
        stem_ch = ch[0]
        head_ch = ch[-1]

        depth_mult = _DEPTH_MULT[arch]
        self.drop_rate = float(drop_rate)
        self.drop_connect_rate = float(drop_connect_rate)

        # stride pattern:
        # stem 2, stage1 1, stage2 2, stage3 2, stage4 2, stage5 1, stage6 2, stage7 1, head 1
        stage_strides = [1, 2, 2, 2, 1, 2, 1]  # stage1..stage7

        # -----------------------------------------------------
        # Build with "MobileNetV2-like" output_stride logic
        # -----------------------------------------------------
        current_stride = 1
        dilation = 1

        # Stem
        self.stem = ConvBNAct(3, stem_ch, kernel_size=3, stride=2, dilation=1, act="silu")
        current_stride *= 2  # -> 2

        # Stages
        stage_out = ch[1:8]  # stage1..stage7 out channels
        self.stages = nn.ModuleList()

        # Count total blocks for linear drop-connect schedule
        total_blocks = 0
        repeats_per_stage: List[int] = []
        for i, (exp, k, r, s, se) in enumerate(_EFF_B0_BLOCKS):
            rr = _round_repeats(r, depth_mult)
            repeats_per_stage.append(rr)
            total_blocks += rr

        block_idx = 0
        in_ch = stem_ch

        for si, out_ch in enumerate(stage_out):
            exp, k, r_base, s_base, se = _EFF_B0_BLOCKS[si]
            repeats = repeats_per_stage[si]
            stage_stride = stage_strides[si]  # from your stride pattern

            # decide stride/dilation for the FIRST block of the stage
            prev_dilation = dilation
            if current_stride == self.output_stride:
                first_stride = 1
                dilation *= stage_stride
            else:
                first_stride = stage_stride
                current_stride *= stage_stride

            blocks: List[nn.Module] = []
            for bi in range(repeats):
                stride_i = first_stride if bi == 0 else 1
                dilation_i = prev_dilation if bi == 0 else dilation

                dc = self.drop_connect_rate * (block_idx / max(1, total_blocks - 1))
                blocks.append(
                    MBConv(
                        in_ch=in_ch,
                        out_ch=out_ch,
                        stride=stride_i,
                        dilation=dilation_i,
                        expand_ratio=exp,
                        kernel_size=k,
                        se_ratio=se,
                        drop_connect_rate=dc,
                    )
                )
                in_ch = out_ch
                block_idx += 1

            self.stages.append(nn.Sequential(*blocks))

        # Head
        # head is stride 1 always, but dilation should be current dilation
        self.head = ConvBNAct(in_ch, head_ch, kernel_size=1, stride=1, dilation=1, act="silu")

        # Classifier (original style)
        self.classifier = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(head_ch, self.num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # mimic torchvision-ish init, ok for training-from-scratch too
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for st in self.stages:
            x = st(x)
        x = self.head(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = x.mean((2, 3))
        x = self.classifier(x)
        return x


def _load_pretrained(model: nn.Module, arch: str, progress: bool = True) -> None:
    url = model_urls.get(arch, None)
    if url is None:
        raise ValueError(f"No pretrained URL for arch='{arch}'")

    state_dict = load_state_dict_from_url(url, progress=progress)

    # Be robust if user changes num_classes
    # Remove classifier weights if shape mismatch
    if "classifier.1.weight" in state_dict and hasattr(model, "classifier"):
        w = state_dict["classifier.1.weight"]
        b = state_dict.get("classifier.1.bias", None)
        try:
            target_w = model.classifier[1].weight
            if w.shape != target_w.shape:
                state_dict.pop("classifier.1.weight", None)
                state_dict.pop("classifier.1.bias", None)
        except Exception:
            pass
        # also handle alt keys sometimes used
    state_dict.pop("classifier.weight", None)
    state_dict.pop("classifier.bias", None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # (volutamente niente print: come torchvision)


def get_efficientnet(
    arch: str,
    pretrained: bool = False,
    progress: bool = True,
    **kwargs,
) -> EfficientNetV1:
    """
    Factory in style of mobilenet_v2(pretrained=..., **kwargs)
    kwargs: num_classes, output_stride, drop_rate, drop_connect_rate
    """
    # default drop rates per-family if user doesn't pass them
    if "drop_rate" not in kwargs:
        kwargs["drop_rate"] = _DROPOUT[arch]
    if "drop_connect_rate" not in kwargs:
        kwargs["drop_connect_rate"] = _DROP_CONNECT[arch]

    model = EfficientNetV1(arch=arch, **kwargs)
    if pretrained:
        _load_pretrained(model, arch=arch, progress=progress)
    return model


def efficientnet_b0(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return get_efficientnet("efficientnet_b0", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b1(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return get_efficientnet("efficientnet_b1", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b2(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return get_efficientnet("efficientnet_b2", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b3(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return get_efficientnet("efficientnet_b3", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b4(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return get_efficientnet("efficientnet_b4", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b5(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return get_efficientnet("efficientnet_b5", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b6(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return get_efficientnet("efficientnet_b6", pretrained=pretrained, progress=progress, **kwargs)


def efficientnet_b7(pretrained: bool = False, progress: bool = True, **kwargs) -> EfficientNetV1:
    return get_efficientnet("efficientnet_b7", pretrained=pretrained, progress=progress, **kwargs)
