#!/usr/bin/env python3
"""
efficientnet_backbone.py

EfficientNet backbone (B0..B7) con API compatibile con il tuo MobileNetV2:
  - num_classes
  - output_stride (8/16/32)
  - width_mult (accettato ma non usato, per compatibilità)
  - inverted_residual_setting (accettato ma non usato)
  - round_nearest (accettato ma non usato)

Basato su torchvision.models.efficientnet_*.

Uso tipico come backbone per segmentation:
  model = efficientnet_backbone("efficientnet_b0", pretrained=True, output_stride=8)
  feats = model.forward_features(x)   # [B, C, H/OS, W/OS]
"""

from __future__ import annotations

from typing import Optional, Literal, Dict

import torch
from torch import nn

try:
    from torchvision import models as tv_models
    from torchvision.models.efficientnet import (
        EfficientNet_B0_Weights,
        EfficientNet_B1_Weights,
        EfficientNet_B2_Weights,
        EfficientNet_B3_Weights,
        EfficientNet_B4_Weights,
        EfficientNet_B5_Weights,
        EfficientNet_B6_Weights,
        EfficientNet_B7_Weights,
    )
except Exception as e:
    raise ImportError(
        "Questo modulo richiede torchvision>=0.13 circa (efficientnet in torchvision.models). "
        "Errore import: %s" % str(e)
    )


__all__ = [
    "EfficientNetBackbone",
    "efficientnet_backbone",
]


_EFF_VARIANTS = Literal[
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
]



#strides over channels:
 # [2, 1, 2, 2, 2, 1, 2, 1, 1]
#output channels on the final feature map (before last layer with avgpool and classifier)
_EFF_OUT_CHANNELS: Dict[str, int] = {
    "efficientnet_b0": 320,
    "efficientnet_b1": 320,
    "efficientnet_b2": 352,
    "efficientnet_b3": 384,
    "efficientnet_b4": 448,
    "efficientnet_b5": 512,
    "efficientnet_b6": 576,
    "efficientnet_b7": 640,
}


#number of channels depending on the variant. it is the OS = 4 output channels
_EFF_LOW_LEVEL_CHANNELS: Dict[str, int] = {
    "efficientnet_b0": 24,
    "efficientnet_b1": 24,
    "efficientnet_b2": 24,
    "efficientnet_b3": 32,
    "efficientnet_b4": 32,
    "efficientnet_b5": 40,
    "efficientnet_b6": 40,
    "efficientnet_b7": 48,
}

EFFICIENTNET_CHANNELS = {
    "efficientnet_b0": [32, 16, 24, 40, 80, 112, 192, 320, 1280],
    "efficientnet_b1": [32, 16, 24, 40, 80, 112, 192, 320, 1280],
    "efficientnet_b2": [32, 16, 24, 48, 88, 120, 208, 352, 1408],
    "efficientnet_b3": [32, 24, 32, 48, 96, 136, 232, 384, 1536],
    "efficientnet_b4": [32, 24, 32, 56, 112, 160, 272, 448, 1792],
    "efficientnet_b5": [32, 24, 40, 64, 128, 176, 304, 512, 2048],
    "efficientnet_b6": [32, 32, 40, 72, 144, 200, 344, 576, 2304],
    "efficientnet_b7": [32, 32, 48, 80, 160, 224, 384, 640, 2560],
}

def _get_weights(variant: str, pretrained: bool):
    if not pretrained:
        return None

    mapping = {
        "efficientnet_b0": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnet_b1": EfficientNet_B1_Weights.IMAGENET1K_V1,
        "efficientnet_b2": EfficientNet_B2_Weights.IMAGENET1K_V1,
        "efficientnet_b3": EfficientNet_B3_Weights.IMAGENET1K_V1,
        "efficientnet_b4": EfficientNet_B4_Weights.IMAGENET1K_V1,
        "efficientnet_b5": EfficientNet_B5_Weights.IMAGENET1K_V1,
        "efficientnet_b6": EfficientNet_B6_Weights.IMAGENET1K_V1,
        "efficientnet_b7": EfficientNet_B7_Weights.IMAGENET1K_V1,
    }
    if variant not in mapping:
        raise ValueError(f"Variant non supportata: {variant}. Attese: {list(mapping.keys())}")
    return mapping[variant]


def _build_torchvision_efficientnet(variant: str, pretrained: bool) -> nn.Module:
    weights = _get_weights(variant, pretrained)

    fn_map = {
        "efficientnet_b0": tv_models.efficientnet_b0,
        "efficientnet_b1": tv_models.efficientnet_b1,
        "efficientnet_b2": tv_models.efficientnet_b2,
        "efficientnet_b3": tv_models.efficientnet_b3,
        "efficientnet_b4": tv_models.efficientnet_b4,
        "efficientnet_b5": tv_models.efficientnet_b5,
        "efficientnet_b6": tv_models.efficientnet_b6,
        "efficientnet_b7": tv_models.efficientnet_b7,
    }
    if variant not in fn_map:
        raise ValueError(f"Variant non supportata: {variant}")

    return fn_map[variant](weights=weights)


def _same_padding(kernel_size: int, dilation: int) -> int:
    # padding per "same" su kernel dispari
    return ((kernel_size - 1) // 2) * dilation


def _patch_conv_stride_dilation(conv: nn.Conv2d, new_stride: int, new_dilation: int):
    # cambia stride e imposta dilation/padding coerenti (senza cambiare kernel)
    k = conv.kernel_size[0]
    conv.stride = (new_stride, new_stride)
    conv.dilation = (new_dilation, new_dilation)
    p = _same_padding(k, new_dilation)
    conv.padding = (p, p)


def _apply_output_stride(module: nn.Module, output_stride: int):
    """
    Patch dinamico stile DeepLab:
    - Mantieni downsampling finché current_stride < output_stride
    - Quando current_stride == output_stride, sostituisci stride=2 -> stride=1
      e aumenta dilation*=2 su quei conv (e conv successive) per preservare receptive field.
    """
    if output_stride not in (8, 16, 32):
        raise ValueError(f"output_stride deve essere 8, 16 o 32 (ricevuto {output_stride})")

    current_stride = 1
    dilation = 1

    # percorri i conv in ordine di definizione (modules()).
    # patcha solo conv 3x3/5x5 che fanno effettivo downsample (stride=2).
    for m in module.modules():
        if not isinstance(m, nn.Conv2d):
            continue

        s = m.stride
        if isinstance(s, tuple):
            s = s[0]

        # Consideriamo solo stride=2 (downsampling). Stem incluso.
        if s != 2:
            # se siamo già in regime di dilation>1, aggiorna anche i conv "spaziali"
            # (3x3 o 5x5) per mantenerla consistente.
            if dilation > 1 and m.kernel_size[0] in (3, 5) and m.groups == m.in_channels:
                # tipicamente depthwise conv
                _patch_conv_stride_dilation(m, new_stride=1, new_dilation=dilation)
            elif dilation > 1 and m.kernel_size[0] in (3, 5) and m.groups == 1:
                # a volte conv standard "spaziale"
                _patch_conv_stride_dilation(m, new_stride=1, new_dilation=dilation)
            continue

        # stride=2 trovato
        if current_stride == output_stride:
            # non vogliamo più ridurre: sostituisci stride=1 e aumenta dilation
            dilation *= 2
            _patch_conv_stride_dilation(m, new_stride=1, new_dilation=dilation)
        else:
            # ok ridurre
            current_stride *= 2
            # mantieni dilation attuale (di solito 1) e stride=2 com'è


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet backbone per DeepLab:
    - restituisce un dict: { "low_level", "out" }
    - low_level = primo feature map con output_stride = 4
    - out = feature map finale (OS = 8 o 16)
    """

    def __init__(
        self,
        variant: str = "efficientnet_b0",
        output_stride: int = 16,
        pretrained: bool = True,
    ):
        super().__init__()

        self.variant = variant
        self.output_stride = output_stride

        tv_model = _build_torchvision_efficientnet(variant, pretrained)

        # Patch stride/dilation in stile DeepLab
        _apply_output_stride(tv_model.features, output_stride)

        self.features = tv_model.features

    def forward(self, x):
        low_level = None
        cur_stride = 1

        for layer in self.features:
            x_prev = x
            x = layer(x)

            # detect spatial downsample
            if x.shape[-1] < x_prev.shape[-1]:
                cur_stride *= 2

            # cattura il primo OS=4
            if cur_stride == 4 and low_level is None:
                low_level = x

        assert low_level is not None, "Low-level feature (OS=4) non trovata"

        return {
            "low_level": low_level,   # [B, C_low, H/4, W/4]
            "out": x                 # [B, C_out, H/OS, W/OS]
        }



def efficientnet_backbone(
    variant: _EFF_VARIANTS = "efficientnet_b0",
    pretrained: bool = True,
    progress: bool = True,  # mantenuto per firma simile; torchvision gestisce progress internamente in base alla versione
    **kwargs,
) -> EfficientNetBackbone:
    """
    Factory function simile a mobilenet_v2().
    """
    print(f"Building EfficientNet backbone: {variant}, pretrained={pretrained}, kwargs={kwargs}")
    _ = progress  # no-op (compat)
    return EfficientNetBackbone(variant=variant, pretrained=pretrained, **kwargs)
