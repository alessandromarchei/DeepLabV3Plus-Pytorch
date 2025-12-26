from __future__ import annotations


from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import (
    resnet,
    mobilenetv2,
    efficientnet,
    hrnetv2,
    xception
)
import torch.nn as nn

def _segm_hrnet(name, backbone_name, num_classes, pretrained_backbone):

    backbone = hrnetv2.__dict__[backbone_name](pretrained_backbone)
    # HRNetV2 config:
    # the final output channels is dependent on highest resolution channel config (c).
    # output of backbone will be the inplanes to assp:
    hrnet_channels = int(backbone_name.split('_')[-1])
    inplanes = sum([hrnet_channels * 2 ** i for i in range(4)])
    low_level_planes = 256 # all hrnet version channel output from bottleneck is the same
    aspp_dilate = [12, 24, 36] # If follow paper trend, can put [24, 48, 72].

    if name=='deeplabv3plus':
        return_layers = {'stage4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'stage4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers, hrnet_flag=True)
    model = DeepLabV3(backbone, classifier)
    return model

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_xception(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        replace_stride_with_dilation=[False, False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, False, True]
        aspp_dilate = [6, 12, 18]
    
    backbone = xception.xception(pretrained= 'imagenet' if pretrained_backbone else False, replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 128
    
    if name=='deeplabv3plus':
        return_layers = {'conv4': 'out', 'block1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'conv4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    
    # rename layers
    backbone.low_level_features = backbone.features[0:4]        #take the first 4 layers (32, 16, 24, 24 channels)
    backbone.high_level_features = backbone.features[4:-1]      #take the remaining layers except the last ConvBNReLU layer (96, 1280 channels)
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _aspp_dilate(output_stride: int):
    if output_stride == 8:
        return [12, 24, 36]
    return [6, 12, 18]


# -------------------------------------------------------------------------
# EfficientNetV1
# -------------------------------------------------------------------------
def _segm_efficientnet(
    name: str,
    backbone_name: str,
    num_classes: int,
    output_stride: int,
    pretrained_backbone: bool,
    **kwargs
):
    """
    EfficientNet split (coerente con stride pattern):
      stem: stride 2
      stage1: stride 1  -> still 2
      stage2: stride 2  -> 4  => LOW LEVEL (fine stage2)
      stage3: stride 2  -> 8
      stage4: stride 2  -> 16
      stage5: stride 1  -> 16
      stage6: stride 2  -> 32
      stage7: stride 1  -> 32
      head:  stride 1   -> 32 => HIGH LEVEL (fine head)

    output_stride=8/16/32 gestito dentro backbone (stride->dilation).
    """
    aspp_from = kwargs.pop("aspp_from", "head")
    aspp_dilate = _aspp_dilate(output_stride)

    # build efficientnet backbone
    backbone = efficientnet.get_efficientnet(
        backbone_name,
        pretrained=pretrained_backbone,
        output_stride=output_stride,
        num_classes=1000,  # classificazione originale
    )

    ch = efficientnet.EFFICIENTNET_CHANNELS[backbone_name]

    # ----------------------------
    # LOW LEVEL (stage2)
    # ----------------------------
    low_level_planes = ch[2]   # stage2 out channels

    backbone.low_level_features = nn.Sequential(
        backbone.stem,
        backbone.stages[0],  # stage1
        backbone.stages[1],  # stage2
    )

    # ----------------------------
    # HIGH LEVEL CANDIDATES
    # ----------------------------
    backbone.stage7_features = nn.Sequential(
        backbone.stages[2],  # stage3
        backbone.stages[3],  # stage4
        backbone.stages[4],  # stage5
        backbone.stages[5],  # stage6
        backbone.stages[6],  # stage7
    )

    backbone.head_features = nn.Sequential(
        backbone.stage7_features,
        backbone.head,
    )

    # ----------------------------
    # ASPP TAP SELECTION
    # ----------------------------
    if aspp_from == "head":
        backbone.high_level_features = backbone.head_features
        inplanes = ch[-1]       # head channels (1280 / 1536 / ...)

    elif aspp_from == "stage7":
        backbone.high_level_features = backbone.stage7_features
        inplanes = ch[-2]       # stage7 channels (320 / 384 / ...)

    else:
        raise ValueError(
            f"Invalid aspp_from='{aspp_from}', choose ['head', 'stage7']"
        )

    # ----------------------------
    # CLEANUP BACKBONE
    # ----------------------------
    backbone.stem = None
    backbone.stages = None
    backbone.head = None
    backbone.classifier = None

    # ----------------------------
    # DEEPLAB HEAD
    # ----------------------------
    if name == "deeplabv3plus":
        return_layers = {
            "high_level_features": "out",
            "low_level_features": "low_level",
        }
        classifier = DeepLabHeadV3Plus(
            inplanes,
            low_level_planes,   
            num_classes,
            aspp_dilate,
        )

    elif name == "deeplabv3":
        return_layers = {"high_level_features": "out"}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    else:
        raise ValueError(f"Unknown model name: {name}")

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)

    print(f"EfficientNetV1 Backbone: {backbone_name}, ASPP input channels: {inplanes}, low-level channels: {low_level_planes}")
    
    return model



def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone, **kwargs):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

    #list of efficientnet variants: b0, b1, b2, b3, b4, b5, b6, b7
    elif 'efficientnet' in backbone:
        model = _segm_efficientnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, **kwargs)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('hrnetv2'):
        model = _segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone=='xception':
        model = _segm_xception(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model


# Deeplab v3
def deeplabv3_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False, **kwargs): # no pretrained backbone yet
    return _load_model('deeplabv3', 'hrnetv2_48', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True, **kwargs):
    return _load_model('deeplabv3', 'hrnetv2_32', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_xception(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


# Deeplab v3+
def deeplabv3plus_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False, **kwargs): # no pretrained backbone yet
    return _load_model('deeplabv3plus', 'hrnetv2_48', num_classes, output_stride, pretrained_backbone=pretrained_backbone, **kwargs)

def deeplabv3plus_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True, **kwargs):
    return _load_model('deeplabv3plus', 'hrnetv2_32', num_classes, output_stride, pretrained_backbone=pretrained_backbone, **kwargs)
def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_xception(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


#efficientnet models
def deeplabv3plus_efficientnet_b0(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a EfficientNet-B0 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'efficientnet_b0', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, **kwargs)

def deeplabv3plus_efficientnet_b1(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a EfficientNet-B1 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'efficientnet_b1', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, **kwargs)

def deeplabv3plus_efficientnet_b2(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a EfficientNet-B2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'efficientnet_b2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, **kwargs)

def deeplabv3plus_efficientnet_b3(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a EfficientNet-B3 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'efficientnet_b3', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, **kwargs)

def deeplabv3plus_efficientnet_b4(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a EfficientNet-B4 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'efficientnet_b4', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, **kwargs)

def deeplabv3plus_efficientnet_b5(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a EfficientNet-B5 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'efficientnet_b5', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, **kwargs)

def deeplabv3plus_efficientnet_b6(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a EfficientNet-B6 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'efficientnet_b6', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, **kwargs)

def deeplabv3plus_efficientnet_b7(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a EfficientNet-B7 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'efficientnet_b7', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, **kwargs)
