from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHeadV3Plus, DeepLabV3
from .backbone import (
    mobilenetv2
)
import torch

def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    # 根据输出步幅设置ASPP（空洞空间金字塔池化）的膨胀率
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]  # 当步幅为8时，使用较大的膨胀率
    else:
        aspp_dilate = [6, 12, 18]  # 当步幅不为8时，使用较小的膨胀率

    # 创建MobileNetV2骨干网络，加载预训练权重并设置输出步幅
    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    # for i in range(19):
    #     print(i)
    #     print(backbone.features[i])
    # 重命名骨干网络的层，便于后续特征提取
    backbone.low_level_features = backbone.features[0:4]  # 低级特征（前4层）
    # backbone.mid_low_features = backbone.features[4:7]
    backbone.high_level_features = backbone.features[4:-1]

    backbone.features = None  # 清空原始特征层引用
    backbone.classifier = None  # 清空原始分类器引用

    # 定义输入通道数和低级特征的通道数
    inplanes = 320  # 高级特征的输入通道数
    low_level_planes = 24  # 低级特征的通道数
    # 根据模型名称选择不同的DeepLab变体
    if name == 'deeplabv3plus':
        # 定义返回的特征层：高级特征作为输出，低级特征作为辅助
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        # return_layers = {'mid_low_features':'mid','high_level_features': 'out','low_level_features': 'low_level'}
        # 创建DeepLabV3+的分类头，融合高级和低级特征
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    # 使用IntermediateLayerGetter从骨干网络中提取指定层
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # 组合骨干网络和分类头，构建完整的DeepLabV3模型
    model = DeepLabV3(backbone, classifier)
    return model  # 返回构建好的模型
def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model
def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
