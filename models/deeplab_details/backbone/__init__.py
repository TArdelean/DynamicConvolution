from ...deeplab_details.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, ConvLayer):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, ConvLayer=ConvLayer)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, ConvLayer=ConvLayer)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm, ConvLayer=ConvLayer)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, ConvLayer=ConvLayer)
    else:
        raise NotImplementedError
