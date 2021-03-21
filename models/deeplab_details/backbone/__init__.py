from ...deeplab_details.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, wm=1.0):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, width_mult=wm)
    else:
        raise NotImplementedError
