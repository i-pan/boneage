import pretrainedmodels 
import pretrainedmodels.utils
import torch
import torch.nn as nn

from .efficientnet import EfficientNet
from .resnext_wsl import (
    resnext101_32x8d_wsl  as rx101_32x8, 
    resnext101_32x16d_wsl as rx101_32x16, 
    resnext101_32x32d_wsl as rx101_32x32,
    resnext101_32x48d_wsl as rx101_32x48
)

from .mxresnet import (
    mxresnet18 as _mxresnet18,
    mxresnet34 as _mxresnet34,
    mxresnet50 as _mxresnet50,
    mxresnet101 as _mxresnet101,
   mxresnet152 as _mxresnet152
)

def densenet121(pretrained='imagenet'):
    model = getattr(pretrainedmodels, 'densenet121')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def densenet161(pretrained='imagenet'):
    model = getattr(pretrainedmodels, 'densenet161')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def densenet169(pretrained='imagenet'):
    model = getattr(pretrainedmodels, 'densenet169')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(1664, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def generic(name, pretrained):
    model = getattr(pretrainedmodels, name)(num_classes=1000, pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def resnet18(pretrained='imagenet'):
    return generic('resnet18', pretrained=pretrained)

def resnet34(pretrained='imagenet'):
    return generic('resnet34', pretrained=pretrained)

def resnet50(pretrained='imagenet'):
    return generic('resnet50', pretrained=pretrained)

def resnet101(pretrained='imagenet'):
    return generic('resnet101', pretrained=pretrained)

def resnet152(pretrained='imagenet'):
    return generic('resnet152', pretrained=pretrained)

def se_resnet50(pretrained='imagenet'):
    return generic('se_resnet50', pretrained=pretrained)

def se_resnet101(pretrained='imagenet'):
    return generic('se_resnet101', pretrained=pretrained)

def se_resnet152(pretrained='imagenet'):
    return generic('se_resnet152', pretrained=pretrained)

def se_resnext50(pretrained='imagenet'):
    return generic('se_resnext50_32x4d', pretrained=pretrained)

def se_resnext101(pretrained='imagenet'):
    return generic('se_resnext101_32x4d', pretrained=pretrained)

def generic_mx(name, pretrained):
    model = eval(name)(pretrained=pretrained)
    dim_feats = model[10].in_features
    model[10] = pretrainedmodels.utils.Identity()
    return model, dim_feats

def mxresnet18(pretrained='imagenet'):
    return generic_mx('_mxresnet18', pretrained=pretrained)

def mxresnet34(pretrained='imagenet'):
    return generic_mx('_mxresnet34', pretrained=pretrained)

def mxresnet50(pretrained='imagenet'):
    return generic_mx('_mxresnet50', pretrained=pretrained)

def mxresnet101(pretrained='imagenet'):
    return generic_mx('_mxresnet101', pretrained=pretrained)

def mxresnet152(pretrained='imagenet'):
    return generic_mx('_mxresnet152', pretrained=pretrained)

def inceptionv3(pretrained='imagenet'):
    model, dim_feats = generic('inceptionv3', pretrained=pretrained)
    model.aux_logits = False
    return model, dim_feats

def inceptionv4(pretrained='imagenet'):
    return generic('inceptionv4', pretrained=pretrained)

def inceptionresnetv2(pretrained='imagenet'):
    return generic('inceptionresnetv2', pretrained=pretrained)

def resnext101_wsl(d, pretrained='instagram'):
    model = eval('rx101_32x{}'.format(d))(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats

def resnext101_32x8d_wsl(pretrained='instagram'):
    return resnext101_wsl(8, pretrained=pretrained)
        
def resnext101_32x16d_wsl(pretrained='instagram'):
    return resnext101_wsl(16, pretrained=pretrained)

def resnext101_32x32d_wsl(pretrained='instagram'):
    return resnext101_wsl(32, pretrained=pretrained)

def resnext101_32x48d_wsl(pretrained='instagram'):
    return resnext101_wsl(48, pretrained=pretrained)

def nasnetamobile(pretrained='imagenet'):
    model = getattr(pretrainedmodels, 'nasnetamobile')(num_classes=1000, pretrained=pretrained)
    dim_feats = model.last_linear.in_features 
    model.last_linear = pretrainedmodels.utils.Identity()
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.dropout = pretrainedmodels.utils.Identity()
    return model, dim_feats 

def xception(pretrained='imagenet'):
    model = getattr(pretrainedmodels, 'xception')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def efficientnet(b, pretrained):
    if pretrained == 'imagenet':
        model = EfficientNet.from_pretrained('efficientnet-{}'.format(b))
    elif pretrained is None:
        model = EfficientNet.from_name('efficientnet-{}'.format(b))
    dim_feats = model._fc.in_features
    model._dropout = pretrainedmodels.utils.Identity()
    model._fc = pretrainedmodels.utils.Identity()
    return model, dim_feats

def efficientnet_b0(pretrained='imagenet'):
    return efficientnet('b0', pretrained=pretrained)

def efficientnet_b1(pretrained='imagenet'):
    return efficientnet('b1', pretrained=pretrained)

def efficientnet_b2(pretrained='imagenet'):
    return efficientnet('b2', pretrained=pretrained)

def efficientnet_b3(pretrained='imagenet'):
    return efficientnet('b3', pretrained=pretrained)

def efficientnet_b4(pretrained='imagenet'):
    return efficientnet('b4', pretrained=pretrained)

def efficientnet_b5(pretrained='imagenet'):
    return efficientnet('b5', pretrained=pretrained)

def efficientnet_b6(pretrained='imagenet'):
    return efficientnet('b6', pretrained=pretrained)

def efficientnet_b7(pretrained='imagenet'):
    return efficientnet('b7', pretrained=pretrained)


