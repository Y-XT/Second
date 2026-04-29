from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)


def _load_state_dict_with_match_stats(module, state_dict):
    model_dict = module.state_dict()
    matched = {
        k: v
        for k, v in state_dict.items()
        if k in model_dict and hasattr(v, "shape") and model_dict[k].shape == v.shape
    }
    model_dict.update(matched)
    module.load_state_dict(model_dict, strict=False)
    return len(matched), len(model_dict)



class FeatureEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self,num_layers,pretrained=True):
        super().__init__()
        self.num_ch_enc = np.array([64,64,128,256,512])
        self.pretrained_num_loaded = None
        self.pretrained_num_total = None

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101
                   }

        resnet_weights_map = {
            18: models.ResNet18_Weights.DEFAULT,
            34: models.ResNet34_Weights.DEFAULT,
            50: models.ResNet50_Weights.DEFAULT,
            101: models.ResNet101_Weights.DEFAULT
        }

        if num_layers not in resnets:  raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](weights=None)

        if pretrained:
            loaded = load_state_dict_from_url(resnet_weights_map[num_layers].url, progress=True)
            loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] , 1)
            matched_num, total_num = _load_state_dict_with_match_stats(self.encoder, loaded)
            self.pretrained_num_loaded = matched_num
            self.pretrained_num_total = total_num

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self,input_image):


        self.features = []

        self.features.append(self.encoder.relu(self.encoder.bn1(self.encoder.conv1(input_image))))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))


        return self.features











