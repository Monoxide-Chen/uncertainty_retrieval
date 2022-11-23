from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

from trainers.abc import AbstractBaseImageLowerEncoder, AbstractBaseImageUpperEncoder


class ResNet18Layer4Lower(AbstractBaseImageLowerEncoder):
    def __init__(self, pretrained=True):
        super().__init__()
        self._model = resnet18(pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)

        layer1_out = self._model.layer1(x)
        layer2_out = self._model.layer2(layer1_out)
        layer3_out = self._model.layer3(layer2_out)
        layer4_out = self._model.layer4(layer3_out)

        return layer4_out, (layer3_out, layer2_out, layer1_out)

    def layer_shapes(self):
        return {'layer4': 512, 'layer3': 256, 'layer2': 128, 'layer1': 64}


class ResNet18Layer4Upper(AbstractBaseImageUpperEncoder):
    def __init__(self, lower_feature_shape, feature_size, pretrained=True, *args, **kwargs):
        super().__init__(lower_feature_shape, feature_size, pretrained=pretrained, *args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.lower_feature_shape, self.feature_size)
        self.norm_scale = kwargs['norm_scale']

    def forward(self, layer4_out: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(layer4_out)
        x = torch.flatten(x, 1)
        x = F.normalize(self.fc(x)) * self.norm_scale

        return x


class GAPResNet18Layer4Upper(AbstractBaseImageUpperEncoder):
    def __init__(self, lower_feature_shape, feature_size, pretrained=True, *args, **kwargs):
        super().__init__(lower_feature_shape, feature_size, pretrained=pretrained, *args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm_scale = kwargs['norm_scale']

    def forward(self, layer4_out: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(layer4_out)
        x = torch.flatten(x, 1)
        x = F.normalize(x) * self.norm_scale

        return x


class ResNet50Layer4Lower(AbstractBaseImageLowerEncoder):
    def __init__(self, pretrained=True, stride=False):
        super().__init__()
        self._model = resnet50(pretrained=pretrained)
        # avg pooling to global pooling
        if stride == True:
            self._model.layer4[0].downsample[0].stride = (1,1)
            self._model.layer4[0].conv2.stride = (1,1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)

        layer1_out = self._model.layer1(x)
        layer2_out = self._model.layer2(layer1_out)
        layer3_out = self._model.layer3(layer2_out)
        layer4_out = self._model.layer4(layer3_out)

        return layer4_out, (layer3_out, layer2_out, layer1_out)

    def layer_shapes(self):
        return {'layer4': 2048, 'layer3': 1024, 'layer2': 512, 'layer1': 256}


class ResNet50Layer4Upper(AbstractBaseImageUpperEncoder):
    def __init__(self, lower_feature_shape, feature_size, pretrained=True, *args, **kwargs):
        super().__init__(lower_feature_shape, feature_size, pretrained=pretrained, *args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.lower_feature_shape, self.feature_size)
        self.norm_scale = kwargs['norm_scale']

    def forward(self, layer4_out: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(layer4_out)
        x = torch.flatten(x, 1)
        x = F.normalize(self.fc(x)) * self.norm_scale

        return x
