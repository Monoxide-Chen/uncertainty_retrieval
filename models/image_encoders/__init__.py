from typing import Tuple

from trainers.abc import AbstractBaseImageLowerEncoder, AbstractBaseImageUpperEncoder
from models.image_encoders.resnet import ResNet18Layer4Lower, ResNet18Layer4Upper, ResNet50Layer4Lower, \
    ResNet50Layer4Upper


def image_encoder_factory(config: dict) -> Tuple[AbstractBaseImageLowerEncoder, AbstractBaseImageUpperEncoder]:
    model_code = config['image_encoder']
    feature_size = config['feature_size']
    pretrained = config.get('pretrained', True)
    norm_scale = config.get('norm_scale', 4)

    if model_code == 'resnet18_layer4':
        lower_encoder = ResNet18Layer4Lower(pretrained)
        lower_feature_shape = lower_encoder.layer_shapes()['layer4']
        upper_encoder = ResNet18Layer4Upper(lower_feature_shape, feature_size, pretrained=pretrained,
                                            norm_scale=norm_scale)
        return lower_encoder, upper_encoder
    elif model_code == 'resnet50_layer4':
        lower_encoder = ResNet50Layer4Lower(pretrained, stride=config['stride'])
        lower_feature_shape = lower_encoder.layer_shapes()['layer4']
        upper_encoder = ResNet50Layer4Upper(lower_feature_shape, feature_size, pretrained=pretrained,
                                            norm_scale=norm_scale)
        return lower_encoder, upper_encoder
    else:
        raise ValueError("There's no image encoder matched with {}".format(model_code))
