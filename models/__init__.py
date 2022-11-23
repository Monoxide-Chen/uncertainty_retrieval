from models.compositors import transformer_factory
from models.image_encoders import image_encoder_factory
from models.text_encoders import text_encoder_factory
from models.augmenter import augmenter_factory
from utils.mixins import GradientControlDataParallel
import torch


def create_models(configs, vocabulary):
    text_encoder, text_fc = text_encoder_factory(vocabulary, config=configs)
    lower_img_encoder, upper_img_encoder = image_encoder_factory(config=configs)

    layer_shapes = lower_img_encoder.layer_shapes()
    compositors = transformer_factory({'layer4': layer_shapes['layer4'],
                                       'image_feature_size': upper_img_encoder.feature_size,
                                       'text_feature_size': text_encoder.feature_size}, configs=configs)
    augmenter = augmenter_factory(config=configs)

    models = {
        'text_encoder': text_encoder,
        'lower_image_encoder': lower_img_encoder,
        'upper_image_encoder': upper_img_encoder,
    }
    if text_fc != None:
        models['text_fc'] = text_fc
    if augmenter != None:
        models['augmenter'] = augmenter
    models.update(compositors)

    if configs['num_gpu'] > 1:
        for name, model in models.items():
            models[name] = GradientControlDataParallel(model.cuda())

    return models


