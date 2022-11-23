from models.compositors.global_style_models import GlobalStyleTransformer2
from models.compositors.transformers import DisentangledTransformer


def global_styler_factory(code, feature_size, text_feature_size):
    if code == GlobalStyleTransformer2.code():
        return GlobalStyleTransformer2(feature_size, text_feature_size)
    else:
        raise ValueError("{} not exists".format(code))


def transformer_factory(feature_sizes, configs):
    text_feature_size = feature_sizes['text_feature_size']
    num_heads = configs['num_heads']

    global_styler_code = configs['global_styler']
    global_styler = global_styler_factory(global_styler_code, feature_sizes['layer4'], text_feature_size)
    return {'layer4': DisentangledTransformer(feature_sizes['layer4'], text_feature_size, num_heads=num_heads,
                                              global_styler=global_styler)}
