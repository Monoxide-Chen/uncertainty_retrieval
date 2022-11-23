from models.augmenter.normal_augmenter import NormalAugmenter

def augmenter_factory(config):
    augmenter = config['augmenter']
    if augmenter == NormalAugmenter.code():
        return NormalAugmenter(config['feature_size'], config['alpha_scale'], config['beta_scale'])
    else:
        raise ValueError("{} not exists".format(augmenter))
