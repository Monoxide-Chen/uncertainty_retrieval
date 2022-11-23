import torch
from torch import nn

class NormalAugmenter(nn.Module):
    '''add Gaussian noisy to feature'''
    def __init__(self, feature_size, alpha_scale=1, beta_scale=1, *args, **kwargs):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.instance_norm = nn.InstanceNorm1d(feature_size)
        self.alpha_scale = alpha_scale
        self.beta_scale = beta_scale

    def forward(self, features, *args, **kwargs):
        std, mean = torch.std_mean(features, dim=1)
        normal_alpha = torch.distributions.Normal(loc=1, scale=std)
        normal_beta = torch.distributions.Normal(loc=mean, scale=std)
        alpha = self.alpha_scale * normal_alpha.sample([features.shape[1]]).transpose(-1, -2)
        beta = self.beta_scale * normal_beta.sample([features.shape[1]]).transpose(-1, -2)

        features = self.instance_norm(features)
        x = alpha * features +  beta
        return x

    @classmethod
    def code(cls) -> str:
        return 'normal_gaussian'
