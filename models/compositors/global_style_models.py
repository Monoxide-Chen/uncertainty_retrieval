from torch import nn

from models.utils import calculate_mean_std, EqualLinear
from trainers.abc import AbstractGlobalStyleTransformer


class GlobalStyleTransformer2(AbstractGlobalStyleTransformer):
    def __init__(self, feature_size, text_feature_size, *args, **kwargs):
        super().__init__()
        self.global_transform = EqualLinear(text_feature_size, feature_size * 2)
        self.gate = EqualLinear(text_feature_size, feature_size * 2)
        self.sigmoid = nn.Sigmoid()

        self.init_style_weights(feature_size)

    def forward(self, normed_x, t, *args, **kwargs):
        x_mu, x_std = calculate_mean_std(kwargs['x'])
        gate = self.sigmoid(self.gate(t)).unsqueeze(-1).unsqueeze(-1)
        std_gate, mu_gate = gate.chunk(2, 1)

        global_style = self.global_transform(t).unsqueeze(2).unsqueeze(3)
        gamma, beta = global_style.chunk(2, 1)

        gamma = std_gate * x_std + gamma
        beta = mu_gate * x_mu + beta
        out = gamma * normed_x + beta
        return out

    def init_style_weights(self, feature_size):
        self.global_transform.linear.bias.data[:feature_size] = 1
        self.global_transform.linear.bias.data[feature_size:] = 0

    @classmethod
    def code(cls) -> str:
        return 'global2'
