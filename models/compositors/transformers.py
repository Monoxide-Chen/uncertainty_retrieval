import torch
import torch.nn as nn

from models.attention_modules.self_attention import AttentionModule


class DisentangledTransformer(nn.Module):
    def __init__(self, feature_size, text_feature_size, num_heads, global_styler=None, *args, **kwargs):
        super().__init__()
        self.n_heads = num_heads
        self.c_per_head = feature_size // num_heads
        assert feature_size == self.n_heads * self.c_per_head

        self.att_module = AttentionModule(feature_size, text_feature_size, num_heads, *args, **kwargs)
        self.att_module2 = AttentionModule(feature_size, text_feature_size, num_heads, *args, **kwargs)
        self.global_styler = global_styler

        self.weights = nn.Parameter(torch.tensor([1., 1.]))
        self.instance_norm = nn.InstanceNorm2d(feature_size)

    def forward(self, x, t, *args, **kwargs):
        normed_x = self.instance_norm(x)
        att_out, att_map = self.att_module(normed_x, t, return_map=True)
        out = normed_x + self.weights[0] * att_out

        att_out2, att_map2 = self.att_module2(out, t, return_map=True)
        out = out + self.weights[1] * att_out2

        out = self.global_styler(out, t, x=x)

        return out, att_map
