import torch
import torch.nn as nn

from models.utils import reshape_text_features_to_concat


class AttentionModule(nn.Module):
    def __init__(self, feature_size, text_feature_size, num_heads, *args, **kwargs):
        super().__init__()

        self.n_heads = num_heads
        self.c_per_head = feature_size // num_heads
        assert feature_size == self.n_heads * self.c_per_head

        self.self_att_generator = SelfAttentionMap(feature_size, num_heads, *args, **kwargs)
        self.global_att_generator = GlobalCrossAttentionMap(feature_size, text_feature_size, num_heads, *args, **kwargs)

        self.merge = nn.Conv2d(feature_size + text_feature_size, feature_size, kernel_size=1, bias=False)
        self.W_v = nn.Conv2d(feature_size, feature_size, kernel_size=1, bias=False)
        self.W_r = nn.Conv2d(feature_size, feature_size, kernel_size=1)

    def forward(self, x, t, return_map=False, *args, **kwargs):
        b, c, h, w = x.size()

        t_reshaped = reshape_text_features_to_concat(t, x.size())
        vl_features = self.merge(torch.cat([x, t_reshaped], dim=1))  # (b, c, h, w)

        values = self.W_v(vl_features)
        values = values.view(b * self.n_heads, self.c_per_head, h, w).view(b * self.n_heads, self.c_per_head, h * w)

        self_att_map = self.self_att_generator(x)  # (b, num_heads, h * w, h * w)
        global_cross_att_map = self.global_att_generator(x, t)
        global_cross_att_map = global_cross_att_map.view(b, self.n_heads, 1, h * w)  # (b, num_heads, 1, h * w)
        att_map = self_att_map + global_cross_att_map  # (b, num_heads, h * w, h * w)
        att_map_reshaped = att_map.view(b * self.n_heads, h * w, h * w)  # (b * num_heads, h * w, h * w)

        att_out = torch.bmm(values, att_map_reshaped.transpose(1, 2))  # (b * num_heads, c_per_head, h * w)
        att_out = att_out.view(b, self.n_heads * self.c_per_head, h * w)
        att_out = att_out.view(b, self.n_heads * self.c_per_head, h, w)
        att_out = self.W_r(att_out)

        return att_out, att_map if return_map else att_out


class SelfAttentionMap(nn.Module):
    def __init__(self, feature_size, num_heads, *args, **kwargs):
        super().__init__()

        self.n_heads = num_heads
        self.c_per_head = feature_size // num_heads
        assert feature_size == self.n_heads * self.c_per_head

        self.W_k = nn.Conv2d(feature_size, feature_size, kernel_size=1, bias=False)
        self.W_q = nn.Conv2d(feature_size, feature_size, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, *args, **kwargs):
        b, c, h, w = x.size()

        keys, queries = self.W_k(x), self.W_q(x)
        keys = keys.view(b * self.n_heads, self.c_per_head, h, w).view(b * self.n_heads, self.c_per_head, h * w)
        queries = queries.view(b * self.n_heads, self.c_per_head, h, w).view(b * self.n_heads, self.c_per_head, h * w)

        att_map = torch.bmm(queries.transpose(1, 2), keys) / (self.c_per_head ** 0.5)
        att_map = self.softmax(att_map)  # (b * num_heads, h * w, h * w), torch.sum(att_map[batch_idx][?]) == 1
        att_map = att_map.view(b, self.n_heads, h * w, h * w)

        return att_map


class GlobalCrossAttentionMap(nn.Module):
    def __init__(self, feature_size, text_feature_size, num_heads, normalizer=None, *args, **kwargs):
        super().__init__()

        self.n_heads = num_heads
        self.c_per_head = feature_size // num_heads
        assert feature_size == self.n_heads * self.c_per_head

        self.W_t = nn.Linear(text_feature_size, feature_size)
        self.normalize = normalizer if normalizer else nn.Softmax(dim=1)

    def forward(self, x, t):
        b, c, h, w = x.size()

        x_reshape = x.view(b * self.n_heads, self.c_per_head, h, w)
        x_reshape = x_reshape.view(b * self.n_heads, self.c_per_head, h * w)

        t_mapped = self.W_t(t)
        t_mapped = t_mapped.view(b * self.n_heads, self.c_per_head, 1)

        att_map = torch.bmm(x_reshape.transpose(1, 2), t_mapped).squeeze(-1) / (self.c_per_head ** 0.5)
        att_map = self.normalize(att_map)  # (b * n_heads, h * w)
        att_map = att_map.view(b * self.n_heads, 1, h * w)
        att_map = att_map.view(b, self.n_heads, h * w)

        return att_map
