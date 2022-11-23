import unittest

import torch

from models.attention_modules.self_attention import SelfAttentionMap, GlobalCrossAttentionMap, AttentionModule


class TestSelfAttentionMap(unittest.TestCase):
    b, c, h, w = 2, 10, 5, 5
    t_size = 8
    n_head = 2

    def test_attention_map(self):
        m = SelfAttentionMap(self.c, self.n_head)
        x = torch.randn(self.b, self.c, self.h, self.w)
        map = m(x)

        self.assertTupleEqual((self.b, self.n_head, self.h * self.w, self.h * self.w), map.size())
        for b_i in range(self.b):
            for head_i in range(self.n_head):
                for pos in range(self.h * self.w):
                    self.assertTrue(1, torch.sum(map[b_i][head_i][pos]).data)

    def test_global_cross_attention_map(self):
        m = GlobalCrossAttentionMap(self.c, self.t_size, self.n_head)
        x = torch.randn(self.b, self.c, self.h, self.w)
        t = torch.randn(self.b, self.t_size)
        map = m(x, t)

        self.assertTupleEqual((self.b, self.n_head, self.h * self.w), map.size())
        for b_i in range(self.b):
            for head_i in range(self.n_head):
                self.assertTrue(1, torch.sum(map[b_i][head_i]).data)

    def test_attention(self):
        m = AttentionModule(self.c, self.t_size, self.n_head)
        x = torch.randn(self.b, self.c, self.h, self.w)
        t = torch.randn(self.b, self.t_size)
        out, map = m(x, t, return_map=True)

        self.assertTupleEqual((self.b, self.c, self.h, self.w), out.size())
        self.assertTupleEqual((self.b, self.n_head, self.h * self.w, self.h * self.w), map.size())
        for b_i in range(self.b):
            for head_i in range(self.n_head):
                for pos in range(self.h * self.w):
                    self.assertTrue(2, torch.sum(map[b_i][head_i][pos]).data)


if __name__ == '__main__':
    unittest.main()
