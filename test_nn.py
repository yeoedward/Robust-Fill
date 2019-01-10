from unittest import TestCase

import torch

from nn import LuongAttention


class TestNN(TestCase):
    # TODO: Test with "ragged" sequence lengths
    def test_luong_attention(self):
        query = torch.Tensor([
            [1, 2, 3],
            [4, 5, 6],
        ])
        attended = torch.Tensor([
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [9, 8, 7],
            ],
            [
                [6, 5, 4],
                [3, 2, 1],
            ],
        ])

        attention = LuongAttention(lambda t: t)
        context = attention(query, attended)

        expected = torch.Tensor([
            [7, 8, 9],
            [9, 8, 7],
        ])

        self.assertEqual(expected.size(), context.size())
        self.assertTrue(expected.eq(context).all())
