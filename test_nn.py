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
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            [
                [0.7, 0.8, 0.9],
                [0.9, 0.8, 0.7],
            ],
            [
                [0.6, 0.5, 0.4],
                [0.3, 0.2, 0.1],
            ],
        ])
        sequence_lengths = torch.LongTensor([3, 3])

        attention = LuongAttention(lambda t: t)
        context = attention(query, attended, sequence_lengths)

        expected = torch.Tensor([
            [0.6758598685, 0.7563887239, 0.8369175196],
            [0.8917768598, 0.7950369716, 0.6982971430],
        ])

        self.assertEqual(expected.size(), context.size())
        self.assertTrue(expected.eq(context).all())
