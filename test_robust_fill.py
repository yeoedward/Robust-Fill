from unittest import TestCase

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import torch
import torch.nn as nn

from robust_fill import AttentionLSTM, LSTMAdapter, LuongAttention, pad


class TestNN(TestCase):
    def test_attention_lstm_unroll(self):
        lstm = nn.LSTM(2, 3)
        attention_lstm = AttentionLSTM(LSTMAdapter(lstm))

        a = torch.Tensor([
            [1, 1],
            [2, 2],
            [3, 3],
        ])
        b = torch.Tensor([
            [4, 4],
            [5, 5],
        ])
        c = torch.Tensor([
            [6, 6],
        ])
        packed = pack_sequence([a, b, c])

        all_hidden, final_hidden = attention_lstm._unroll(
            packed,
            hidden=None,
            attended=None,
        )
        all_hidden2, final_hidden2 = lstm(packed, None)

        self.assertTrue(torch.equal(final_hidden[0], final_hidden2[0]))
        self.assertTrue(torch.equal(final_hidden[1], final_hidden2[1]))
        self.assertTrue(torch.equal(
            all_hidden,
            pad_packed_sequence(all_hidden2)[0],
        ))

    def test_luong_attention(self):
        query = torch.Tensor([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ])
        attended = torch.Tensor([
            [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
            ],
            [
                [4, 5, 6],  # Ignored because of sequence length below
                [4, 5, 6],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],  # Ignored because of sequence length below
                [7, 8, 9],
                [7, 8, 9],  # Ignored because of sequence length below
            ],
        ])
        sequence_lengths = torch.LongTensor([1, 3, 2])

        attention = LuongAttention(lambda t: t)
        context = attention(query, attended, sequence_lengths)

        expected = torch.Tensor([
            [1, 2, 3],
            [7, 8, 9],
            [4, 5, 6],
        ])

        self.assertTrue(torch.equal(expected, context))

    def test_pad_2d(self):
        tensor = torch.Tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        sequence_lengths = torch.LongTensor([3, 1, 2])

        pad(tensor, sequence_lengths, 1337, batch_dim=0, sequence_dim=1)

        expected = torch.Tensor([
            [1, 2, 3],
            [4, 1337, 1337],
            [7, 8, 1337],
        ])
        self.assertTrue(torch.equal(expected, tensor))

    def test_pad_3d(self):
        tensor = torch.Tensor([
            [[1, 2],
             [3, 4],
             [5, 6]],
            [[6, 5],
             [4, 3],
             [2, 1]],
        ])
        sequence_lengths = torch.LongTensor([1, 2, 1])

        pad(tensor, sequence_lengths, -1337, batch_dim=1, sequence_dim=0)

        expected = torch.Tensor([
            [[1, 2],
             [3, 4],
             [5, 6]],
            [[-1337, -1337],
             [4, 3],
             [-1337, -1337]],
        ])
        self.assertTrue(torch.equal(expected, tensor))
