from unittest import TestCase

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import torch
import torch.nn as nn

from nn import AttentionLSTM, BasicSeqToSeq, LuongAttention


class TestNN(TestCase):
    def test_attention_lstm_unroll(self):
        lstm = nn.LSTM(2, 3)
        attention_lstm = AttentionLSTM(BasicSeqToSeq(lstm))

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

        all_hidden, final_hidden = attention_lstm._unroll(packed, None)
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
