'''
TODO:
- Train it
- Beam search
- DSL
'''


import torch
import torch.nn as nn


MAX_PROGRAM_LENGTH = 10


class RobustFill(nn.Module):
    def __init__(
            self,
            num_tokens,
            string_size,
            program_size,
            hidden_size,
            end_of_sequence):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, string_size)
        self.input_lstm = nn.LSTM(string_size, hidden_size)
        self.output_lstm = nn.LSTM(string_size, hidden_size)
        self.program_lstm = nn.LSTM(program_size, hidden_size)
        self.end_of_sequence = end_of_sequence

    def forward(self, input_sequence, output_sequence):
        input_sequence = [
            self.embedding(torch.LongTensor([index]))
            for index in input_sequence
        ]
        output_sequence = [
            self.embedding(torch.LongTensor([index]))
            for index in output_sequence
        ]

        hidden = None
        for c in input_sequence:
            _, hidden = self.input_lstm(c.view(1, 1, -1), hidden)
        _, hidden = self.input_lstm(
                self.end_of_sequence.view(1, 1, -1),
                hidden,
        )

        for c in output_sequence:
            _, hidden = self.output_lstm(c.view(1, 1, -1), hidden)
        _, hidden = self.output_lstm(
                self.end_of_sequence.view(1, 1, -1),
                hidden,
        )

        program_sequence = []
        for _ in range(MAX_PROGRAM_LENGTH):
            # Should the first input be hidden?
            _, hidden = self.program_lstm(hidden[0].view(1, 1, -1), hidden)
            program_sequence.append(hidden[0])
            # TODO: Break if hidden == end of sequence

        return program_sequence


def main():
    torch.manual_seed(1337)
    robust_fill = RobustFill(
        num_tokens=3,
        string_size=3,
        program_size=3,
        hidden_size=3,
        end_of_sequence=torch.randn(3),
    )
    input_sequence = [0, 1, 2]
    output_sequence = [2, 1, 0]
    print(robust_fill(input_sequence, output_sequence))


if __name__ == '__main__':
    main()
