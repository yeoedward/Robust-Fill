'''
TODO:
- DSL
- generate training data
- train it
- Beam search
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RobustFill(nn.Module):
    def __init__(
            self,
            num_tokens,
            string_size,
            program_size,
            program_length):
        super().__init__()

        # Add 1 to num_tokens for end-of-sequence token
        self.embedding = nn.Embedding(num_tokens + 1, string_size)
        self.end_of_sequence_index = torch.LongTensor([num_tokens])

        self.input_lstm = nn.LSTM(string_size, program_size)
        self.output_lstm = nn.LSTM(string_size, program_size)
        self.program_lstm = nn.LSTM(program_size, program_size)
        self.program_length = program_length

    def forward(self, input_sequence, output_sequence):
        end_of_sequence = self.embedding(self.end_of_sequence_index)
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
                end_of_sequence.view(1, 1, -1),
                hidden,
        )

        for c in output_sequence:
            _, hidden = self.output_lstm(c.view(1, 1, -1), hidden)
        _, hidden = self.output_lstm(
                end_of_sequence.view(1, 1, -1),
                hidden,
        )

        program_sequence = []
        for _ in range(self.program_length):
            # Should the first input be hidden?
            _, hidden = self.program_lstm(hidden[0].view(1, 1, -1), hidden)
            program_embedding = F.log_softmax(
                torch.squeeze(hidden[0], dim=1),
                dim=1,
            )
            program_sequence.append(program_embedding)

        return program_sequence


def main():
    torch.manual_seed(1337)

    robust_fill = RobustFill(
        num_tokens=2,
        string_size=2,
        program_size=3,
        program_length=2,
    )
    optimizer = optim.SGD(robust_fill.parameters(), lr=0.001)

    input_sequence = [0, 0, 0]
    output_sequence = [0, 0, 0]

    for i in range(10000):
        optimizer.zero_grad()

        program_sequence = robust_fill(input_sequence, output_sequence)
        loss = F.nll_loss(
            torch.cat(program_sequence),
            torch.LongTensor([0, 2]),
        )

        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(loss)


if __name__ == '__main__':
    main()
