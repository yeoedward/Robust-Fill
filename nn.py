import pprint as pp
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RobustFill(nn.Module):
    def __init__(
            self,
            string_size,
            hidden_size,
            program_size,
            num_lstm_layers,
            program_length):
        super().__init__()

        self.string_size = string_size
        self.num_lstm_layers = num_lstm_layers
        self.program_length = program_length

        self.input_lstm = nn.LSTM(
            input_size=string_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
        )
        self.output_lstm = nn.LSTM(
            input_size=string_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
        )
        self.program_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
        )
        self.linear = nn.Linear(hidden_size, program_size)

    def _one_hot(self, index):
        return (
            torch.zeros(1, self.string_size)
            .scatter_(1, torch.LongTensor([[index]]), 1)
        )

    def forward(self, input_sequence, output_sequence):
        input_sequence = [
            self._one_hot(index)
            for index in input_sequence
        ]
        output_sequence = [
            self._one_hot(index)
            for index in output_sequence
        ]

        hidden = None
        for c in input_sequence:
            _, hidden = self.input_lstm(c.view(1, 1, -1), hidden)

        for c in output_sequence:
            _, hidden = self.output_lstm(c.view(1, 1, -1), hidden)

        program_sequence = []
        previous_hidden = hidden[0][-1, :, :]
        for _ in range(self.program_length):
            _, hidden = self.program_lstm(
                previous_hidden.view(1, 1, -1),
                hidden,
            )
            program_embedding = self.linear(hidden[0][-1, :, :])
            program_sequence.append(torch.squeeze(program_embedding, dim=1))

        return program_sequence


def generate_program():
    p = random.randint(0, 1)
    return [p]


def generate_data(program):
    input_sequence = [random.randint(0, 1)]

    if program[0] == 0:
        return input_sequence, input_sequence

    if program[0] == 1:
        return input_sequence, input_sequence * 2

    return None


def main():
    torch.manual_seed(1337)
    random.seed(420)

    checkpoint_name = './checkpoint.pth'

    robust_fill = RobustFill(
        string_size=2,
        hidden_size=8,
        program_size=2,
        num_lstm_layers=1,
        program_length=1,
    )
    optimizer = optim.SGD(robust_fill.parameters(), lr=0.001)

    example_idx = 0
    while True:
        optimizer.zero_grad()

        program = generate_program()
        input_sequence, output_sequence = generate_data(program)
        program_sequence = robust_fill(input_sequence, output_sequence)
        loss = F.nll_loss(
            F.log_softmax(torch.cat(program_sequence), dim=1),
            torch.LongTensor(program),
        )

        loss.backward()
        optimizer.step()

        if example_idx % 100 == 0:
            print('Loss: {}'.format(loss))
            print(input_sequence)
            print(output_sequence)
            print(program)
            pp.pprint([
                F.softmax(p, dim=1)
                for p in program_sequence
            ])
            print('Checkpointing at example {}'.format(example_idx))
            torch.save(robust_fill.state_dict(), checkpoint_name)
            print('Done')

        example_idx += 1


if __name__ == '__main__':
    main()
