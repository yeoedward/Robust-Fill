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

    # TODO: Replace with sparse embedding
    def _one_hot(self, index):
        return (
            torch.zeros(1, 1, self.string_size)
            .scatter_(2, torch.LongTensor([[[index]]]), 1)
        )

    def _embed(self, sequence):
        return torch.cat([
            self._one_hot(index)
            for index in sequence
        ])

    def _embed_batch(self, sequence_batch):
        return [
            self._embed(sequence)
            for sequence in sequence_batch
        ]

    def forward(self, input_batch, output_batch):
        input_batch = self._embed_batch(input_batch)
        output_batch = self._embed_batch(output_batch)

        input_sequence = input_batch[0]
        output_sequence = output_batch[0]

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
            program_sequence.append(program_embedding)

        return program_sequence


def generate_program(batch_size):
    return [[random.randint(0, 1)] for _ in range(batch_size)]


def generate_data(program_batch, string_size):
    input_batch, output_batch = [], []
    for program in program_batch:
        input_sequence = [random.randint(0, 1)]

        if program[0] == 0:
            output_sequence = input_sequence
        elif program[0] == 1:
            output_sequence = input_sequence * 2
        else:
            raise ValueError('Invalid program {}'.format(program))

        input_batch.append(input_sequence)
        output_batch.append(output_sequence)

    return input_batch, output_batch


def main():
    torch.manual_seed(1337)
    random.seed(420)

    checkpoint_name = './checkpoint.pth'

    string_size = 2
    robust_fill = RobustFill(
        string_size=string_size,
        hidden_size=8,
        program_size=2,
        num_lstm_layers=1,
        program_length=1,
    )
    optimizer = optim.SGD(robust_fill.parameters(), lr=0.01)

    example_idx = 0
    while True:
        optimizer.zero_grad()

        program = generate_program(batch_size=1)
        input_batch, output_batch = generate_data(program, string_size)
        program_sequence = robust_fill(input_batch, output_batch)
        loss = F.nll_loss(
            F.log_softmax(torch.cat(program_sequence), dim=1),
            # TODO: Generalize to multi-length programs
            torch.LongTensor([p[0] for p in program]),
        )

        loss.backward()
        optimizer.step()

        if example_idx % 100 == 0:
            print('Loss: {}'.format(loss))
            print(input_batch)
            print(output_batch)
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
