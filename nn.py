import pprint as pp
import random

from torch.nn.utils.rnn import pack_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RobustFill(nn.Module):
    def __init__(
            self,
            string_size,
            string_embedding_size,
            hidden_size,
            program_size,
            num_lstm_layers,
            program_length):
        super().__init__()

        self.program_length = program_length

        self.embedding = nn.Embedding(string_size, string_embedding_size)
        self.input_lstm = nn.LSTM(
            input_size=string_embedding_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
        )
        self.output_lstm = nn.LSTM(
            input_size=string_embedding_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
        )
        self.program_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
        )
        self.max_pool_linear = nn.Linear(hidden_size, hidden_size)
        self.softmax_linear = nn.Linear(hidden_size, program_size)

    @staticmethod
    def _check_num_examples(batch):
        assert len(batch) > 0
        num_examples = len(batch[0])
        assert all([
            len(examples) == num_examples
            for examples in batch
        ])
        return num_examples

    @staticmethod
    def _split_flatten_examples(batch):
        input_batch = [
            input_sequence
            for examples in batch
            for input_sequence, _ in examples
        ]
        output_batch = [
            output_sequence
            for examples in batch
            for _, output_sequence in examples
        ]
        return input_batch, output_batch

    def _embed_batch(self, batch):
        return [
            self.embedding(torch.LongTensor(sequence))
            for sequence in batch
        ]

    @staticmethod
    def _sort_and_pack(sequence_batch):
        sorted_indices = sorted(
            range(len(sequence_batch)),
            key=lambda i: sequence_batch[i].shape[0],
            reverse=True,
        )
        packed = pack_sequence([sequence_batch[i] for i in sorted_indices])
        return packed, sorted_indices

    @staticmethod
    def _sort(hidden, sorted_indices):
        sorted_hn = hidden[0][:, sorted_indices, :]
        sorted_cn = hidden[0][:, sorted_indices, :]
        return sorted_hn, sorted_cn

    @staticmethod
    def _unsort(hidden, sorted_indices):
        unsorted_indices = [None] * len(sorted_indices)
        for i, original_idx in enumerate(sorted_indices):
            unsorted_indices[original_idx] = i

        unsorted_hn = hidden[0][:, unsorted_indices, :]
        unsorted_cn = hidden[1][:, unsorted_indices, :]

        return unsorted_hn, unsorted_cn

    @staticmethod
    def _forward_lstm(lstm, sequence_batch, hidden):
        packed, sorted_indices = RobustFill._sort_and_pack(sequence_batch)
        sorted_hidden = (
            None if hidden is None
            else RobustFill._sort(hidden, sorted_indices)
        )
        _, output_hidden = lstm(packed, sorted_hidden)
        return RobustFill._unsort(output_hidden, sorted_indices)

    # Expects:
    # list (batch_size) of tuples (input, output) of list (sequence_length)
    # of token indices
    def forward(self, batch):
        num_examples = RobustFill._check_num_examples(batch)
        input_batch, output_batch = RobustFill._split_flatten_examples(batch)

        input_batch = self._embed_batch(input_batch)
        output_batch = self._embed_batch(output_batch)

        hidden = RobustFill._forward_lstm(
            self.input_lstm,
            input_batch,
            None,
        )
        hidden = RobustFill._forward_lstm(
            self.output_lstm,
            output_batch,
            hidden,
        )

        program_sequence = []
        previous_hidden = torch.unsqueeze(hidden[0][-1, :, :], dim=0)
        for _ in range(self.program_length):
            _, hidden = self.program_lstm(previous_hidden, hidden)
            unpooled = torch.tanh(self.max_pool_linear(hidden[0][-1, :, :]))
            unpooled = (
                unpooled
                .view(-1, num_examples, unpooled.size()[1])
                .permute(0, 2, 1)
            )
            pooled = F.max_pool1d(unpooled, num_examples)
            program_embedding = self.softmax_linear(pooled.squeeze(2))

            program_sequence.append(program_embedding)

        return program_sequence


def generate_program(batch_size):
    return [random.randint(0, 1) for _ in range(batch_size)]


def generate_data(program_batch, num_examples, string_size):
    # Batch is a:
    # list (batch_size) of tuples (input, output) of list (sequence_length)
    # of token indices
    batch = []
    for program in program_batch:
        examples = []
        for _ in range(num_examples):
            input_sequence = [random.randint(0, string_size-1)]

            if program == 0:
                output_sequence = input_sequence
            elif program == 1:
                output_sequence = input_sequence * 2
            else:
                raise ValueError('Invalid program {}'.format(program))

            examples.append((input_sequence, output_sequence))

        batch.append(examples)

    return batch


def one_hot(self, index, size):
    return (
        torch.zeros(1, size)
        .scatter_(1, torch.LongTensor([[index]]), 1)
    )


def main():
    torch.manual_seed(1337)
    random.seed(420)

    checkpoint_name = './checkpoint.pth'

    string_size = 3
    robust_fill = RobustFill(
        string_size=string_size,
        string_embedding_size=2,
        hidden_size=8,
        program_size=2,
        num_lstm_layers=1,
        program_length=1,
    )
    optimizer = optim.SGD(robust_fill.parameters(), lr=0.01)

    example_idx = 0
    while True:
        optimizer.zero_grad()

        program_batch = generate_program(batch_size=32)
        num_examples = 2
        data_batch = generate_data(program_batch, num_examples, string_size)
        program_sequence = robust_fill(data_batch)
        loss = F.nll_loss(
            F.log_softmax(torch.cat(program_sequence), dim=1),
            torch.LongTensor(program_batch),
        )

        loss.backward()
        optimizer.step()

        if example_idx % 100 == 0:
            print('Loss: {}'.format(loss))
            print_batch_limit = 3
            pp.pprint(data_batch[:print_batch_limit])
            print(program_batch[:print_batch_limit])
            pp.pprint([
                F.softmax(p, dim=1)[:print_batch_limit]
                for p in program_sequence
            ])
            print('Checkpointing at example {}'.format(example_idx))
            torch.save(robust_fill.state_dict(), checkpoint_name)
            print('Done')

        example_idx += 1


if __name__ == '__main__':
    main()
