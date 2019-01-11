import pprint as pp
import random

from torch.nn.utils.rnn import pack_sequence, pad_sequence
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
            program_length):
        super().__init__()

        self.program_length = program_length

        self.embedding = nn.Embedding(string_size, string_embedding_size)
        self.input_lstm = AttentionLSTM.lstm(
            input_size=string_embedding_size,
            hidden_size=hidden_size,
        )
        self.output_lstm = AttentionLSTM.lstm(
            input_size=string_embedding_size,
            hidden_size=hidden_size,
        )
        self.program_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
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

    # Expects:
    # list (batch_size) of tuples (input, output) of list (sequence_length)
    # of token indices
    def forward(self, batch):
        num_examples = RobustFill._check_num_examples(batch)
        input_batch, output_batch = RobustFill._split_flatten_examples(batch)

        input_batch = self._embed_batch(input_batch)
        output_batch = self._embed_batch(output_batch)

        hidden = self.input_lstm(input_batch, None)
        hidden = self.output_lstm(output_batch, hidden)

        program_sequence = []
        previous_hidden = torch.unsqueeze(hidden[0][-1, :, :], dim=0)
        for _ in range(self.program_length):
            _, hidden = self.program_lstm(previous_hidden, hidden)
            hidden_size = hidden[0].size()[2]
            unpooled = (
                torch.tanh(self.max_pool_linear(hidden[0][-1, :, :]))
                .view(-1, num_examples, hidden_size)
                .permute(0, 2, 1)
            )
            pooled = F.max_pool1d(unpooled, num_examples).squeeze(2)
            program_embedding = self.softmax_linear(pooled)

            program_sequence.append(program_embedding)

        return program_sequence


class LuongAttention(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def create(query_size):
        return LuongAttention(nn.Linear(query_size, query_size))

    @staticmethod
    def _masked_softmax(vectors, sequence_lengths):
        batch_size, max_length = vectors.size()
        indices = torch.arange(max_length).unsqueeze(0).expand(batch_size, -1)
        mask = indices >= sequence_lengths.unsqueeze(1)
        vectors.masked_fill_(mask, float('-inf'))
        return F.softmax(vectors, dim=1)

    # attended: (other sequence length x batch size x query size)
    # Uses the "general" content-based function
    def forward(self, query, attended, sequence_lengths):
        # (batch size x query size)
        key = self.linear(query)
        # (batch size x other sequence length)
        align = LuongAttention._masked_softmax(
            torch.matmul(attended.unsqueeze(2), key.unsqueeze(2))
            .squeeze()
            .transpose(1, 0),
            sequence_lengths,
        )
        # (batch_size x query size)
        context = (
            align.unsqueeze(1).bmm(attended.transpose(1, 0))
            .squeeze(1)
        )
        return context


class LSTMAdapter(nn.Module):
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm

    @staticmethod
    def create(input_size, hidden_size):
        return LSTMAdapter(nn.LSTM(input_size, hidden_size))

    # attended and sequence_lengths are here to conform to the same interfaces
    # as the attention-variants
    def forward(self, input_, hidden, attended=None, sequence_lengths=None):
        _, hidden = self.lstm(input_, hidden)
        return hidden


class SingleAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.attention = LuongAttention.create(hidden_size)
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size)

    def forward(self, input_, hidden, attended, sequence_lengths):
        context = self.attention(hidden, attended, sequence_lengths)
        _, hidden = self.lstm(
            torch.cat(input_, context, 1).unsqueeze(0),
            hidden,
        )
        return hidden


class AttentionLSTM(nn.Module):
    def __init__(self, attention_lstm):
        super().__init__()
        self.attention_lstm = attention_lstm

    @staticmethod
    def lstm(input_size, hidden_size):
        return AttentionLSTM(LSTMAdapter.create(input_size, hidden_size))

    @staticmethod
    def single_attention(input_size, hidden_size):
        return AttentionLSTM(SingleAttention(input_size, hidden_size))

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
        sorted_cn = hidden[1][:, sorted_indices, :]
        return sorted_hn, sorted_cn

    @staticmethod
    def _unsort(hidden, sorted_indices):
        unsorted_indices = [None] * len(sorted_indices)
        for i, original_idx in enumerate(sorted_indices):
            unsorted_indices[original_idx] = i

        unsorted_hn = hidden[0][:, unsorted_indices, :]
        unsorted_cn = hidden[1][:, unsorted_indices, :]

        return unsorted_hn, unsorted_cn

    def _unroll(self, packed, original_hidden):
        hidden = original_hidden

        all_hn = []
        final_hn = []
        final_cn = []

        pos = 0
        for size in packed.batch_sizes:
            timestep_data = packed.data[pos:pos+size, :]

            if hidden is not None and hidden[0].size()[1] > size:
                hn, cn = hidden
                hidden = hn[:, :size, :], cn[:, :size, :]
                final_hn.append(hn[:, size:, :])
                final_cn.append(cn[:, size:, :])

            # TODO: Pass in attended and sequence_lengths
            hidden = self.attention_lstm(timestep_data.unsqueeze(0), hidden)

            all_hn.append(hidden[0].squeeze(0))
            pos += size

        final_hn.append(hidden[0])
        final_cn.append(hidden[1])

        final_hidden = (torch.cat(final_hn[::-1], 1),
                        torch.cat(final_cn[::-1], 1))
        # all_hn is a list (sequence_length) of
        # tensors (batch_size for timestep x hidden_size).
        # So if we set batch_first=True, we get back tensor
        # (sequence_length x batch_size x hidden_size)
        all_hidden = pad_sequence(all_hn, batch_first=True)

        return all_hidden, final_hidden

    def forward(self, sequence_batch, hidden):
        packed, sorted_indices = AttentionLSTM._sort_and_pack(sequence_batch)
        sorted_hidden = (
            None if hidden is None
            else AttentionLSTM._sort(hidden, sorted_indices)
        )
        _, final_hidden = self._unroll(packed, sorted_hidden)
        return AttentionLSTM._unsort(final_hidden, sorted_indices)


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
