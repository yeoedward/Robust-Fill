from typing import Any, List, Optional, Tuple
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustFill(nn.Module):
    def __init__(
            self,
            string_size: int,
            string_embedding_size: int,
            hidden_size: int,
            program_size: int):
        """
        Implements the RobustFill program synthesis model.

        :param string_size: The number of tokens in the string vocabulary.
        :param string_embedding_size: The size of the string embedding.
        :param hidden_size: The size of the hidden states of the
            input/output encoders and decoder.
        :param program_size: The number of tokens in the program output.
        """
        super().__init__()
        self.embedding = nn.Embedding(string_size, string_embedding_size)
        # TODO: Create static factory methods for different configurations
        # e.g. Basic seq-to-seq vs attention A vs attention B...
        self.input_encoder = AttentionLSTM.lstm(
            input_size=string_embedding_size,
            hidden_size=hidden_size,
        )
        self.output_encoder = AttentionLSTM.single_attention(
            input_size=string_embedding_size,
            hidden_size=hidden_size,
        )
        self.program_decoder = ProgramDecoder(
            hidden_size=hidden_size,
            program_size=program_size,
        )

    @staticmethod
    def _check_num_examples(batch) -> int:
        """Check that the numbers of examples are consistent across batches."""
        assert len(batch) > 0
        num_examples = len(batch[0])
        assert all([
            len(examples) == num_examples
            for examples in batch
        ])
        return num_examples

    @staticmethod
    def _split_flatten_examples(batch: List) -> Tuple[List, List]:
        """
        Flatten the examples so that they just separate data in the same batch.
        They will be integrated again at the max-pool operator.

        :param batch: List (batch_size) of tuples (input, output) of
            lists (sequence_length) of token indices.
        :returns: Tuple of two lists (batch_size * num_examples) of lists
            (sequence_length) of token indices.
        """
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

    def _embed_batch(
            self,
            batch: List,
            device: Optional[torch.device]) -> List[torch.Tensor]:
        """
        Convert each list of tokens in a batch into a tensor of
        shape (sequence_length, string_embedding_size).
        """
        return [
            self.embedding(torch.tensor(sequence, device=device))
            for sequence in batch
        ]

    def forward(
            self,
            batch: List,
            max_program_length: int,
            device: Optional[torch.device] = None):
        """
        Forward pass through RobustFill.

        :param batch: List (batch_size) of tuples (input, output) of
            list (sequence_length) of token indices.
        :param max_program_length: The maximum length of the
            program to generate.
        :param device: The device to send the input data to.
        """
        num_examples = RobustFill._check_num_examples(batch)
        input_batch, output_batch = RobustFill._split_flatten_examples(batch)

        # List (batch_size) of
        # tensors (sequence_length, string_embedding_size).
        input_batch = self._embed_batch(input_batch, device=device)
        # List (batch_size) of
        # tensors (sequence_length, string_embedding_size).
        output_batch = self._embed_batch(output_batch, device=device)

        input_all_hidden, hidden = self.input_encoder(
            input_batch,
            device=device)
        output_all_hidden, hidden = self.output_encoder(
            output_batch,
            hidden=hidden,
            attended=input_all_hidden,
            device=device,
        )
        return self.program_decoder(
            hidden=hidden,
            output_all_hidden=output_all_hidden,
            num_examples=num_examples,
            max_program_length=max_program_length,
            device=device,
        )


class ProgramDecoder(nn.Module):
    """Program decoder module."""

    def __init__(self, hidden_size, program_size):
        super().__init__()
        self.program_size = program_size
        self.hidden_size = hidden_size
        self.program_lstm = AttentionLSTM.single_attention(
            input_size=program_size,
            hidden_size=hidden_size,
        )
        self.max_pool_linear = nn.Linear(hidden_size, hidden_size)
        self.softmax_linear = nn.Linear(hidden_size, program_size)

    def forward(
            self,
            hidden: Tuple[torch.Tensor, torch.Tensor],
            output_all_hidden: Tuple[torch.Tensor, torch.Tensor],
            num_examples: int,
            max_program_length: int,
            device: Optional[torch.device]) -> torch.Tensor:
        """
        Forward pass through the decoder.

        :param hidden: Hidden states of LSTM from output encoder.
        :param output_all_hidden: Entire sequence of hidden states of
            LSTM from output encoder (to be attended to).
        :param num_examples: The number of examples in the batch.
        :param max_program_length: The maximum length of the program
            to generate.
        """
        program_sequence = []
        # List (batch_size) of tensors (1, program_size).
        decoder_input = [
            torch.zeros(1, self.program_size, device=device)
            for _ in range(hidden[0].size()[1])
        ]
        for _ in range(max_program_length):
            _, hidden = self.program_lstm(
                decoder_input,
                hidden=hidden,
                attended=output_all_hidden,
                device=device,
            )
            # (batch_size, hidden_size, num_examples).
            unpooled = (
                torch.tanh(self.max_pool_linear(hidden[0][-1, :, :]))
                .view(-1, num_examples, self.hidden_size)
                .permute(0, 2, 1)
                # Necessary for 'mps' device otherwise maxpool
                # throws an error.
                .contiguous()
            )
            # (batch_size, hidden_size)
            pooled = F.max_pool1d(unpooled, num_examples).squeeze(2)
            # (batch_size, program_size)
            program_embedding = self.softmax_linear(pooled)

            program_sequence.append(program_embedding.unsqueeze(0))
            decoder_input = [
                F.softmax(p, dim=1)
                for p in program_embedding.split(1)
                for _ in range(num_examples)
            ]

        return torch.cat(program_sequence)


class LuongAttention(nn.Module):
    """
    Implements Attention module from:
    Effective Approaches to Attention-based Neural Machine Translation.

    Uses the "general" content-based function.
    """
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def create(query_size):
        return LuongAttention(nn.Linear(query_size, query_size))

    @staticmethod
    def _masked_softmax(
            vectors: torch.Tensor,
            sequence_lengths: torch.LongTensor,
            device: Optional[torch.device]) -> torch.Tensor:
        """
        Returns the softmax of the given vectors, but masking out values
        above the sequence length.

        :param vectors: The vectors to compute the softmax over.
        :param sequence_lengths: The sequence lengths for each batch,
            beyond which to mask out values.
        """
        pad(
            vectors,
            sequence_lengths,
            float('-inf'),
            batch_dim=0,
            sequence_dim=1,
            device=device,
        )
        return F.softmax(vectors, dim=1)

    def forward(
            self,
            query: torch.Tensor,
            attended: torch.Tensor,
            sequence_lengths: torch.LongTensor,
            device: Optional[torch.device]) -> torch.Tensor:
        """
        Compute context vector using weighted average of attended vectors.

        :param query: Query vectors (batch size, query size).
        :param attended: Set of vectors to attend to
            (other sequence length, batch size, query size).
        :param sequence_lengths: Sequence lengths used to mask out
            values at invalid indices.
        :param device: The device to send data to.
        """
        if query.dim() != 2:
            raise ValueError(
                'Expected query to have 2 dimensions. Instead got {}'.format(
                    query.dim(),
                )
            )

        # Pass query through some weights.
        # (batch size, query size)
        q = self.linear(query)
        # Compute alignment by taking dot product of query and
        # attended vectors.
        # (batch size, other sequence length)
        align = LuongAttention._masked_softmax(
            # (seq len, batch, 1, 1)
            torch.matmul(attended.unsqueeze(2), q.unsqueeze(2))
            .squeeze(3)  # (seq len, batch, 1)
            .squeeze(2)  # (seq len, batch)
            .transpose(1, 0),  # (batch, seq len)
            sequence_lengths,
            device=device,
        )
        # Compute weighted average using alignment weights.
        # (batch_size, query size)
        context = (
            # (batch, 1, query size)
            align.unsqueeze(1).bmm(attended.transpose(1, 0))
            .squeeze(1)  # (batch, query size)
        )
        return context


class LSTMAdapter(nn.Module):
    """
    LSTM module that conforms to the same interface as
    the attention-variants.
    """

    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm

    @staticmethod
    def create(input_size, hidden_size):
        return LSTMAdapter(nn.LSTM(input_size, hidden_size))

    def forward(
            self,
            input_: torch.Tensor,
            hidden: torch.Tensor,
            attended_args: Tuple[torch.Tensor, torch.Tensor],
            device: Optional[torch.device]):
        _, hidden = self.lstm(input_.unsqueeze(0), hidden)
        return hidden


class SingleAttention(nn.Module):
    """Attention-LSTM module with single attention."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.attention = LuongAttention.create(hidden_size)
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size)

    def forward(
            self,
            input_: torch.Tensor,
            hidden: torch.Tensor,
            attended_args: Tuple[torch.Tensor, torch.Tensor],
            device: Optional[torch.device]) -> torch.Tensor:
        """
        Forward pass for the single attention-lstm module.

        :param input_: The input tensor (batch_size, input_size).
        :param hidden: The hidden (+ cell) states of the
            LSTM (1, batch_size, hidden_size).
        :param attended_args: The tuple of:
            1. The attended tensor (sequence_length, batch_size, hidden_size).
            2. The sequence lengths (batch_size).
        :param device: The device to send data to.
        """
        attended, sequence_lengths = attended_args
        context = self.attention(
            hidden[0].squeeze(0),
            attended,
            sequence_lengths,
            device=device,
        )
        _, hidden = self.lstm(
            # (1, batch_size, input_size + hidden_size)
            torch.cat((input_, context), 1).unsqueeze(0),
            hidden,
        )
        return hidden


class AttentionLSTM(nn.Module):
    """
    This implements the common Attention-LSTM module in RobustFill.

    The code in this class mostly takes care of unrolling the
    sequential RNN (whether it is a LSTM or LSTM with one or two
    attention modules). We allow the RNN module to be passed in
    as a dependency.
    """

    def __init__(self, rnn: nn.Module):
        """
        Construct the Attention-LSTM module.

        :param rnn: The RNN module to use. This will change
            depending on the experiment.
        """
        super().__init__()
        self.rnn = rnn

    @staticmethod
    def lstm(input_size: int, hidden_size: int) -> 'AttentionLSTM':
        """Create an Attention-LSTM module with no attention."""
        return AttentionLSTM(LSTMAdapter.create(input_size, hidden_size))

    @staticmethod
    def single_attention(input_size: int, hidden_size: int) -> 'AttentionLSTM':
        """Create an Attention-LSTM with a single attention module."""
        return AttentionLSTM(SingleAttention(input_size, hidden_size))

    @staticmethod
    def _pack(
            sequence_batch: List[torch.Tensor],
            ) -> Tuple[PackedSequence, List[int]]:
        """
        Sort and pack a list of sequences to be used with an RNN.

        :param sequence_batch: A list (batch_size) of
            tensors (sequence_length, string_embedding_size).
        """
        # It used to be required to sort it first. Now it seems
        # `pack_sequence()` has an option to do it for us.
        sorted_indices = sorted(
            range(len(sequence_batch)),
            key=lambda i: sequence_batch[i].shape[0],
            reverse=True,
        )
        packed = pack_sequence([sequence_batch[i] for i in sorted_indices])
        return packed, sorted_indices

    @staticmethod
    def _sort(
            hidden: Tuple[torch.Tensor, torch.Tensor],
            attended: Tuple[torch.Tensor, torch.Tensor],
            sorted_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function to sort the hidden and attended tensors based on
        the sorted indices.
        """
        if hidden is None:
            return None, None

        sorted_hidden = (
            hidden[0][:, sorted_indices, :],
            hidden[1][:, sorted_indices, :],
        )

        sorted_attended = None
        if attended is not None:
            sorted_attended = (
                attended[0][:, sorted_indices, :],
                attended[1][sorted_indices],
            )

        return sorted_hidden, sorted_attended

    @staticmethod
    def _unsort(
            all_hidden: torch.Tensor,
            final_hidden: Tuple[torch.Tensor, torch.Tensor],
            sorted_indices: List[int],
            ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Helper function to unsort the hidden and attended tensors based on
        the sorted indices.
        """
        unsorted_indices = [None] * len(sorted_indices)
        for i, original_idx in enumerate(sorted_indices):
            unsorted_indices[original_idx] = i

        unsorted_all_hidden = all_hidden[:, unsorted_indices, :]
        unsorted_final_hidden = (
            final_hidden[0][:, unsorted_indices, :],
            final_hidden[1][:, unsorted_indices, :],
        )

        return unsorted_all_hidden, unsorted_final_hidden

    def _unroll(
            self,
            packed: PackedSequence,
            hidden: Tuple[torch.Tensor, torch.Tensor],
            attended: Tuple[torch.Tensor, torch.Tensor],
            device: Optional[torch.device],
            ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sequentially invoke the RNN on the packed sequence with
        the attended tensor.

        :param packed: The packed sequence to invoke the RNN on.
        :param hidden: The initial hidden (+ cell) states of the RNN.
        :param attended: The attended tensor to use for the RNN.
        """
        all_hn = []
        final_hn = []
        final_cn = []

        pos = 0
        for size in packed.batch_sizes:
            timestep_data = packed.data[pos:pos+size, :]
            pos += size

            if hidden is not None and hidden[0].size()[1] > size:
                hn, cn = hidden
                # Since the packed sequences were sorted by descending
                # sequence length, some later sequences will no longer
                # be relevant here, which means some later hidden states
                # are no longer in play. So here we truncate the states
                # to the relevant ones.
                hidden = hn[:, :size, :], cn[:, :size, :]
                # Since the ones that were truncated away are
                # no longer relevant, it is also their final state,
                # so we save them here.
                final_hn.append(hn[:, size:, :])
                final_cn.append(cn[:, size:, :])

                if attended is not None:
                    attended = (
                        attended[0][:, :size, :],
                        attended[1][:size],
                    )

            hidden = self.rnn(
                input_=timestep_data,
                hidden=hidden,
                attended_args=attended,
                device=device,
            )

            all_hn.append(hidden[0].squeeze(0))

        # Make sure we don't forget to save the final hidden states
        # one last time.
        final_hn.append(hidden[0])
        final_cn.append(hidden[1])

        # Concatenate the final states along the batch dimension.
        # In reverse order so the state mapping to largest seq is first.
        final_hidden = (
            torch.cat(final_hn[::-1], 1),
            torch.cat(final_cn[::-1], 1),
        )
        # all_hn is a list (sequence_length) of
        # tensors (batch_size for timestep x hidden_size).
        # So if we set batch_first=True, we get back tensor
        # (sequence_length x batch_size x hidden_size)
        all_hidden = pad_sequence(all_hn, batch_first=True)

        return all_hidden, final_hidden

    def forward(
            self,
            sequence_batch: List[torch.Tensor],
            hidden: Tuple[torch.Tensor, torch.Tensor] = None,
            attended: Tuple[torch.Tensor, torch.Tensor] = None,
            device: Optional[torch.device] = None):
        """
        Forward pass through the attention-lstm module.

        :param sequence_batch: A list (batch_size) of
            tensors (sequence_length, string_embedding_size).
        :param hidden: A tuple of tensors (the hidden and
            cell states of an LSTM).
        :param attended: A tuple of tensors:
            1. Set of vectors being attended to
                (sequence_length, batch_size, hidden_size).
            2. Sequence lengths (batch_size).
        :param device: The device to send data to.
        """
        if not isinstance(sequence_batch, list):
            raise ValueError(
                'sequence_batch has to be a list. Instead got {}.'.format(
                    type(sequence_batch).__name__,
                )
            )

        packed, sorted_indices = AttentionLSTM._pack(sequence_batch)
        sorted_hidden, sorted_attended = AttentionLSTM._sort(
            hidden,
            attended,
            sorted_indices,
        )
        all_hidden, final_hidden = self._unroll(
            packed,
            sorted_hidden,
            sorted_attended,
            device=device,
        )
        unsorted_all_hidden, unsorted_final_hidden = AttentionLSTM._unsort(
            all_hidden=all_hidden,
            final_hidden=final_hidden,
            sorted_indices=sorted_indices,
        )
        sequence_lengths = torch.tensor([
            s.shape[0] for s in sequence_batch
        ], device=device)
        return (unsorted_all_hidden, sequence_lengths), unsorted_final_hidden


def expand_vector(
        vector: torch.Tensor,
        dim: int,
        num_dims: int) -> torch.Tensor:
    """
    Rehapes a uni-dimensional vector to a multi-dimensional tensor,
    with 1s in all dimensions except the one specified.

    :param vector: The vector to reshape.
    :param dim: The dimension in the new tensor that the vector's
        values will be along.
    :param num_dims: The number of dimensions in the new tensor.
    """
    if vector.dim() != 1:
        raise ValueError('Expected vector of dim 1. Instead got {}.'.format(
            vector.dim(),
        ))

    return vector.view(*[
        vector.size()[0] if d == dim else 1
        for d in range(num_dims)
    ])


def pad(
        tensor: torch.Tensor,
        sequence_lengths: torch.Tensor,
        value: Any,
        batch_dim: int,
        sequence_dim: int,
        device: Optional[torch.device]) -> None:
    """
    Pad the tensor with the given value at indices where it exceeds
    the sequence length for that batch.
    """
    max_length = tensor.size()[sequence_dim]
    indices = expand_vector(
        torch.arange(max_length, device=device),
        sequence_dim,
        tensor.dim(),
    )
    mask = indices >= expand_vector(sequence_lengths, batch_dim, tensor.dim())
    tensor.masked_fill_(mask, value)
