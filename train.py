import pprint as pp
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from robust_fill import RobustFill


def generate_program(batch_size):
    return [
        [0] if random.randint(0, 1) == 0 else [1, 0]
        for _ in range(batch_size)
    ]


def generate_data(program_batch, num_examples, string_size):
    # Batch is a:
    # list (batch_size) of tuples (input, output) of list (sequence_length)
    # of token indices
    batch = []
    for program in program_batch:
        examples = []
        for _ in range(num_examples):
            input_sequence = [random.randint(0, string_size-1)]

            if program == [0]:
                output_sequence = input_sequence
            elif program == [1, 0]:
                output_sequence = input_sequence * 2
            else:
                raise ValueError('Invalid program {}'.format(program))

            examples.append((input_sequence, output_sequence))

        batch.append(examples)

    return batch


def max_program_length(expected_programs):
    return max([len(program) for program in expected_programs])


def main():
    torch.manual_seed(1337)
    random.seed(420)

    checkpoint_name = './checkpoint.pth'

    string_size = 3
    program_size = 2
    robust_fill = RobustFill(
        string_size=string_size,
        string_embedding_size=2,
        hidden_size=8,
        program_size=program_size,
    )
    optimizer = optim.SGD(robust_fill.parameters(), lr=0.01)

    example_idx = 0
    while True:
        optimizer.zero_grad()

        expected_programs = generate_program(batch_size=32)
        num_examples = 2
        data_batch = generate_data(
            expected_programs,
            num_examples,
            string_size,
        )
        max_length = max_program_length(expected_programs)
        actual_programs = robust_fill(
            data_batch,
            max_program_length=max_length,
        )

        padding_index = -1
        reshaped_actual_programs = (
            actual_programs.transpose(1, 0)
            .contiguous()
            .view(-1, program_size)
        )
        padded_expected_programs = torch.LongTensor([
                program[i] if i < len(program) else padding_index
                for program in expected_programs
                for i in range(max_length)
        ])
        loss = F.cross_entropy(
            reshaped_actual_programs,
            padded_expected_programs,
            ignore_index=padding_index,
        )

        loss.backward()
        optimizer.step()

        print_batch_limit = 3
        if example_idx % 100 == 0:
            print('Loss: {}'.format(loss))

            print('Examples:')
            pp.pprint(data_batch[:print_batch_limit])

            print('Expected programs:')
            print(expected_programs[:print_batch_limit])

            print('Actual programs:')
            print(
                F.softmax(actual_programs, dim=2)
                .transpose(1, 0)[:print_batch_limit, :, :]
            )

            print('Checkpointing at example {}'.format(example_idx))
            torch.save(robust_fill.state_dict(), checkpoint_name)

            print('Done')

        example_idx += 1


if __name__ == '__main__':
    main()
