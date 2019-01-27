import argparse
import pprint as pp
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from robust_fill import RobustFill
from sample import sample_example
from tokens import build_token_tables, tokenize_string
import operators as op


def max_program_length(expected_programs):
    return max([len(program) for program in expected_programs])


def train(
        robust_fill,
        optimizer,
        sample,
        checkpoint_filename,
        checkpoint_step_size,
        checkpoint_print_tensors):
    example_idx = 0
    while True:
        optimizer.zero_grad()

        expected_programs, examples = sample()
        max_length = max_program_length(expected_programs)
        actual_programs = robust_fill(examples, max_length)

        program_size = actual_programs.size()[2]
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

        if example_idx % checkpoint_step_size == 0:
            print('Checkpointing at example {}'.format(example_idx))
            print('Loss: {}'.format(loss))

            if checkpoint_print_tensors:
                print_batch_limit = 3

                print('Examples:')
                pp.pprint(examples[:print_batch_limit])

                print('Expected programs:')
                print(expected_programs[:print_batch_limit])

                print('Actual programs:')
                print(
                    F.softmax(actual_programs, dim=2)
                    .transpose(1, 0)[:print_batch_limit, :, :]
                )

            if checkpoint_filename is not None:
                print('Saving to file {}'.format(checkpoint_filename))
                torch.save(robust_fill.state_dict(), checkpoint_filename)

            print('Done')

        example_idx += 1


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


def sample_easy(batch_size, string_size, num_examples):
    programs = generate_program(batch_size)
    examples = generate_data(programs, num_examples, string_size)
    return programs, examples


def train_easy():
    string_size = 3
    robust_fill = RobustFill(
        string_size=string_size,
        string_embedding_size=2,
        hidden_size=8,
        program_size=2,
    )
    optimizer = optim.SGD(robust_fill.parameters(), lr=0.01)

    def sample():
        return sample_easy(
            batch_size=32,
            string_size=string_size,
            num_examples=2,
        )

    train(
        robust_fill=robust_fill,
        optimizer=optimizer,
        sample=sample,
        checkpoint_filename=None,
        checkpoint_step_size=100,
        checkpoint_print_tensors=True,
    )


def sample_full(token_tables, batch_size, max_expressions, max_characters):
    program_batch, strings_batch = [], []

    for _ in range(batch_size):
        example = sample_example(
            max_expressions=max_expressions,
            max_characters=max_characters,
        )
        program = example.program.to_tokens(token_tables.op_token_table)
        strings = [
            (tokenize_string(input_, token_tables.string_token_table),
             tokenize_string(output, token_tables.string_token_table))
            for input_, output in example.strings
        ]
        program_batch.append(program)
        strings_batch.append(strings)
    return program_batch, strings_batch


def train_full():
    token_tables = build_token_tables()

    checkpoint_filename = './checkpoint.pth'
    robust_fill = RobustFill(
        string_size=len(op.CHARACTER),
        string_embedding_size=32,
        hidden_size=256,
        program_size=len(token_tables.op_token_table),
    )
    optimizer = optim.SGD(robust_fill.parameters(), lr=0.01)

    def sample():
        return sample_full(
            token_tables,
            batch_size=32,
            max_expressions=3,
            max_characters=50,
        )

    train(
        robust_fill=robust_fill,
        optimizer=optimizer,
        sample=sample,
        checkpoint_filename=checkpoint_filename,
        checkpoint_step_size=1,
        checkpoint_print_tensors=False,
    )


def main():
    parser = argparse.ArgumentParser(description='Train RobustFill.')
    parser.add_argument(
        '--dry',
        action='store_true',
        help='run smaller network on easier version of the problem',
    )
    args = parser.parse_args()

    torch.manual_seed(1337)
    random.seed(420)

    if args.dry:
        train_easy()
    else:
        train_full()


if __name__ == '__main__':
    main()
