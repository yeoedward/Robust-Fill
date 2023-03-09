import argparse
from collections import namedtuple
import os
import pprint as pp
import random
from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity, schedule

from robust_fill import RobustFill
from sample import sample_example
from tokens import TokenTables, build_token_tables, tokenize_string
import operators as op


# Number of times to retry if cuda OOM is encountered.
OOM_RETRIES = 2


# Configuration for training.
class Config(NamedTuple):
    model: nn.Module
    sample: Callable[[], Tuple[List, List]]
    optimizer: optim.Optimizer
    clip_grad_value: float
    device: Optional[torch.device]
    checkpoint_filename: str
    checkpoint_step_size: int
    checkpoint_print_tensors: bool


# Misc info returned by training_step() for logging.
StepInfo = namedtuple(
    'StepInfo',
    [
        'loss',
        'examples',
        'expected_programs',
        'actual_programs',
    ],
)


def max_program_length(expected_programs: List[List[int]]) -> int:
    """Return length of longest program."""
    return max([len(program) for program in expected_programs])


def training_step(config: Config) -> StepInfo:
    """Execute a single training step."""
    expected_programs, examples = config.sample()
    max_length = max_program_length(expected_programs)
    actual_programs = config.model(examples, max_length, device=config.device)

    # Compute cross-entropy loss ignoring padding tokens due to
    # different program lengths.
    program_size = actual_programs.size()[2]
    padding_index = -1
    # Reshape actual_programs (seq length, batch size, program size)
    # to (batch size * seq length, program size).
    reshaped_actual_programs = (
        actual_programs.transpose(1, 0)
        # Necessary because .view() expects contiguity but
        # .transpose() doesn't copy.
        .contiguous()
        .view(-1, program_size)
    )
    # Convert expected programs from list of lists of ints (uneven lengths)
    # to a tensor of (batch size * max length) with padding tokens.
    padded_expected_programs = torch.tensor([
            program[i] if i < len(program) else padding_index
            for program in expected_programs
            for i in range(max_length)
    ], device=config.device)
    loss = F.cross_entropy(
        reshaped_actual_programs,
        padded_expected_programs,
        ignore_index=padding_index,
    )

    config.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(
        config.model.parameters(),
        clip_value=config.clip_grad_value)
    config.optimizer.step()

    return StepInfo(
        loss=loss,
        examples=examples,
        expected_programs=expected_programs,
        actual_programs=actual_programs)


def train(config: Config) -> None:
    """Infinite loop for training."""
    if (config.checkpoint_filename is not None
       and os.path.exists(config.checkpoint_filename)):
        print('Starting model from existing checkpoint file: '
              f'{config.checkpoint_filename}')
        config.model.load_state_dict(torch.load(config.checkpoint_filename))

    config.model.to(config.device)

    example_idx = 0
    while True:
        for i in range(OOM_RETRIES + 1):
            try:
                step_info = training_step(config)
                break
            except torch.cuda.OutOfMemoryError:
                if i == OOM_RETRIES:
                    raise
                print('Out of memory, retrying')

        if example_idx % config.checkpoint_step_size == 0:
            print('Checkpointing at example {}'.format(example_idx))
            print('Loss: {}'.format(step_info.loss))
            if config.checkpoint_print_tensors:
                print_batch_limit = 3

                print('Examples:')
                pp.pprint(step_info.examples[:print_batch_limit])

                print('Expected programs:')
                print(step_info.expected_programs[:print_batch_limit])

                print('Actual programs:')
                print(
                    F.softmax(step_info.actual_programs, dim=2)
                    .transpose(1, 0)[:print_batch_limit, :, :]
                )

            if config.checkpoint_filename is not None:
                print('Saving to file {}'.format(config.checkpoint_filename))
                torch.save(
                    config.model.state_dict(),
                    config.checkpoint_filename)

            print('Done')

        example_idx += 1


def generate_program(batch_size: int) -> List[List[int]]:
    """Generate some simple and short programs for dry-run training."""
    return [
        # Only two programs.
        [0] if random.randint(0, 1) == 0 else [1, 0]
        for _ in range(batch_size)
    ]


def generate_data(
        program_batch: List[List[int]],
        num_examples: int,
        string_size: int) -> List[List[Tuple[List[int], List[int]]]]:
    """
    Generate some input-output data for our simple and short programs
    for dry-run training.

    Batch is a list (batch_size) of tuples (input, output) of
    list (sequence_length) f token indices.
    """
    batch = []
    for program in program_batch:
        examples = []
        for _ in range(num_examples):
            input_sequence = [random.randint(0, string_size-1)]

            # Only two programs here (copy and copy-twice).
            if program == [0]:
                output_sequence = input_sequence
            elif program == [1, 0]:
                output_sequence = input_sequence * 2
            else:
                raise ValueError('Invalid program {}'.format(program))

            examples.append((input_sequence, output_sequence))

        batch.append(examples)

    return batch


def sample_easy(
        batch_size: int,
        string_size: int,
        num_examples: int) -> Tuple[List, List]:
    """
    Sample simple and short programs and example input-output data for
    dry-run training.
    """
    programs = generate_program(batch_size)
    examples = generate_data(programs, num_examples, string_size)
    return programs, examples


def easy_config() -> Config:
    """
    Return config for smaller model on simple and short programs
    as dry-run.
    """
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

    return Config(
        model=robust_fill,
        optimizer=optimizer,
        clip_grad_value=1.0,
        sample=sample,
        device=None,  # CPU training.
        checkpoint_filename=None,
        checkpoint_step_size=100,
        checkpoint_print_tensors=True,
    )


def sample_full(
        token_tables: TokenTables,
        batch_size: int,
        max_expressions: int,
        max_characters: int) -> Tuple[List, List]:
    """Sample a batch of programs and example input-output data."""
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


def full_config() -> Config:
    """
    Return config for full model on programs and example input-output data.
    """
    token_tables = build_token_tables()

    checkpoint_filename = './checkpoint.pth'
    robust_fill = RobustFill(
        string_size=len(op.CHARACTER),
        string_embedding_size=128,
        hidden_size=512,
        program_size=len(token_tables.op_token_table),
    )
    optimizer = optim.SGD(robust_fill.parameters(), lr=0.001)

    def sample():
        return sample_full(
            token_tables,
            batch_size=32,
            max_expressions=10,
            max_characters=50,
        )

    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device `cuda`')
    # Device `mps` doesn't currently work because of Pytorch bugs.
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #    device = torch.device('mps')
    #    print('Using device `mps`')

    return Config(
        model=robust_fill,
        optimizer=optimizer,
        clip_grad_value=1.0,
        sample=sample,
        device=device,
        checkpoint_filename=checkpoint_filename,
        checkpoint_step_size=1,
        checkpoint_print_tensors=False,
    )


def profile_training() -> None:
    """Use PyTorch profiler to profile training step."""
    config = full_config()
    config.model.to(config.device)
    sch = schedule(
        wait=1,
        warmup=1,
        active=3,
    )
    with profile(
            activities=[ProfilerActivity.CUDA],
            schedule=sch,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './profile/log/'),
            with_stack=True,
            record_shapes=True,
            profile_memory=True) as prof:
        for _ in range(10):
            training_step(config)
            prof.step()


def main() -> None:
    """
    Main function responsible for parsing command line arguments and
    invoking model training.
    """
    parser = argparse.ArgumentParser(description='Train RobustFill.')
    parser.add_argument(
        '-m', '--mode',
        choices=['full', 'easy', 'profile'],
        required=True,
        help='Training mode to run in.',
    )
    args = parser.parse_args()

    torch.manual_seed(1337)
    random.seed(420)

    if args.mode == 'full':
        config = full_config()
        train(config)
    elif args.mode == 'easy':
        config = easy_config()
        train(config)
    else:
        profile_training()


if __name__ == '__main__':
    main()
