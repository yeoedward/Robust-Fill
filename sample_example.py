from collections import namedtuple
import logging
import random

from sample_program import sample_program
from to_string import to_string
import operators as op


LOGGER = logging.getLogger(__name__)


Example = namedtuple(
    'Example',
    ['program', 'strings', 'num_discarded_programs'],
)


def sample_string(program, max_characters):
    num_characters = random.randint(1, max_characters)
    random_string = ''.join(random.choices(op.CHARACTER, k=num_characters))
    return random_string


def sample_example(
        *,
        max_expressions=10,
        max_characters=100,
        max_empty_strings=2,
        num_strings=4,
        discard_program_num_empty=100,
        discard_program_num_exceptions=100):
    num_discarded = 0
    while True:
        program = sample_program(max_expressions)

        num_empty, num_exception = 0, 0
        sampled_strings = []

        while True:
            string = sample_string(program, max_characters)
            try:
                transformed = program.eval(string)

                assert isinstance(transformed, str)

                if len(transformed) == 0:
                    num_empty += 1
                if num_empty < max_empty_strings:
                    sampled_strings.append((string, transformed))

            except IndexError:
                num_exception += 1

            if len(sampled_strings) == num_strings:
                return Example(program, sampled_strings, num_discarded)

            if (num_empty > discard_program_num_empty
                    or num_exception > discard_program_num_exceptions):
                LOGGER.debug('Throwing program away')
                LOGGER.debug(
                    'Empty: %s, exception: %s',
                    num_empty,
                    num_exception,
                )
                LOGGER.debug(to_string(program))
                num_discarded += 1
                break
