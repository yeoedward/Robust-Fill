from collections import namedtuple
import logging
import random

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
                    if num_empty <= max_empty_strings:
                        sampled_strings.append((string, transformed))
                else:
                    sampled_strings.append((string, transformed))

            except IndexError:
                num_exception += 1

            if len(sampled_strings) == num_strings:
                return Example(program, sampled_strings, num_discarded)

            # We have to throw programs away because some of them always
            # throw IndexError's or produce empty strings.
            if (num_empty > discard_program_num_empty
                    or num_exception > discard_program_num_exceptions):
                LOGGER.debug('Throwing program away')
                LOGGER.debug(
                    'Empty: %s, exception: %s',
                    num_empty,
                    num_exception,
                )
                LOGGER.debug(program)
                num_discarded += 1
                break


def sample_program(max_expressions):
    num_expressions = random.randint(1, max_expressions)
    return op.Concat(*[
        sample_expression()
        for _ in range(num_expressions)
    ])


def sample_from(*samplers):
    choice = random.choice(samplers)
    return choice()


def sample_expression():
    return sample_from(
        sample_substring,
        sample_nesting,
        sample_Compose,
        sample_ConstStr,
    )


def sample_substring():
    return sample_from(
        sample_SubStr,
        sample_GetSpan,
    )


def sample_nesting():
    return sample_from(
        sample_GetToken,
        sample_ToCase,
        sample_Replace,
        sample_Trim,
        sample_GetUpto,
        sample_GetFrom,
        sample_GetFirst,
        sample_GetAll,
    )


def sample_Compose():
    nesting = sample_nesting()
    nesting_or_substring = sample_from(
        sample_nesting,
        sample_substring,
    )
    return op.Compose(nesting, nesting_or_substring)


def sample_ConstStr():
    char = random.choice(op.CHARACTER)
    return op.ConstStr(char)


def sample_position():
    return random.randint(*op.POSITION)


def sample_SubStr():
    pos1 = sample_position()
    pos2 = sample_position()
    return op.SubStr(pos1, pos2)


def sample_Boundary():
    return random.choice(list(op.Boundary))


def sample_GetSpan():
    return op.GetSpan(
        dsl_regex1=sample_dsl_regex(),
        index1=sample_index(),
        bound1=sample_Boundary(),
        dsl_regex2=sample_dsl_regex(),
        index2=sample_index(),
        bound2=sample_Boundary(),
    )


def sample_Type():
    return random.choice(list(op.Type))


def sample_index():
    return random.choice(op.INDEX)


def sample_GetToken():
    type_ = sample_Type()
    index = sample_index()
    return op.GetToken(type_, index)


def sample_ToCase():
    case = random.choice(list(op.Case))
    return op.ToCase(case)


def sample_delimiter():
    return random.choice(op.DELIMITER)


def sample_Replace():
    delim1 = sample_delimiter()
    delim2 = sample_delimiter()
    return op.Replace(delim1, delim2)


def sample_Trim():
    return op.Trim()


def sample_dsl_regex():
    return random.choice(list(op.Type) + list(op.DELIMITER))


def sample_GetUpto():
    dsl_regex = sample_dsl_regex()
    return op.GetUpto(dsl_regex)


def sample_GetFrom():
    dsl_regex = sample_dsl_regex()
    return op.GetFrom(dsl_regex)


def sample_GetFirst():
    type_ = sample_Type()
    index = random.choice([
        i for i in op.INDEX
        if i > 0
    ])
    return op.GetFirst(type_, index)


def sample_GetAll():
    type_ = sample_Type()
    return op.GetAll(type_)
