from collections import namedtuple

import operators as op


# Special token for end-of-sequence
EOS = 'EOS'
TokenTables = namedtuple(
    'TokenTables',
    ['token_op_table', 'op_token_table', 'string_token_table'],
)


def tokenize_string(string, string_token_table):
    return [
        string_token_table[char]
        for char in string
    ]


def build_token_tables():
    token_op_table = [
        EOS,
        op.Concat,
        op.Compose,
        op.ConstStr,
        op.SubStr,
        op.GetSpan,
        op.Trim,
    ]

    # Nesting operators and their args get "compacted" into "primitive" tokens

    for type_ in op.Type:
        for index in op.INDEX:
            token_op_table.append((op.GetToken, type_, index))

    for case in op.Case:
        token_op_table.append((op.ToCase, case))

    for delim1 in op.DELIMITER:
        for delim2 in op.DELIMITER:
            token_op_table.append((op.Replace, delim1, delim2))

    for dsl_regex in list(op.Type) + list(op.DELIMITER):
        token_op_table.append((op.GetUpto, dsl_regex))

    for dsl_regex in list(op.Type) + list(op.DELIMITER):
        token_op_table.append((op.GetFrom, dsl_regex))

    for type_ in op.Type:
        for index in op.INDEX:
            token_op_table.append((op.GetFirst, type_, index))

    for type_ in op.Type:
        token_op_table.append((op.GetAll, type_))

    # Primitive types

    for type_ in op.Type:
        token_op_table.append(type_)

    for case in op.Case:
        token_op_table.append(case)

    for boundary in op.Boundary:
        token_op_table.append(boundary)

    # Covers op.INDEX
    for position in range(op.POSITION[0], op.POSITION[1]+1):
        token_op_table.append(position)

    # This covers op.DELIMITER
    for character in op.CHARACTER:
        token_op_table.append(character)

    token_op_table = {
        token: op
        for token, op in enumerate(token_op_table)
    }

    op_token_table = {
        op: token
        for token, op in token_op_table.items()
    }

    assert len(token_op_table) == len(op_token_table)

    string_token_table = {
        char: token
        for token, char in enumerate(op.CHARACTER)
    }

    return TokenTables(
        token_op_table=token_op_table,
        op_token_table=op_token_table,
        string_token_table=string_token_table,
    )
