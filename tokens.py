from typing import Dict, List, Tuple

import operators as op


# Special token for end-of-sequence
EOS = 'EOS'


class Tokenizer:
    """
    Manages mapping between:
    1. String characters and encoder tokens
    2. Program operators and decoder tokens.
    """

    def __init__(
            self,
            token_op_table: Dict[int, op.DSL],
            op_token_table: Dict[op.DSL, int],
            string_token_table: Dict[str, int]) -> None:
        self.token_op_table = token_op_table
        self.op_token_table = op_token_table
        self.string_token_table = string_token_table

    @staticmethod
    def create() -> 'Tokenizer':
        """
        Build tables for converting between integer tokens and
        operators / string characters.
        """
        token_op_table = [
            EOS,
            op.Concat,
            op.Compose,
            op.ConstStr,
            op.SubStr,
            op.GetSpan,
            op.Trim,
        ]

        # Nesting operators and their args get "compacted" into
        # "primitive" tokens

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

        return Tokenizer(
            token_op_table=token_op_table,
            op_token_table=op_token_table,
            string_token_table=string_token_table,
        )

    def tokenize_string(self, string: str) -> int:
        """Convert string into list of integer tokens."""
        return [
            self.string_token_table[char]
            for char in string
        ]

    def _parse_args(
            self,
            tokens: List[int],
            operator: type,
            num_args: int) -> Tuple[op.Expression, int]:
        """
        Parse the given operator that has the specified number of args.

        :param tokens: The tokens to parse.
        :param operator: The operator to construct.
        :param num_args: The number of arguments (i.e. tokens) that
            the operator has.
        :returns: A tuple of the constructed operator and the number
            of tokens consumed.
        """
        if len(tokens[1:]) < num_args:
            return None, 1
        args = [
            self.token_op_table[tok]
            for tok in tokens[1:1+num_args]
        ]
        return operator(*args), 1+num_args

    def _parse_expression(
            self,
            tokens: List[int],
            allow_nesting: bool) -> Tuple[op.Expression, int]:
        """
        Parse an expression.

        :param tokens: The tokens to parse.
        :param allow_nesting: Whether to allow nesting of operators.
        :returns: A tuple of the constructed expression and the number
            of tokens consumed.
        """
        decoded = self.token_op_table[tokens[0]]

        if decoded == EOS:
            return None, 0

        # Nesting.
        if isinstance(decoded, tuple):
            expr = decoded[0](*decoded[1:])
            if allow_nesting:
                sub_expr, n = self._parse_expression(
                    tokens[1:],
                    allow_nesting=False)
                if isinstance(sub_expr, (op.Nesting, op.Substring)):
                    return op.Compose(expr, sub_expr), n + 1
            return expr, 1

        if issubclass(decoded, op.Trim):
            expr = decoded()
            if allow_nesting:
                sub_expr, n = self._parse_expression(
                    tokens[1:],
                    allow_nesting=False)
                if isinstance(sub_expr, (op.Nesting, op.Substring)):
                    return op.Compose(expr, sub_expr), n + 1
            return expr, 1

        # Expression.
        if issubclass(decoded, op.ConstStr):
            return self._parse_args(
                tokens,
                operator=op.ConstStr,
                num_args=1)

        # Substrings.
        if issubclass(decoded, op.SubStr):
            return self._parse_args(
                tokens,
                operator=op.SubStr,
                num_args=2)

        if issubclass(decoded, op.GetSpan):
            return self._parse_args(
                tokens,
                operator=op.GetSpan,
                num_args=6)

        return None, 0

    def parse_program(self, tokens: List[int]) -> op.Program:
        """Parse a program from tokens."""
        expressions = []
        i = 0
        while i < len(tokens):
            expr, n = self._parse_expression(tokens[i:], allow_nesting=True)
            if expr is None:
                if n > 0:
                    # Partial match, so we ignore this expression.
                    break
                raise ValueError(f'Invalid token {tokens[i]}')
            expressions.append(expr)
            i += n

            if i == len(tokens):
                break

            if i > len(tokens):
                raise ValueError('Probably a bug')

            next_op = self.token_op_table[tokens[i]]

            if next_op == EOS:
                break

            if isinstance(next_op, type) and issubclass(next_op, op.Concat):
                i += 1
                continue

            raise ValueError(f'Unexpected op `{next_op}`')

        return op.Concat(*expressions)
