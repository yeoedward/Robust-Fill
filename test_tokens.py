from unittest import TestCase

import torch

from sample import sample_program, sample_string
from tokens import Tokenizer


class TestTokens(TestCase):
    def test_total_num_tokens(self):
        """
        This test makes sure we don't unintentionally change
        the number of tokens.
        """
        tokenizer = Tokenizer.create()

        expected_num_tokens = 538
        self.assertEqual(expected_num_tokens, len(tokenizer.token_op_table))
        self.assertEqual(expected_num_tokens, len(tokenizer.op_token_table))

    def test_token_table_coverage_smoke_test(self):
        torch.manual_seed(1337)
        tokenizer = Tokenizer.create()

        num_samples = 1000
        for _ in range(num_samples):
            prog, h = sample_program(10)
            prog.to_tokens(tokenizer.op_token_table)
            for char in sample_string(32, h):
                tokenizer.string_token_table[char]

    def test_parsing(self):
        """Test parsing programs from tokens."""
        torch.manual_seed(1337)
        tokenizer = Tokenizer.create()

        num_samples = 1000
        for _ in range(num_samples):
            program, _ = sample_program(10)
            tokens = program.to_tokens(tokenizer.op_token_table)
            parsed = tokenizer.parse_program(tokens)
            self.assertEqual(str(program), str(parsed))
