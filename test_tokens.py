from unittest import TestCase

import torch

from sample import sample_program, sample_string
from tokens import Tokenizer


class TestTokens(TestCase):
    def test_total_num_tokens(self):
        tokenizer = Tokenizer.create()

        expected_num_tokens = 1118
        self.assertEqual(expected_num_tokens, len(tokenizer.token_op_table))
        self.assertEqual(expected_num_tokens, len(tokenizer.op_token_table))

    def test_token_table_coverage_smoke_test(self):
        torch.manual_seed(1337)
        tokenizer = Tokenizer.create()

        num_samples = 1000
        for _ in range(num_samples):
            sample_program(10).to_tokens(tokenizer.op_token_table)
            for char in sample_string(30):
                tokenizer.string_token_table[char]

    def test_parsing(self):
        """Test parsing programs from tokens."""
        torch.manual_seed(1337)
        tokenizer = Tokenizer.create()

        num_samples = 1000
        for _ in range(num_samples):
            program = sample_program(10)
            tokens = program.to_tokens(tokenizer.op_token_table)
            parsed = tokenizer.parse_program(tokens)
            self.assertEqual(str(program), str(parsed))
