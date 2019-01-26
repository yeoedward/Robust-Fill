from unittest import TestCase

from sample import sample_program, sample_string
from tokens import build_token_tables


class TestTokens(TestCase):
    def test_total_num_tokens(self):
        token_tables = build_token_tables()

        expected_num_tokens = 1118
        self.assertEqual(expected_num_tokens, len(token_tables.token_op_table))
        self.assertEqual(expected_num_tokens, len(token_tables.op_token_table))

    def test_token_table_coverage_smoke_test(self):
        token_tables = build_token_tables()

        num_samples = 1000
        for _ in range(num_samples):
            sample_program(10).to_tokens(token_tables.op_token_table)
            for char in sample_string(30):
                token_tables.string_token_table[char]
