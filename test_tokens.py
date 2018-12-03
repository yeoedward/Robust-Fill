from unittest import TestCase

from tokens import build_token_tables


class TestTokens(TestCase):
    def test_total_num_tokens(self):
        token_tables = build_token_tables()

        expected_num_tokens = 1118
        self.assertEqual(expected_num_tokens, len(token_tables.token_op_table))
        self.assertEqual(expected_num_tokens, len(token_tables.op_token_table))
