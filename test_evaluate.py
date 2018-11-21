from unittest import TestCase

import ast
from evaluate import evaluate


class TestEvaluate(TestCase):
    def test_SubStr(self):
        self.assertEqual('123', evaluate(ast.SubStr(1, 3), '1234'))
        self.assertEqual('4', evaluate(ast.SubStr(0, 4), '1234'))
        self.assertEqual('234', evaluate(ast.SubStr(-2, 4), '1234'))
        try:
            evaluate(ast.SubStr(2, 5), '1234')
            self.fail()
        except IndexError:
            pass
        try:
            evaluate(ast.SubStr(-4, 3), '1234')
            self.fail()
        except IndexError:
            pass

    def test_ToCase(self):
        self.assertEqual(
            'Abc def',
            evaluate(ast.ToCase(ast.Case.PROPER), 'aBc DeF'),
        )
        self.assertEqual(
            'ABC DEF',
            evaluate(ast.ToCase(ast.Case.ALL_CAPS), 'aBc DeF'),
        )
        self.assertEqual(
            'abc def',
            evaluate(ast.ToCase(ast.Case.LOWER), 'aBc DeF'),
        )
