from unittest import TestCase

import ast
from evaluate import evaluate


class TestEvaluate(TestCase):
    def test_ConstStr(self):
        self.assertEqual('c', evaluate(ast.ConstStr('c'), 'ignored'))

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

    def test_GetToken(self):
        self.assertEqual(
            '456',
            evaluate(ast.GetToken(ast.Type.NUMBER, 2), '123 abc 456'),
        )
        self.assertEqual(
            '456',
            evaluate(ast.GetToken(ast.Type.NUMBER, 2), '123 abc999 456'),
        )

        try:
            evaluate(ast.GetToken(ast.Type.NUMBER, 3), '123 abc 456')
            self.fail()
        except IndexError:
            pass

        self.assertEqual(
            'hi',
            evaluate(ast.GetToken(ast.Type.WORD, 1), '123 abc999 456.hi'),
        )
        self.assertEqual(
            '456',
            evaluate(ast.GetToken(ast.Type.ALPHANUM, 3), '123 abc999 456.hi'),
        )
        self.assertEqual(
            '123',
            evaluate(ast.GetToken(ast.Type.ALPHANUM, -4), '123 abc999 456.hi'),
        )
        self.assertEqual(
            'GHI',
            evaluate(ast.GetToken(ast.Type.ALL_CAPS, 2), 'ABC?dEF@GHI'),
        )
        self.assertEqual(
            'Ghi',
            evaluate(ast.GetToken(ast.Type.PROP_CASE, 2), 'ABC?Def@Ghi'),
        )
        self.assertEqual(
            'ghi',
            evaluate(ast.GetToken(ast.Type.LOWER, 1), 'ABC?Def ghi'),
        )
        self.assertEqual(
            '9',
            evaluate(ast.GetToken(ast.Type.DIGIT, 1), '123 9'),
        )
        self.assertEqual(
            'c',
            evaluate(ast.GetToken(ast.Type.CHAR, 1), 'abc999 c'),
        )

        try:
            evaluate(ast.GetToken(ast.Type.CHAR, -2), 'abc999 c'),
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

    def test_Replace(self):
        self.assertEqual(
            'abc@def@ghi',
            evaluate(ast.Replace('.', '@'), 'abc.def.ghi'),
        )
        self.assertEqual(
            'unchanged',
            evaluate(ast.Replace('.', '@'), 'unchanged'),
        )

    def test_Trim(self):
        self.assertEqual(
            'trimmed',
            evaluate(ast.Trim(), ' \ttrimmed\n\r'),
        )
