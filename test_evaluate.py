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
        self.assertEqual('234', evaluate(ast.SubStr(2, 5), '1234'))
        self.assertEqual('123', evaluate(ast.SubStr(-5, 3), '1234'))
        self.assertEqual('2', evaluate(ast.SubStr(2, 2), '1234'))
        self.assertEqual('', evaluate(ast.SubStr(3, 2), '1234'))

    def test_GetSpan(self):
        self.assertEqual(
            '123 abcd',
            evaluate(
                ast.GetSpan(
                    dsl_regex1=ast.Type.NUMBER,
                    index1=1,
                    bound1=ast.Boundary.START,
                    dsl_regex2=' ',
                    index2=2,
                    bound2=ast.Boundary.START,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            ' abcd',
            evaluate(
                ast.GetSpan(
                    dsl_regex1=ast.Type.NUMBER,
                    index1=1,
                    bound1=ast.Boundary.END,
                    dsl_regex2=' ',
                    index2=2,
                    bound2=ast.Boundary.START,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            '123 abcd ',
            evaluate(
                ast.GetSpan(
                    dsl_regex1=ast.Type.NUMBER,
                    index1=1,
                    bound1=ast.Boundary.START,
                    dsl_regex2=' ',
                    index2=2,
                    bound2=ast.Boundary.END,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            ' abcd ',
            evaluate(
                ast.GetSpan(
                    dsl_regex1=ast.Type.NUMBER,
                    index1=1,
                    bound1=ast.Boundary.END,
                    dsl_regex2=' ',
                    index2=2,
                    bound2=ast.Boundary.END,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            ' abcd ',
            evaluate(
                ast.GetSpan(
                    dsl_regex1=ast.Type.NUMBER,
                    index1=-2,
                    bound1=ast.Boundary.END,
                    dsl_regex2=' ',
                    index2=2,
                    bound2=ast.Boundary.END,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            ' abcd ',
            evaluate(
                ast.GetSpan(
                    dsl_regex1=ast.Type.NUMBER,
                    index1=1,
                    bound1=ast.Boundary.END,
                    dsl_regex2=' ',
                    index2=-1,
                    bound2=ast.Boundary.END,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            '',
            evaluate(
                ast.GetSpan(
                    dsl_regex1=ast.Type.NUMBER,
                    index1=2,
                    bound1=ast.Boundary.END,
                    dsl_regex2=' ',
                    index2=-1,
                    bound2=ast.Boundary.END,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            '',
            evaluate(
                ast.GetSpan(
                    dsl_regex1=ast.Type.NUMBER,
                    index1=1,
                    bound1=ast.Boundary.END,
                    dsl_regex2=' ',
                    index2=-3,
                    bound2=ast.Boundary.END,
                ),
                '123 abcd 456',
            ),
        )

    def test_GetToken(self):
        self.assertEqual(
            '456',
            evaluate(ast.GetToken(ast.Type.NUMBER, 2), '123 abc 456'),
        )
        self.assertEqual(
            '999',
            evaluate(ast.GetToken(ast.Type.NUMBER, 2), '123 abc999 456'),
        )

        try:
            evaluate(ast.GetToken(ast.Type.NUMBER, 3), '123 abc 456')
            self.fail()
        except IndexError:
            pass

        self.assertEqual(
            'abc',
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
            'EF',
            evaluate(ast.GetToken(ast.Type.ALL_CAPS, 2), 'ABC?dEF@GHI'),
        )
        self.assertEqual(
            'B',
            evaluate(ast.GetToken(ast.Type.PROP_CASE, 2), 'ABC?Def@Ghi'),
        )
        self.assertEqual(
            'ef',
            evaluate(ast.GetToken(ast.Type.LOWER, 1), 'ABC?Def ghi'),
        )
        self.assertEqual(
            '9',
            evaluate(ast.GetToken(ast.Type.DIGIT, 4), '123 9'),
        )
        self.assertEqual(
            'c',
            evaluate(ast.GetToken(ast.Type.CHAR, 3), 'abc999 c'),
        )

        try:
            evaluate(ast.GetToken(ast.Type.CHAR, -5), 'abc999 c'),
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

    def test_GetUpto(self):
        self.assertEqual(
            'a1',
            evaluate(ast.GetUpto(ast.Type.NUMBER), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            'a1.b3? 93 !@',
            evaluate(ast.GetUpto('@'), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            '',
            evaluate(ast.GetUpto('#'), 'a1.b3? 93 !@4'),
        )

    def test_GetFrom(self):
        self.assertEqual(
            '1.b3? 93 !@4',
            evaluate(ast.GetFrom(ast.Type.NUMBER), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            '@4',
            evaluate(ast.GetFrom('@'), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            '',
            evaluate(ast.GetFrom('#'), 'a1.b3? 93 !@4'),
        )

    def test_GetFirst(self):
        self.assertEqual(
            'a1b393',
            evaluate(ast.GetFirst(ast.Type.ALPHANUM, 3), 'a1.b3? 93 !@4'),
        )

        self.assertEqual(
            '13',
            evaluate(ast.GetFirst(ast.Type.NUMBER, 2), 'a1.b3? 93 !@4'),
        )

        self.assertEqual(
            '13934',
            evaluate(ast.GetFirst(ast.Type.NUMBER, 5), 'a1.b3? 93 !@4'),
        )

        try:
            evaluate(ast.GetFirst(ast.Type.NUMBER, -1), 'a1.b3? 93 !@4'),
            self.fail()
        except IndexError:
            pass

    def test_GetAll(self):
        self.assertEqual(
            'a1b3934',
            evaluate(ast.GetAll(ast.Type.ALPHANUM), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            'ab',
            evaluate(ast.GetAll(ast.Type.LOWER), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            '13934',
            evaluate(ast.GetAll(ast.Type.NUMBER), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            '13934',
            evaluate(ast.GetAll(ast.Type.DIGIT), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            'AbcDefGhi',
            evaluate(ast.GetAll(ast.Type.PROP_CASE), 'AbcDef#!asd Ghi'),
        )
