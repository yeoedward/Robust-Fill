from unittest import TestCase

import operators as op
from evaluate import evaluate


class TestEvaluate(TestCase):
    def test_ConstStr(self):
        self.assertEqual('c', evaluate(op.ConstStr('c'), 'ignored'))

    def test_SubStr(self):
        self.assertEqual('123', evaluate(op.SubStr(1, 3), '1234'))
        self.assertEqual('1234', evaluate(op.SubStr(0, 4), '1234'))
        self.assertEqual('34', evaluate(op.SubStr(-2, 4), '1234'))
        self.assertEqual('234', evaluate(op.SubStr(2, 5), '1234'))
        self.assertEqual('123', evaluate(op.SubStr(-5, 3), '1234'))
        self.assertEqual('2', evaluate(op.SubStr(2, 2), '1234'))
        self.assertEqual('3', evaluate(op.SubStr(-2, -2), '1234'))
        self.assertEqual('', evaluate(op.SubStr(3, 2), '1234'))
        self.assertEqual('23', evaluate(op.SubStr(2, -2), '1234'))
        self.assertEqual('234', evaluate(op.SubStr(2, -1), '1234'))
        self.assertEqual('', evaluate(op.SubStr(2, 0), '1234'))
        self.assertEqual('', evaluate(op.SubStr(-2, -3), '1234'))

    def test_GetSpan(self):
        self.assertEqual(
            '123 abcd',
            evaluate(
                op.GetSpan(
                    dsl_regex1=op.Type.NUMBER,
                    index1=1,
                    bound1=op.Boundary.START,
                    dsl_regex2=' ',
                    index2=2,
                    bound2=op.Boundary.START,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            ' abcd',
            evaluate(
                op.GetSpan(
                    dsl_regex1=op.Type.NUMBER,
                    index1=1,
                    bound1=op.Boundary.END,
                    dsl_regex2=' ',
                    index2=2,
                    bound2=op.Boundary.START,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            '123 abcd ',
            evaluate(
                op.GetSpan(
                    dsl_regex1=op.Type.NUMBER,
                    index1=1,
                    bound1=op.Boundary.START,
                    dsl_regex2=' ',
                    index2=2,
                    bound2=op.Boundary.END,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            ' abcd ',
            evaluate(
                op.GetSpan(
                    dsl_regex1=op.Type.NUMBER,
                    index1=1,
                    bound1=op.Boundary.END,
                    dsl_regex2=' ',
                    index2=2,
                    bound2=op.Boundary.END,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            ' abcd ',
            evaluate(
                op.GetSpan(
                    dsl_regex1=op.Type.NUMBER,
                    index1=-2,
                    bound1=op.Boundary.END,
                    dsl_regex2=' ',
                    index2=2,
                    bound2=op.Boundary.END,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            ' abcd ',
            evaluate(
                op.GetSpan(
                    dsl_regex1=op.Type.NUMBER,
                    index1=1,
                    bound1=op.Boundary.END,
                    dsl_regex2=' ',
                    index2=-1,
                    bound2=op.Boundary.END,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            '',
            evaluate(
                op.GetSpan(
                    dsl_regex1=op.Type.NUMBER,
                    index1=2,
                    bound1=op.Boundary.END,
                    dsl_regex2=' ',
                    index2=-1,
                    bound2=op.Boundary.END,
                ),
                '123 abcd 456',
            ),
        )
        self.assertEqual(
            '',
            evaluate(
                op.GetSpan(
                    dsl_regex1=op.Type.NUMBER,
                    index1=1,
                    bound1=op.Boundary.END,
                    dsl_regex2=' ',
                    index2=-3,
                    bound2=op.Boundary.END,
                ),
                '123 abcd 456',
            ),
        )

    def test_GetToken(self):
        self.assertEqual(
            '456',
            evaluate(op.GetToken(op.Type.NUMBER, 2), '123 abc 456'),
        )
        self.assertEqual(
            '999',
            evaluate(op.GetToken(op.Type.NUMBER, 2), '123 abc999 456'),
        )

        try:
            evaluate(op.GetToken(op.Type.NUMBER, 3), '123 abc 456')
            self.fail()
        except IndexError:
            pass

        self.assertEqual(
            'abc',
            evaluate(op.GetToken(op.Type.WORD, 1), '123 abc999 456.hi'),
        )
        self.assertEqual(
            '456',
            evaluate(op.GetToken(op.Type.ALPHANUM, 3), '123 abc999 456.hi'),
        )
        self.assertEqual(
            '123',
            evaluate(op.GetToken(op.Type.ALPHANUM, -4), '123 abc999 456.hi'),
        )
        self.assertEqual(
            'EF',
            evaluate(op.GetToken(op.Type.ALL_CAPS, 2), 'ABC?dEF@GHI'),
        )
        self.assertEqual(
            'Ghi',
            evaluate(op.GetToken(op.Type.PROP_CASE, 2), 'ABC?Def@Ghi'),
        )
        self.assertEqual(
            'ef',
            evaluate(op.GetToken(op.Type.LOWER, 1), 'ABC?Def ghi'),
        )
        self.assertEqual(
            '9',
            evaluate(op.GetToken(op.Type.DIGIT, 4), '123 9'),
        )
        self.assertEqual(
            'c',
            evaluate(op.GetToken(op.Type.CHAR, 3), 'abc999 c'),
        )

        try:
            evaluate(op.GetToken(op.Type.CHAR, -8), 'abc999 c'),
            self.fail()
        except IndexError:
            pass

    def test_ToCase(self):
        self.assertEqual(
            'Abc def',
            evaluate(op.ToCase(op.Case.PROPER), 'aBc DeF'),
        )
        self.assertEqual(
            'ABC DEF',
            evaluate(op.ToCase(op.Case.ALL_CAPS), 'aBc DeF'),
        )
        self.assertEqual(
            'abc def',
            evaluate(op.ToCase(op.Case.LOWER), 'aBc DeF'),
        )

    def test_Replace(self):
        self.assertEqual(
            'abc@def@ghi',
            evaluate(op.Replace('.', '@'), 'abc.def.ghi'),
        )
        self.assertEqual(
            'unchanged',
            evaluate(op.Replace('.', '@'), 'unchanged'),
        )

    def test_Trim(self):
        self.assertEqual(
            'trimmed',
            evaluate(op.Trim(), ' \ttrimmed\n\r'),
        )

    def test_GetUpto(self):
        self.assertEqual(
            'a1',
            evaluate(op.GetUpto(op.Type.NUMBER), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            'a1.b3? 93 !@',
            evaluate(op.GetUpto('@'), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            '',
            evaluate(op.GetUpto('#'), 'a1.b3? 93 !@4'),
        )

    def test_GetFrom(self):
        self.assertEqual(
            '.b3? 93 !@4',
            evaluate(op.GetFrom(op.Type.NUMBER), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            '4',
            evaluate(op.GetFrom('@'), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            '',
            evaluate(op.GetFrom('#'), 'a1.b3? 93 !@4'),
        )

    def test_GetFirst(self):
        self.assertEqual(
            'a1b393',
            evaluate(op.GetFirst(op.Type.ALPHANUM, 3), 'a1.b3? 93 !@4'),
        )

        self.assertEqual(
            '13',
            evaluate(op.GetFirst(op.Type.NUMBER, 2), 'a1.b3? 93 !@4'),
        )

        self.assertEqual(
            '13934',
            evaluate(op.GetFirst(op.Type.NUMBER, 5), 'a1.b3? 93 !@4'),
        )

        try:
            evaluate(op.GetFirst(op.Type.NUMBER, -1), 'a1.b3? 93 !@4'),
            self.fail()
        except IndexError:
            pass

    def test_GetAll(self):
        self.assertEqual(
            'a1 b3 93 4',
            evaluate(op.GetAll(op.Type.ALPHANUM), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            'a b',
            evaluate(op.GetAll(op.Type.LOWER), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            '1 3 93 4',
            evaluate(op.GetAll(op.Type.NUMBER), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            '1 3 9 3 4',
            evaluate(op.GetAll(op.Type.DIGIT), 'a1.b3? 93 !@4'),
        )
        self.assertEqual(
            'Abc Def Ghi',
            evaluate(op.GetAll(op.Type.PROP_CASE), 'AbcDef#!asd Ghi'),
        )

    def test_reference1(self):
        program = op.Concat(
            op.GetToken(op.Type.ALPHANUM, 3),
            op.GetFrom(':'),
            op.GetFirst(op.Type.CHAR, 4),
        )

        self.assertEqual(
            '2525,JV3 ObbUd92',
            evaluate(program, 'Ud 9:25,JV3 Obb'),
        )
        self.assertEqual(
            '843 A44qzLny',
            evaluate(program, 'zLny xmHg 8:43 A44q'),
        )
        self.assertEqual(
            '1063 JfA6g4',
            evaluate(program, 'A6 g45P 10:63 Jf'),
        )
        self.assertEqual(
            'dDX31cuLz',
            evaluate(program, 'cuL.zF.dDX,12:31'),
        )
        self.assertEqual(
            'bj3u11ZiGO',
            evaluate(program, 'ZiG OE bj3u 7:11'),
        )

    # The examples provided in the paper are wrong!
    # Some expected results have letters that don't even appear in
    # the source. I've changed some of the expected results so that
    # they "make sense".
    def test_reference2(self):
        get_span = op.GetSpan(
            dsl_regex1=op.Type.WORD,
            index1=1,
            bound1=op.Boundary.START,
            dsl_regex2='(',
            index2=5,
            bound2=op.Boundary.START,
        )
        program = op.Concat(
            op.Compose(op.GetToken(op.Type.WORD, -1), get_span),
            op.GetToken(op.Type.NUMBER, -5),
            op.GetAll(op.Type.PROP_CASE),
            op.SubStr(-24, -14),
            op.GetToken(op.Type.ALPHANUM, -2),
        )

        self.assertEqual(
            'Qagimg4Kw Sr Vf QagimgxVf )4 )8 QQagimg',
            evaluate(
                program,
                '4 Kw ( )SrK (11 (3 CHA xVf )4 )8 Qagimg ) ()(vs',
            ),
        )
        self.assertEqual(
            'Hpprsqr8Zs Zk HpprsqrZk.6 (E4w )2',
            evaluate(
                program,
                'iY) )hspA.5 ( )8,ZsLL (nZk.6 (E4w )2(Hpprsqr)2(Z',
            ),
        )
        self.assertEqual(
            'hz1005Cqg Hadj Tqwpaxft0 Hadj )zgT5',
            evaluate(
                program,
                'Cqg) ) ( (1005 ( ( )VCE hz ) (10 Hadj )zgTqwpaxft-7 5 6',
            ),
        )
        self.assertEqual(
            'lU7Jv Ihitux Fl7 XLTD sfs 6',
            evaluate(
                program,
                'JvY) (Ihitux ) ) ( (6 SFl (7 XLTD sfs ))11,lU7 (6 9',
            ),
        )
        self.assertEqual(
            'DXR4Njt Pu Ztje )6 aX 4 )D6',
            evaluate(
                program,
                'NjtT(D7QV (4 (yPuY )8.sa ( ) )6 aX 4 )DXR (@6 ) Ztje',
            ),
        )

    def test_reference3(self):
        get_span = op.GetSpan(
            dsl_regex1=op.Type.ALL_CAPS,
            index1=1,
            bound1=op.Boundary.START,
            dsl_regex2=op.Type.ALL_CAPS,
            index2=5,
            bound2=op.Boundary.START,
        )
        program = op.Compose(op.GetToken(op.Type.ALL_CAPS, -2), get_span)

        self.assertEqual(
            'W',
            evaluate(program, 'YDXJZ @ZYUD Wc-YKT GTIL BNX'),
        )
        self.assertEqual(
            'MTHV',
            evaluate(program, 'JUGRB.MPKA.MTHV,tEczT-GZJ.MFT'),
        )
        self.assertEqual(
            'JC',
            evaluate(program, 'VXO.OMQDK.JC-OAR,HZGH-DJKC'),
        )
        self.assertEqual(
            'RTTRQ',
            evaluate(program, 'HCUD-WDOC,RTTRQ-KVETK-whx-DIKDI'),
        )
        self.assertEqual(
            'ODZBT',
            evaluate(program, 'JFNB.Avj,ODZBT-XHV,KYB @,RHVVW'),
        )

    def test_reference4(self):
        program = op.Concat(
            op.SubStr(-20, -8),
            op.GetToken(op.Type.ALL_CAPS, -3),
            op.SubStr(11, 19),
            op.GetToken(op.Type.ALPHANUM, -5),
        )

        self.assertEqual(
            'IN ZZUK,nCF aCF6 OZQIN ZOZQIN',
            evaluate(program, 'DvD 6X xkd6 OZQIN ZZUK,nCF aQR IOHR'),
        )

        self.assertEqual(
            'CRCUC,ONFZA.mONFZAy,44-CRCU44',
            evaluate(program, 'BHP-euSZ,yy,44-CRCUC,ONFZA.mgOJ.Hwm'),
        )

        self.assertEqual(
            ',CMFEX-JPFA,iCMFEXrL.GmOc.PPFLH',
            evaluate(program, 'NGM-8nay,xrL.GmOc.PFLH,CMFEX-JPFA,iIcj,329'),
        )

        self.assertEqual(
            ' NCPYJ oo FS FSycb NCPYJNCPYJ',
            evaluate(program, 'hU TQFLD Lycb NCPYJ oo FS TUM l6F'),
        )

        self.assertEqual(
            'L 8Ucj dUqh CUXKQRN KDLKDL',
            evaluate(program, 'OHHS NNDQ XKQRN KDL 8Ucj dUqh Cpk Kafj'),
        )
