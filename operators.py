from enum import Enum
from string import ascii_letters, digits, punctuation, whitespace


# Inclusive-inclusive interval
POSITION = [-100, 100]
# 0 is intentionally missing
INDEX = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
DELIMITER = punctuation + whitespace
# Should this be same as Type.CHAR?
CHARACTER = ''.join([ascii_letters, digits, DELIMITER])


class Program(object):
    pass


class Concat(Program):
    def __init__(self, *args):
        self.expressions = args


class Expression(object):
    pass


class Substring(Expression):
    pass


class Nesting(Expression):
    pass


class Apply(Nesting):
    def __init__(self, nesting, nesting_or_substring):
        self.nesting = nesting
        self.nesting_or_substring = nesting_or_substring


class ConstStr(Expression):
    def __init__(self, char):
        self.char = char


class SubStr(Substring):
    def __init__(self, pos1, pos2):
        self.pos1 = pos1
        self.pos2 = pos2


class GetSpan(Substring):
    def __init__(self, dsl_regex1, index1, bound1, dsl_regex2, index2, bound2):
        self.dsl_regex1 = dsl_regex1
        self.index1 = index1
        self.bound1 = bound1
        self.dsl_regex2 = dsl_regex2
        self.index2 = index2
        self.bound2 = bound2


class GetToken(Nesting):
    def __init__(self, type_, index):
        self.type_ = type_
        self.index = index


class ToCase(Nesting):
    def __init__(self, case):
        self.case = case


class Replace(Nesting):
    def __init__(self, delim1, delim2):
        self.delim1 = delim1
        self.delim2 = delim2


class Trim(Nesting):
    pass


class GetUpto(Nesting):
    def __init__(self, dsl_regex):
        self.dsl_regex = dsl_regex


class GetFrom(Nesting):
    def __init__(self, dsl_regex):
        self.dsl_regex = dsl_regex


class GetFirst(Nesting):
    def __init__(self, type_, index):
        self.type_ = type_
        self.index = index


class GetAll(Nesting):
    def __init__(self, type_):
        self.type_ = type_


class Type(Enum):
    NUMBER = 1
    WORD = 2
    ALPHANUM = 3
    ALL_CAPS = 4
    PROP_CASE = 5
    LOWER = 6
    DIGIT = 7
    CHAR = 8


class Case(Enum):
    PROPER = 1
    ALL_CAPS = 2
    LOWER = 3


class Boundary(Enum):
    START = 1
    END = 2
