'''
TODO:
    Eval
    Checking
    Unit tests
'''


from abc import ABC, abstractmethod


class DSL(ABC):
    @abstractmethod
    def eval(v):
        raise NotImplementedError


class Program(DSL):
    def __init__(self, *args):
        self.expressions = args

    def eval(self, v):
        return ''.join([
            e.eval(v)
            for e in self.expressions
        ])


class Expression(DSL):
    pass


class Substring(Expression):
    pass


class Nesting(Expression):
    pass


class ApplyNesting(Nesting):
    def __init__(self, nesting1, nesting2):
        self.nesting1 = nesting1
        self.nesting2 = nesting2


class ApplySubstring(Nesting):
    def __init__(self, nesting, substring):
        self.nesting = nesting
        self.substring = substring


class ConstStr(Expression):
    pass


class SubStr(Substring):
    def __init__(self, pos1, pos2):
        self.pos1 = pos1
        self.pos2 = pos2

    def eval(self, v):
        p1 = self.pos1 if self.pos1 > 0 else len(v) + self.pos1
        p2 = self.pos2 if self.pos2 > 0 else len(v) + self.pos2
        return v[p1-1:p2]


class GetSpan(Substring):
    def __init__(self, regex1, index1, bound1, regex2, index2, bound2):
        self.regex1 = regex1
        self.index1 = index1
        self.bound1 = bound1
        self.regex2 = regex2
        self.index2 = index2
        self.bound2 = bound2

    def eval(self, v):
        raise NotImplementedError


class GetToken(Nesting):
    def __init__(self, type_, index):
        self.type_ = type_
        self.index = index

    def eval(self, v):
        raise NotImplementedError


class ToCase(Nesting):
    def __init__(self, case):
        self.case = case

    def eval(self, v):
        raise NotImplementedError


class Replace(Nesting):
    def __init__(self, delim1, delim2):
        self.delim1 = delim1
        self.delim2 = delim2

    def eval(self, v):
        raise NotImplementedError


class Trim(Nesting):
    pass


class GetUpto(Nesting):
    def __init__(self, regex):
        self.regex = regex


class GetFrom(Nesting):
    def __init__(self, regex):
        self.regex = regex


class GetFirst(Nesting):
    def __init__(self, type_, index):
        self.type_ = type_
        self.index = index


class GetAll(Nesting):
    def __init__(self, type_):
        self.type_ = type_
