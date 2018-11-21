import ast


def evaluate(exp, value):
    if isinstance(exp, ast.Program):
        return ''.join([
            evaluate(e, value)
            for e in exp.expressions
        ])

    if isinstance(exp, ast.ApplyNesting):
        return evaluate(exp.nesting1, evaluate(exp.nesting2, value))

    if isinstance(exp, ast.ApplySubstring):
        return evaluate(exp.nesting, evaluate(exp.substring, value))

    if isinstance(exp, ast.ConstStr):
        return exp.char

    if isinstance(exp, ast.SubStr):
        p1 = exp.pos1 if exp.pos1 > 0 else len(value) + exp.pos1
        p2 = exp.pos2 if exp.pos2 > 0 else len(value) + exp.pos2
        return value[p1-1:p2]

    if isinstance(exp, ast.GetSpan):
        raise NotImplementedError

    if isinstance(exp, ast.GetToken):
        raise NotImplementedError

    if isinstance(exp, ast.ToCase):
        raise NotImplementedError

    if isinstance(exp, ast.Replace):
        raise NotImplementedError

    if isinstance(exp, ast.Trim):
        raise NotImplementedError

    if isinstance(exp, ast.GetUpto):
        raise NotImplementedError

    if isinstance(exp, ast.GetFrom):
        raise NotImplementedError

    if isinstance(exp, ast.GetFirst):
        raise NotImplementedError

    if isinstance(exp, ast.GetAll):
        raise NotImplementedError

    raise ValueError('Unsupported operator: {}'.format(exp))
