from string import ascii_letters, ascii_lowercase, ascii_uppercase, digits
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
        if p1 - 1 < 0 or p2 > len(value):
            raise IndexError
        return value[p1-1:p2]

    if isinstance(exp, ast.GetSpan):
        raise NotImplementedError

    if isinstance(exp, ast.GetToken):
        raise NotImplementedError

    if isinstance(exp, ast.ToCase):
        if exp.case == ast.Case.PROPER:
            return value.capitalize()

        if exp.case == ast.Case.ALL_CAPS:
            return value.upper()

        if exp.case == ast.Case.LOWER:
            return value.lower()

        raise ValueError('Invalid case: {}'.format(exp))

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


def match(type_, value):
    if value == '':
        raise ValueError('Non-empty value required')

    if type_ == ast.Type.NUMBER:
        return all([v in digits for v in value])

    if type_ == ast.Type.WORD:
        return all([v in ascii_letters for v in value])

    if type_ == ast.Type.ALPHANUM:
        alpha_num = ascii_letters + digits
        return all([v in alpha_num for v in value])

    if type_ == ast.Type.ALL_CAPS:
        return all([v in ascii_uppercase for v in value])

    if type_ == ast.Type.PROP_CASE:
        return (value[0] in ascii_uppercase
                and all([v in ascii_lowercase for v in value[1:]]))

    if type_ == ast.Type.LOWER:
        return all([v in ascii_lowercase for v in value])

    if type_ == ast.Type.DIGIT:
        return len(value) == 1 and value in digits

    if type_ == ast.Type.CHAR:
        return len(value) == 1 and value in ascii_letters

    raise ValueError('Unsupported type: {}'.format(type_))
