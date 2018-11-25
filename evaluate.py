import re

import operators as op


def evaluate(exp, value):
    if isinstance(exp, op.Concat):
        return ''.join([
            evaluate(e, value)
            for e in exp.expressions
        ])

    if isinstance(exp, op.ApplyNesting):
        return evaluate(exp.nesting1, evaluate(exp.nesting2, value))

    if isinstance(exp, op.ApplySubstring):
        return evaluate(exp.nesting, evaluate(exp.substring, value))

    if isinstance(exp, op.ConstStr):
        return exp.char

    if isinstance(exp, op.SubStr):
        p1 = substr_index(exp.pos1, value)
        p2 = substr_index(exp.pos2, value)
        return value[p1:p2+1]

    if isinstance(exp, op.GetSpan):
        p1 = span_index(
            dsl_regex=exp.dsl_regex1,
            index=exp.index1,
            bound=exp.bound1,
            value=value,
        )
        p2 = span_index(
            dsl_regex=exp.dsl_regex2,
            index=exp.index2,
            bound=exp.bound2,
            value=value,
        )
        return value[p1:p2]

    if isinstance(exp, op.GetToken):
        matches = match_type(exp.type_, value)
        i = exp.index
        if exp.index > 0:
            # Positive indices start at 1
            i -= 1
        return matches[i]

    if isinstance(exp, op.ToCase):
        if exp.case == op.Case.PROPER:
            return value.capitalize()

        if exp.case == op.Case.ALL_CAPS:
            return value.upper()

        if exp.case == op.Case.LOWER:
            return value.lower()

        raise ValueError('Invalid case: {}'.format(exp))

    if isinstance(exp, op.Replace):
        return value.replace(exp.delim1, exp.delim2)

    if isinstance(exp, op.Trim):
        return value.strip()

    if isinstance(exp, op.GetUpto):
        matches = match_dsl_regex(exp.dsl_regex, value)

        if len(matches) == 0:
            return ''

        first = matches[0]
        return value[:first[1]]

    if isinstance(exp, op.GetFrom):
        matches = match_dsl_regex(exp.dsl_regex, value)

        if len(matches) == 0:
            return ''

        first = matches[0]
        return value[first[1]:]

    if isinstance(exp, op.GetFirst):
        matches = match_type(exp.type_, value)

        if exp.index < 0:
            raise IndexError

        return ''.join(matches[:exp.index])

    if isinstance(exp, op.GetAll):
        return ' '.join(match_type(exp.type_, value))

    raise ValueError('Unsupported operator: {}'.format(exp))


def substr_index(pos, value):
    p = pos if pos > 0 else len(value) + pos

    # Positive indices start at 1, so we need to substract 1
    if p > 0:
        p -= 1

    # If negative, we don't want it to wrap around.
    if p < 0:
        p = 0

    return p


# By convention, we always prefix the DSL regex with 'dsl_' as a way to
# distinguish it with regular regexes.
def span_index(dsl_regex, index, bound, value):
    matches = match_dsl_regex(dsl_regex, value)
    # Positive indices start at 1, so we need to substract 1
    index = index if index < 0 else index - 1

    if index >= len(matches):
        return len(matches) - 1

    if index < -len(matches):
        return 0

    span = matches[index]
    return span[0] if bound == op.Boundary.START else span[1]


def match_dsl_regex(dsl_regex, value):
    if isinstance(dsl_regex, op.Type):
        regex = regex_for_type(dsl_regex)
    else:
        assert len(dsl_regex) == 1 and dsl_regex in op.DELIMITER
        regex = '[' + re.escape(dsl_regex) + ']'

    return [
        match.span()
        for match in re.finditer(regex, value)
    ]


def match_type(type_, value):
    regex = regex_for_type(type_)
    return re.findall(regex, value)


def regex_for_type(type_):
    if type_ == op.Type.NUMBER:
        return '[0-9]+'

    if type_ == op.Type.WORD:
        return '[A-Za-z]+'

    if type_ == op.Type.ALPHANUM:
        return '[A-Za-z0-9]+'

    if type_ == op.Type.ALL_CAPS:
        return '[A-Z]+'

    if type_ == op.Type.PROP_CASE:
        return '[A-Z][a-z]+'

    if type_ == op.Type.LOWER:
        return '[a-z]+'

    if type_ == op.Type.DIGIT:
        return '[0-9]'

    if type_ == op.Type.CHAR:
        return '[A-Za-z0-9]'

    raise ValueError('Unsupported type: {}'.format(type_))
