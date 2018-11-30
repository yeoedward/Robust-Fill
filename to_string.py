import operators as op


def to_string(exp, indent=0, tab=4):
    if isinstance(exp, op.Concat):
        sub_exps = [
            to_string(e, indent=indent+tab, tab=tab)
            for e in exp.expressions
        ]
        return op_to_string('Concat', sub_exps, indent, recursive=True)

    if isinstance(exp, op.Compose):
        new_indent = indent + tab
        nesting = to_string(exp.nesting, indent=new_indent, tab=tab)
        nesting_or_substring = to_string(
            exp.nesting_or_substring,
            indent=new_indent,
            tab=tab,
        )
        return op_to_string(
            'Compose',
            [nesting, nesting_or_substring],
            indent,
            recursive=True)

    if isinstance(exp, op.ConstStr):
        return op_to_string('ConstStr', [exp.char], indent)

    if isinstance(exp, op.SubStr):
        return op_to_string('SubStr', [exp.pos1, exp.pos2], indent)

    if isinstance(exp, op.GetSpan):
        return op_to_string(
            'GetSpan',
            [
                exp.dsl_regex1,
                exp.index1,
                exp.bound1,
                exp.dsl_regex2,
                exp.index2,
                exp.bound2,
            ],
            indent,
        )

    if isinstance(exp, op.GetToken):
        return op_to_string('GetToken', [exp.type_, exp.index], indent)

    if isinstance(exp, op.ToCase):
        return op_to_string('ToCase', [exp.case], indent)

    if isinstance(exp, op.Replace):
        return op_to_string('Replace', [exp.delim1, exp.delim2], indent)

    if isinstance(exp, op.Trim):
        return op_to_string('Trim', [], indent)

    if isinstance(exp, op.GetUpto):
        return op_to_string('GetUpto', [exp.dsl_regex], indent)

    if isinstance(exp, op.GetFrom):
        return op_to_string('GetFrom', [exp.dsl_regex], indent)

    if isinstance(exp, op.GetFirst):
        return op_to_string('GetFirst', [exp.type_, exp.index], indent)

    if isinstance(exp, op.GetAll):
        return op_to_string('GetAll', [exp.type_], indent)

    raise ValueError('Unsupported operator {}'.format(exp))


def op_to_string(name, raw_args, indent, recursive=False):
    indent_str = ' ' * indent

    if recursive:
        args_str = ',\n'.join(raw_args)
        return '{indent_str}{name}(\n{args_str}\n{indent_str})'.format(
            indent_str=indent_str,
            name=name,
            args_str=args_str,
        )

    args = [repr(a) for a in raw_args]
    args_str = ', '.join(args)
    return '{indent_str}{name}({args_str})'.format(
        indent_str=indent_str,
        name=name,
        args_str=args_str,
    )
