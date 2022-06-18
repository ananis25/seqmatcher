"""
This module parses the input string into a match/replace instruction. 
"""

from typing import Optional
import parsy as p
import re

from .primitives import *
from .codegen import vet_custom_code, rewrite_custom_code


def generate(name: str):
    """Tags the parser generated by `parsy` with a name; appease the type-checker."""

    def decorator(fn):
        return p.generate(fn).desc(name)

    return decorator


# read a positive integer value
integer = p.digit.at_least(1).concat().map(int).desc("number")


@generate("slice")
def slice_():
    """read a range value written as `start..end`"""
    st = yield integer.optional()
    yield p.string("..")
    end = yield integer.optional()
    return st, end


# read either a range or a number, bracketed
count = p.string("[") >> (slice_ | integer) << p.string("]")


@generate("property")
def prop():
    key_str = yield p.regex(r"[\w_]+")
    op_str = yield p.regex(r"=|!=|<=|>=|<|>").desc("operator")
    val_str = yield p.regex(r"@?[\w_]+").sep_by(p.string("|"), min=1)

    return Property(Operator(op_str), key_str, val_str)


@generate("event")
def event():
    evt = EvtPattern()

    # read the number of times it should appear
    number = yield count.optional()
    if number is None:
        evt.min_count = 1
        evt.max_count = 1
    elif isinstance(number, int):
        evt.min_count = number
        evt.max_count = number
    else:
        evt.min_count = number[0] if number[0] is not None else 0
        if number[1] is not None:
            evt.max_count = number[1]
            assert evt.max_count >= evt.min_count
            assert evt.max_count > 0
        else:
            evt.max_count = 1000000  # arbitrary large integer

    yield p.string("(")
    # read the custom name for the event
    custom_name = yield p.regex(r"[@\w]+").desc("custom_name").optional()
    if custom_name is not None:
        evt.custom_name = custom_name

    # read the event names to include/exclude
    exclude = yield p.string("!").optional()
    names = yield (p.string(":") >> p.regex(r"@?\w+").desc("type")).sep_by(
        p.string("|")
    )
    if len(names) > 0:
        _op = Operator.EQ if exclude is None else Operator.NE
        evt.properties.append(Property(_op, "_eventName", names))

    # read the properties to match
    tmp = yield p.string("{").optional()
    if tmp is not None:
        properties = yield prop.sep_by(p.string(","))
        evt.properties.extend(properties)
        yield p.string("}")

    yield p.string(")")
    return evt


@generate("pattern")
def seq_pattern():
    pat = SeqPattern()

    tmp = yield p.string("|-").optional()
    if tmp is not None:
        pat.match_seq_start = True

    start_mark = p.string("^")
    end_mark = p.string("$")
    token = start_mark | end_mark | event

    tokens = yield token.sep_by(p.string("-"))
    for tok in tokens:
        if tok == "^":
            if pat.idx_start_event is not None:
                raise Exception("multiple start event markers in the pattern")
            pat.idx_start_event = len(pat.event_patterns)
        elif tok == "$":
            if pat.idx_end_event is not None:
                raise Exception("multiple end event markers in the pattern")
            pat.idx_end_event = len(pat.event_patterns)
        else:
            pat.event_patterns.append(tok)

    tmp = yield p.string("-|").optional()
    if tmp is not None:
        pat.match_seq_end = True

    # read the properties to match
    tmp = yield p.string("{{").optional()
    if tmp is not None:
        properties: list[Property] = yield prop.sep_by(p.string(","))
        for _prop in properties:
            for _val in _prop.values:
                assert isinstance(_val, str)
            if _prop.key == "_match_all":
                assert _prop.op == Operator.EQ
                pat.match_all = True if _prop.values[0].lower() == "true" else False
            elif _prop.key == "_allow_overlaps":
                assert _prop.op == Operator.EQ
                pat.allow_overlaps = (
                    True if _prop.values[0].lower() == "true" else False
                )
            else:
                pat.properties.append(_prop)
        yield p.string("}}")

    return pat


def parse_match_pattern(pattern_str: str, code_str: Optional[str] = None) -> SeqPattern:
    """Parse a sequence pattern string into a SeqPattern object."""
    pattern_str = pattern_str.replace(" ", "")
    pat: SeqPattern = seq_pattern.parse(pattern_str)

    # fill in the cross references made across events
    for i, evt in enumerate(pat.event_patterns):
        if evt.custom_name is not None:
            assert evt.custom_name.startswith(
                "@"
            ), f"custom name for events must start with `@`: {evt.custom_name}"
            assert (
                evt.custom_name not in pat.custom_names
            ), f"duplicate assignment for custom name: {evt.custom_name}"
            pat.custom_names[evt.custom_name] = i
        for _prop in evt.properties:
            keep_vals = []
            for _val in _prop.values:
                if _val.startswith("@"):
                    if _val not in pat.custom_names:
                        raise Exception(
                            f"Undefined reference found in values for property: {_val}"
                        )
                    _prop.value_refs.append(pat.custom_names[_val])
                else:
                    keep_vals.append(_val)
            _prop.values = keep_vals

    pat.pattern_str = pattern_str

    # if the code string is provided, vet it and resolve the custom name references
    if code_str is None:
        return pat

    code_str = code_str.strip()
    re_pat = r"(@[\w]+)\[(\d+)\]"
    custom_refs = re.findall(re_pat, code_str)

    lines = []  # sub custom event references with aliases
    for i, (name, offset) in enumerate(set(custom_refs)):
        assert name in pat.custom_names, "Undefined reference found in code"
        ref_idx = pat.custom_names[name]
        ref_offset = int(offset)
        lines.append(
            f'val_{i} = seq["events"][match_indices[{ref_idx}] + {ref_offset}]'
        )
        code_str = code_str.replace(f"{name}[{offset}]", f"val_{i}")

    vet_custom_code(code_str)
    code_str = rewrite_custom_code(code_str)
    pat.code = "\n".join([*lines, f"return {code_str}"])

    return pat


def parse_replace_pattern(
    repl_pattern_str: str, match_pattern: SeqPattern
) -> ReplSeqPattern:
    """Parse a replace pattern string into a ReplacePattern object."""
    repl_pattern_str = repl_pattern_str.replace(" ", "")
    repl_pattern: SeqPattern = seq_pattern.parse(repl_pattern_str)

    events: list[ReplEvtPattern] = []
    # match cross references with those defined in the match pattern
    for i, evt in enumerate(repl_pattern.event_patterns):
        if evt.custom_name is not None:
            _copy_all = False
            _copy_reverse = False
            if evt.custom_name.startswith("ALL@"):
                pick_name = evt.custom_name[3:]
                _copy_all = True
            elif evt.custom_name.startswith("REVERSE@"):
                pick_name = evt.custom_name[7:]
                _copy_reverse = True
            elif evt.custom_name.startswith("@"):
                pick_name = evt.custom_name
            else:
                raise Exception(
                    f"invalid custom name reference in the replace pattern: {evt.custom_name}"
                )

            assert (
                pick_name in match_pattern.custom_names
            ), f"undefined custom name reference in the replace pattern: {pick_name}"
            events.append(
                ReplEvtPattern(
                    ref_custom_name=match_pattern.custom_names[pick_name],
                    copy_ref_all=_copy_all,
                    copy_ref_reverse=_copy_reverse,
                )
            )
        else:
            # copy over all the properties specified
            properties = []
            for _prop in evt.properties:
                assert (
                    _prop.op == Operator.EQ and len(_prop.values) == 1
                ), "replace pattern properties only specify assignment"

                new_prop = Property(op=Operator.EQ, key=_prop.key)
                for _val in _prop.values:
                    if _val.startswith("@"):
                        if _val not in match_pattern.custom_names:
                            raise Exception(
                                f"Undefined reference found as value for property: {_val}"
                            )
                        new_prop.value_refs.append(match_pattern.custom_names[_val])
                    else:
                        new_prop.values.append(_val)
                properties.append(new_prop)
            events.append(ReplEvtPattern(properties=properties))

    return ReplSeqPattern(events=events, properties=repl_pattern.properties)
