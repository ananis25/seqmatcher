import ast
import inspect
from typing import Optional

import astunparse

from .primitives import *


# -----------------------------------------------------------
# utilities to work with the ast module
# -----------------------------------------------------------


def make_ast(obj) -> ast.AST:
    """Given a python object, generate the ast for it"""
    return ast.parse(inspect.getsource(obj))


def dump_ast(syntree: ast.AST) -> None:
    print(astunparse.dump(syntree))


def dump_code(syntree: ast.AST) -> str:
    return astunparse.unparse(syntree)


def _load(name: str):
    return ast.Name(name, ast.Load())


def _store(name: str):
    return ast.Name(name, ast.Store())


def _sub(var, idx):
    return ast.Subscript(var, idx)


def _assign(lhs, rhs):
    return ast.Assign([lhs], rhs)


def _const(value):
    return ast.Constant(value)


_op_dict = {
    Operator.EQ: ast.Eq(),
    Operator.NE: ast.NotEq(),
    Operator.GT: ast.Gt(),
    Operator.GE: ast.GtE(),
    Operator.LT: ast.Lt(),
    Operator.LE: ast.LtE(),
}


def add_jit_decorator(fn: ast.FunctionDef) -> None:
    """Add the @nb.jit decorator to the function expression"""
    fn.decorator_list.append(
        ast.Attribute(value=_load("nb"), attr="njit", ctx=ast.Load())
    )


def prop_filter_fn(prop: Property, prop_idx: int) -> list[ast.stmt]:
    """Generate the statement to filter out events that don't satisfy the property. If the
    property is true for any one of the values, we are good. We don't need a terminal return
    since this snippet feeds into the function for the full event.

    So, the code looks something like:
    ```py
    value_1 = events[2]["team"]
    if value_1 is None:
        return False
    if not (value_1 == "team1" or value_1 == "team2" or value_1 == events[match_indices[1]]["team"]):
        return False
    ```
    """
    var = f"value_{prop_idx}"
    op = _op_dict[prop.op]

    cmps: list[ast.Compare] = []
    for val in prop.value:
        cmps.append(ast.Compare(_load(var), [op], [_const(val)]))
    for ref in prop.value_refs:
        val = _sub(
            _sub(_load("events"), _sub(_load("match_indices"), _const(ref))),
            _load(prop.key),
        )
        cmps.append(ast.Compare(_load(var), [op], [val]))

    stmts = []
    stmts.append(
        _assign(_store(var), _sub(_sub(_load("events"), _load("i")), _load(prop.key)))
    )
    stmts.append(
        ast.If(
            ast.Compare(_load(var), [ast.Is()], [_const(None)]),
            [ast.Return(_const(False))],
            [],
        )
    )
    stmts.append(
        ast.If(
            ast.UnaryOp(ast.Not(), ast.BoolOp(ast.Or(), cmps)),
            [ast.Return(_const(False))],
            [],
        )
    )
    return stmts


def event_filter_fn(evt_pat: EvtPattern, pat_idx: int) -> ast.FunctionDef:
    """Generate the statement to filter out events that don't satisfy the pattern.
    Something like:

    ```py
    def match_i(events, evt_idx):
        # {if prop1 is False, return a False value early}
        # {if prop2 is False, return a False value early}
        ...

        # all properties matched, so return True
        return True
    ```
    """
    block: list[ast.stmt] = []
    for prop_idx, prop in enumerate(evt_pat.properties):
        block.extend(prop_filter_fn(prop, prop_idx))
    block.append(ast.Return(_const(True)))

    fn_name = f"match_{pat_idx}"
    fn_args = ast.arguments(args=[ast.arg("events"), ast.arg("i")], defaults=[])
    return ast.FunctionDef(fn_name, fn_args, block, [], [])


def seq_filter_fn(seq_pat: SeqPattern) -> ast.FunctionDef:
    """Generate the statement to filter out sequences that don't satisfy the properties
    specified.
    ```py
    def match_seq(seq):
        value_1 = seq["team"]
        if not (value_1 == "team1" or value_1 == "team2"):
            return False
        value_2 = seq["city"]
        if not value_2 == "city1":
            return False
        ...

        return True
    ```
    """
    block: list[ast.stmt] = []
    for prop_idx, prop in enumerate(seq_pat.properties):
        var = f"value_{prop_idx}"
        op = _op_dict[prop.op]
        cmps: list[ast.Compare] = []
        for val in prop.value:
            cmps.append(ast.Compare(_load(var), [op], [_const(val)]))

        block.append(_assign(_store(var), _sub(_load("seq"), _load(prop.key))))
        block.append(
            ast.If(
                ast.Compare(_load(var), [ast.Is()], [_const(None)]),
                [ast.Return(_const(False))],
                [],
            )
        )
        block.append(
            ast.If(ast.BoolOp(ast.Or(), cmps), [ast.Return(_const(False))], [])
        )
    block.append(ast.Return(_const(True)))

    fn_name = "match_seq"
    fn_args = ast.arguments(args=[ast.arg("seq")], defaults=[])
    return ast.FunctionDef(fn_name, fn_args, block, [], [])


def pattern_match_fns(seq_pat: SeqPattern) -> list[ast.FunctionDef]:
    """Generate the statement to filter out events that don't satisfy the pattern.
    Something like:

    ```py
    def match_seq(seq):
        ...

    def match_event(pat_idx, events, i):
        if pat_idx == 0:
            return match_0(events, i)
        elif pat_idx == 1:
            return match_1(events, i)
        # unreachable
        return True

    def match_0(events, i):
        ...

    def match_1(events, i):
        ...
    ```
    """
    list_fns: list[ast.FunctionDef] = []
    for pat_idx, evt_pat in enumerate(seq_pat.event_patterns):
        list_fns.append(event_filter_fn(evt_pat, pat_idx))

    list_branches: list[ast.If] = []
    args = [_load("events"), _load("i")]
    for pat_idx, fn in enumerate(list_fns):
        body = ast.Return(ast.Call(_load(fn.name), args, []))
        cmp = ast.Compare(_load("pat_idx"), ast.Eq(), _const(pat_idx))
        list_branches.append(ast.If(cmp, [body], []))

    dispatch_name = "match_event"
    dispatch_args = ast.arguments(
        args=[ast.arg("pat_idx"), ast.arg("events"), ast.arg("i")], defaults=[]
    )
    _block: Optional[ast.If] = None
    for branch in reversed(list_branches):
        if _block is not None:
            branch.orelse.append(_block)
        _block = branch
    dispatch_body = [_block, ast.Return(_const(True))]
    dispatch_fn = ast.FunctionDef(dispatch_name, dispatch_args, dispatch_body, [], [])

    return [seq_filter_fn(seq_pat), dispatch_fn] + list_fns
