import ast
import inspect
from typing import Optional
from types import ModuleType

import astunparse

from .primitives import *


# -----------------------------------------------------------
# utilities to work with the ast module
# -----------------------------------------------------------


def make_ast(obj) -> ast.AST:
    """Given a python object, generate the ast for it"""
    return ast.parse(inspect.getsource(obj))


def pprint_ast(syntree: ast.AST) -> None:
    print(astunparse.dump(syntree))


def ast_to_code(syntree: ast.AST) -> str:
    return ast.unparse(ast.fix_missing_locations(syntree))


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


def _args(args: list[str]):
    return ast.arguments(
        args=[ast.arg(arg) for arg in args],
        posonlyargs=[],
        kwonlyargs=[],
        defaults=[],
    )


# -----------------------------------------------------------
# assembling code for matching the pattern
# -----------------------------------------------------------

_op_dict = {
    Operator.EQ: ast.Eq(),
    Operator.NE: ast.NotEq(),
    Operator.GT: ast.Gt(),
    Operator.GE: ast.GtE(),
    Operator.LT: ast.Lt(),
    Operator.LE: ast.LtE(),
}


def prop_filter_block(prop: Property, prop_idx: int) -> list[ast.stmt]:
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
    for val in prop.values:
        cmps.append(ast.Compare(_load(var), [op], [_const(val)]))
    for ref in prop.value_refs:
        val = _sub(
            _sub(_load("events"), _sub(_load("match_indices"), _const(ref))),
            _const(prop.key),
        )
        cmps.append(ast.Compare(_load(var), [op], [val]))

    stmts = []
    stmts.append(
        _assign(_store(var), _sub(_sub(_load("events"), _load("i")), _const(prop.key)))
    )
    # null check
    stmts.append(
        ast.If(
            ast.Compare(_load(var), [ast.Is()], [_const(None)]),
            [ast.Return(_const(False))],
            [],
        )
    )
    # if the event attribute is not one of the values/value at a reference, return False
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
    def match_{i}(events, evt_idx, match_indices):
        # {if prop1 is False, return a False value early}
        # {if prop2 is False, return a False value early}
        ...

        # all properties matched, so return True
        return True
    ```
    """
    block: list[ast.stmt] = []
    for prop_idx, prop in enumerate(evt_pat.properties):
        block.extend(prop_filter_block(prop, prop_idx))
    block.append(ast.Return(_const(True)))

    fn_name = f"match_{pat_idx}"
    fn_args = _args(["events", "i", "match_indices"])
    return ast.FunctionDef(
        fn_name,
        fn_args,
        block,
        [numba_decorator()],
        [],
    )


def seq_pre_filter_fn(seq_pat: SeqPattern) -> ast.FunctionDef:
    """Generate the statement to filter out sequences that don't satisfy the properties
    specified. This is evaluated for a sequence before we start matching subsequences
    of it against the event patterns.

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
        for val in prop.values:
            cmps.append(ast.Compare(_load(var), [op], [_const(val)]))

        block.append(_assign(_store(var), _sub(_load("seq"), _const(prop.key))))
        block.append(
            ast.If(
                ast.Compare(_load(var), [ast.Is()], [_const(None)]),
                [ast.Return(_const(False))],
                [],
            )
        )
        block.append(
            ast.If(
                ast.UnaryOp(ast.Not(), ast.BoolOp(ast.Or(), cmps)),
                [ast.Return(_const(False))],
                [],
            )
        )
    block.append(ast.Return(_const(True)))

    fn_name = "match_seq_pre"
    fn_args = _args(["seq"])
    return ast.FunctionDef(
        fn_name,
        fn_args,
        block,
        [numba_decorator()],
        [],
    )


def seq_post_filter_fn(seq_pat: SeqPattern) -> str:
    """Generate the statement to filter out sequences that don't evaluate to True for
    the specified custom code expression. This is evaluated _separately_ for each matched
    subsequence.
    """
    fn_decl = "def match_seq_post(seq, match_indices):"
    if seq_pat.code is None:
        return "\n".join([fn_decl, "    return True"])
    else:
        stmts = seq_pat.code.split("\n")
        return "\n".join([fn_decl, *(f"    {stmt}" for stmt in stmts)])


def generate_match_fns(seq_pat: SeqPattern) -> list[ast.FunctionDef]:
    """Generate the statement to filter out events that don't satisfy the pattern.
    Something like:

    ```py
    def match_seq_pre(seq):
        ...

    def match_event(pat_idx, events, i, match_indices):
        if pat_idx == 0:
            return match_0(events, i, match_indices)
        elif pat_idx == 1:
            return match_1(events, i, match_indices)
        # unreachable
        return True

    def match_0(events, i, match_indices):
        ...

    def match_1(events, i, match_indices):
        ...
    ```
    """
    list_evt_match_fns: list[ast.FunctionDef] = []
    for pat_idx, evt_pat in enumerate(seq_pat.event_patterns):
        list_evt_match_fns.append(event_filter_fn(evt_pat, pat_idx))

    list_branches: list[ast.If] = []
    args = [_load("events"), _load("i"), _load("match_indices")]
    for pat_idx, fn in enumerate(list_evt_match_fns):
        body = ast.Return(ast.Call(_load(fn.name), args, []))
        cmp = ast.Compare(_load("pat_idx"), [ast.Eq()], [_const(pat_idx)])
        list_branches.append(ast.If(cmp, [body], []))

    evt_match_dispatch_name = "match_event"
    dispatch_args = _args(["pat_idx", "events", "i", "match_indices"])

    dispatch_body: list[ast.stmt] = []
    if len(list_branches) > 0:
        _block: Optional[ast.If] = None
        for branch in reversed(list_branches):
            if _block is not None:
                branch.orelse.append(_block)
            _block = branch
        dispatch_body = [_block]  # type: ignore

    dispatch_body.append(ast.Return(_const(True)))
    evt_match_dispatch_fn = ast.FunctionDef(
        evt_match_dispatch_name,
        dispatch_args,
        dispatch_body,
        [numba_decorator()],
        [],
    )

    return [seq_pre_filter_fn(seq_pat), evt_match_dispatch_fn] + list_evt_match_fns


def numba_decorator() -> ast.Call:
    return ast.Call(
        ast.Attribute(value=_load("nb"), attr="jit", ctx=ast.Load()),
        [],
        [ast.keyword("nopython", _const(True))],
    )


# -----------------------------------------------------------
# Vet and edit custom code property
# -----------------------------------------------------------


def vet_custom_code(code: str) -> None:
    """Vet the code to make sure it is valid."""
    tree = ast.parse(code)

    # refer - https://greentreesnakes.readthedocs.io/en/latest/nodes.html
    _ALLOWED_NODE_TYPES = (
        ast.Constant,
        ast.Num,
        ast.Str,
        ast.NameConstant,  # True, False, None
        ast.Name,
        ast.Load,
        ast.Store,
        ast.UnaryOp,
        ast.Not,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Gt,
        ast.GtE,
        ast.Lt,
        ast.LtE,
        ast.IfExp,
        ast.Attribute,
    )
    _ALLOWED_FUNCS = ["max", "min"]

    assert len(tree.body) == 1 and isinstance(
        tree.body[0], ast.Expr
    ), "code property can only be a single expression"

    # recurse on the expression and assert there are no unwanted constructs
    for node in ast.walk(tree.body[0].value):
        if isinstance(node, ast.Call):
            assert (
                isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS
            ), "function call in input code isn't supported"
        elif isinstance(node, ast.Attribute):
            assert isinstance(node.value, ast.Name), "invalid attribute access"
        elif isinstance(node, ast.Name):
            assert isinstance(node.id, str) and (
                node.id.startswith("val_") or node.id == "seq" or node.id == "length"
            ), "invalid identifier"
        else:
            assert isinstance(
                node, _ALLOWED_NODE_TYPES
            ), f"ast node of unsupported type encountered: {type(node)}"


class RewriteCode(ast.NodeTransformer):
    def visit_Attribute(self, node):
        return ast.Subscript(value=node.value, slice=_const(node.attr), ctx=node.ctx)

    def visit_Name(self, node):
        if node.id == "length":
            return ast.Call(
                func=_load("len"),
                args=[ast.Subscript(value=_load("seq"), slice=_const("events"))],
                keywords=[],
            )


def rewrite_custom_code(code: str) -> str:
    tree = ast.parse(code)
    new_tree = RewriteCode().visit(tree)
    return ast_to_code(new_tree)


# -----------------------------------------------------------
# routines to help cache the generated code.
# credits: https://github.com/DannyWeitekamp/Cognitive-Rule-Engine/blob/main/cre/caching.py
# -----------------------------------------------------------

import hashlib
import importlib
import os
import pathlib
import platform
import sys
import tempfile


# credits: https://stackoverflow.com/a/43418319/10450004
TEMP_DIR = "/tmp" if platform.system().upper() == "DARWIN" else tempfile.gettempdir()
CACHE_DIR = os.path.join(TEMP_DIR, "numba")
sys.path.append(CACHE_DIR)

CODE_CACHE_DIR = os.path.join(CACHE_DIR, "code_cache")
if not os.path.exists(CODE_CACHE_DIR):
    os.makedirs(CODE_CACHE_DIR, exist_ok=True)
    pathlib.Path(os.path.join(CODE_CACHE_DIR, "__init__.py")).touch(exist_ok=True)

JIT_CACHE_DIR = os.path.join(CACHE_DIR, "jit_cache")
if not os.path.exists(JIT_CACHE_DIR):
    os.makedirs(JIT_CACHE_DIR, exist_ok=True)
if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = JIT_CACHE_DIR


def get_cache_key(pattern_str: str) -> str:
    """Get the cache id for the given code."""
    return hashlib.md5(pattern_str.encode("utf-8")).hexdigest()


def present_in_cache(pattern_str: str) -> bool:
    """Get the cached module for the given code."""
    cache_key = get_cache_key(pattern_str)
    return os.path.exists(os.path.join(CODE_CACHE_DIR, f"{cache_key}.py"))


def write_to_cache(pattern_str: str, code: str) -> None:
    """Write the generated code to the cache."""
    cache_key = get_cache_key(pattern_str)
    with open(os.path.join(CODE_CACHE_DIR, f"{cache_key}.py"), "w") as f:
        f.write(code)


def import_from_cache(pattern_str: str) -> ModuleType:
    """Load the generated code from the cache."""
    cache_key = get_cache_key(pattern_str)
    return importlib.import_module(f"code_cache.{cache_key}")


def clear_cache(key: Optional[str]) -> None:
    """Clear the code cache."""
    if not key:
        for fname in os.listdir(CODE_CACHE_DIR):
            if fname.endswith(".py") and not fname != "__init__.py":
                os.remove(os.path.join(CODE_CACHE_DIR, fname))
    else:
        # remove the specified module only
        fname = os.path.join(CODE_CACHE_DIR, f"{key}.py")
        if os.path.exists(fname):
            os.remove(fname)
