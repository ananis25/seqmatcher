"""
Data structures to represent a sequence matching instruction. 
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Any, Protocol, runtime_checkable, Iterable

__all__ = [
    "Operator",
    "cast_operator",
    "Property",
    "EvtPattern",
    "SeqPattern",
    "ReplEvtPattern",
    "ReplSeqPattern",
    "Rule",
]


class Operator(Enum):
    EQ = 1
    NE = 2
    LT = 3
    LE = 4
    GT = 5
    GE = 6


def cast_operator(op: str) -> Operator:
    if op == "==":
        return Operator.EQ
    elif op == "!=":
        return Operator.NE
    elif op == "<":
        return Operator.LT
    elif op == "<=":
        return Operator.LE
    elif op == ">":
        return Operator.GT
    elif op == ">=":
        return Operator.GE
    else:
        raise ValueError(f"Unknown operator: {op}")


@dataclass
class Property:
    op: "Operator"
    key: str
    value: list[Any] = field(default_factory=list)
    value_refs: list[int] = field(default_factory=list)


@dataclass
class EvtPattern:
    custom_name: Optional[str] = None
    min_count: int = 0
    max_count: Optional[int] = None
    properties: list["Property"] = field(default_factory=list)
    code: list[str] = field(default_factory=list)


@dataclass
class ReplEvtPattern:
    ref_custom_name: Optional[int] = None
    copy_ref_all: bool = False
    copy_ref_reverse: bool = False
    properties: list["Property"] = field(default_factory=list)
    code: list[str] = field(default_factory=list)


@dataclass
class SeqPattern:
    match_all: bool = True
    allow_overlaps: bool = False
    match_seq_start: bool = False
    match_seq_end: bool = False
    idx_start_event: Optional[int] = None
    idx_end_event: Optional[int] = None
    custom_names: dict[str, int] = field(default_factory=dict)
    events: list[EvtPattern] = field(default_factory=list)
    properties: list["Property"] = field(default_factory=list)
    code: list[str] = field(default_factory=list)


@dataclass
class ReplSeqPattern:
    events: list[ReplEvtPattern] = field(default_factory=list)
    properties: list["Property"] = field(default_factory=list)
    code: list[str] = field(default_factory=list)


@runtime_checkable
class Rule(Protocol):
    def execute(self, sequences: Any) -> Any:
        pass


class RuleRegular:
    match_pattern: SeqPattern

    def __init__(self, pattern: SeqPattern):
        self.match_pattern = pattern

    def execute(self, sequences: Any) -> Any:
        return sequences


class RuleReplace:
    match_pattern: SeqPattern
    replace_pattern: SeqPattern

    def __init__(self, pattern: SeqPattern, replace_pattern: SeqPattern):
        self.match_pattern = pattern
        self.replace_pattern = replace_pattern

    def execute(self, sequences: Any) -> Any:
        return sequences


class RuleNot:
    match_pattern: SeqPattern

    def __init__(self, pattern: SeqPattern):
        self.match_pattern = pattern

    def execute(self, sequences: Any) -> Any:
        return sequences


class RuleOr:
    list_match_patterns: list[SeqPattern]

    def __init__(self, patterns: list[SeqPattern]):
        self.list_match_patterns = patterns

    def execute(self, sequences: Any) -> Any:
        return sequences
