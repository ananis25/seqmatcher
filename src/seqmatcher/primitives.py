"""
Data structures to represent a sequence matching instruction. The user input is 
parsed into one of these before executing it. 
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Protocol, runtime_checkable, Iterable

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
    """filter expressions only permit one of these operators"""

    EQ = 1
    NE = 2
    LT = 3
    LE = 4
    GT = 5
    GE = 6


def cast_operator(op: str) -> Operator:
    """parse the text substring into an Operator"""
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
    """Sequences or events can specify boolean conditions, called `properties` as filters.
    Each property is specified as: `<key> <operator> <value>`, where `key` is an attribute
    of the event/sequence. The `value` can be a single literal, multiple literals, and even
    include a reference to the same attribute value for another event in the pattern.
    """

    op: "Operator"
    key: str
    value: list[Any] = field(default_factory=list)
    value_refs: list[int] = field(default_factory=list)


@dataclass
class EvtPattern:
    """Specifies a single event in the sequence to match.

    Attributes:
        custom_name: If specified, this event will be referenced by the name in the pattern.
        min_count: minimum num of this event to match in the sequence
        max_count: maximum num of this event to match in the sequence
        properties: list of properties to match
    """

    custom_name: Optional[str] = None
    min_count: int = 0
    max_count: Optional[int] = None
    properties: list["Property"] = field(default_factory=list)


@dataclass
class ReplEvtPattern:
    """Specifies a single event in the replacement pattern for a sequence.

    Attributes:
        ref_custom_name: If specified, `min_count` num of the corresponding event from the match
            pattern will be copied to the output sequence.
        copy_ref_all: If True, all matches for the corresponding event in the match pattern are
            copied to the output
        copy_ref_reverse: If True, all matches for the corresponding event in the match pattern are copied
            to the output in reverse order
        properties: list of properties to copy
    """

    ref_custom_name: Optional[int] = None
    copy_ref_all: bool = False
    copy_ref_reverse: bool = False
    properties: list["Property"] = field(default_factory=list)


@dataclass
class SeqPattern:
    """Specifies which sequences of events to match; roughly as: `[<evt-pattern>*] {{ properties }}`.
    Each `evt-pattern` is an object of the type `EvtPattern`, while `properties` is a list of
    `Property` objects.
    NOTE: Most attributes are similar to what you'd expect in a regex pattern, only instead of characters,
    here we match events.

    Attributes:
        match_all: If True, only the first matched subsequence is extracted. Else, matching continues until
            the end of the sequence is reached.
        allow_overlaps: If True, overlapping subsequences are allowed. Else, the next match in a sequence is
            searched after the last event in the current match.
        match_seq_start: If True, the match must start with the first event in the sequence.
        match_seq_end: If True, the match must end with the last event in the sequence.
        idx_start_event: If specified, the subsequence is captured only starting at the specified index.
        idx_end_event: If specified, the subsequence is captured only ending at the specified index.
        custom_names: mapping of custom names for events to an integer index for them.
        events: list of `EvtPattern` objects to match
        properties: list of properties to match
    """

    match_all: bool = True
    allow_overlaps: bool = False
    match_seq_start: bool = False
    match_seq_end: bool = False
    idx_start_event: Optional[int] = None
    idx_end_event: Optional[int] = None
    custom_names: dict[str, int] = field(default_factory=dict)
    events: list[EvtPattern] = field(default_factory=list)
    properties: list["Property"] = field(default_factory=list)


@dataclass
class ReplSeqPattern:
    """Specifies how to construct the subsequence to output, after matching against a `SeqPattern`"""

    events: list[ReplEvtPattern] = field(default_factory=list)
    properties: list["Property"] = field(default_factory=list)


@runtime_checkable
class Rule(Protocol):
    """Each step of the sequence matching pipeline is specified as a `Rule` that takes a list of
    sequences as an input, and can be one of:
    - regular: matches each sequence against the `SeqPattern`, returns the list of all matched subsequences.
    - replacement: matches each sequence against the `SeqPattern`, and returns the list of all matched subsequences
        edited per the `ReplSeqPattern`.
    - not: matches each sequence against the `SeqPattern`, returns all sequences that didn't have a match.
    - or: matches against a list of `SeqPattern` objects, collects all the matched subsequences from each of them,
        and returns a union over them.
    """

    def execute(self, sequences: Iterable[Any]) -> Iterable[Any]:
        ...


class RuleRegular:
    match_pattern: SeqPattern

    def __init__(self, pattern: SeqPattern):
        self.match_pattern = pattern

    def execute(self, sequences: Iterable[Any]) -> Iterable[Any]:
        return sequences


class RuleReplace:
    match_pattern: SeqPattern
    replace_pattern: SeqPattern

    def __init__(self, pattern: SeqPattern, replace_pattern: SeqPattern):
        self.match_pattern = pattern
        self.replace_pattern = replace_pattern

    def execute(self, sequences: Iterable[Any]) -> Iterable[Any]:
        return sequences


class RuleNot:
    match_pattern: SeqPattern

    def __init__(self, pattern: SeqPattern):
        self.match_pattern = pattern

    def execute(self, sequences: Iterable[Any]) -> Iterable[Any]:
        return sequences


class RuleOr:
    list_match_patterns: list[SeqPattern]

    def __init__(self, patterns: list[SeqPattern]):
        self.list_match_patterns = patterns

    def execute(self, sequences: Iterable[Any]) -> Iterable[Any]:
        return sequences
