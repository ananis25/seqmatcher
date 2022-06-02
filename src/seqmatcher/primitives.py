"""
Data structures to represent a sequence matching instruction. The user input is 
parsed into one of these before executing it. 
"""

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import (
    Optional,
    Any,
    Protocol,
    TypedDict,
    runtime_checkable,
    Type,
    Union,
)

__all__ = [
    "Operator",
    "Property",
    "LITERAL_TYPES",
    "EvtPattern",
    "SeqPattern",
    "ReplEvtPattern",
    "ReplSeqPattern",
    "Rule",
    "Event",
    "Sequence",
]


class Operator(Enum):
    """filter expressions only permit one of these operators"""

    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


LITERAL_TYPES = Union[str, int, float, bool]


@dataclass
class Property:
    """Sequences or events can specify boolean conditions, called `properties` as filters.
    Each property is specified as: `<key> <operator> <value>`, where `key` is an attribute
    of the event/sequence. The `value` can be a single literal, multiple literals, and even
    include a reference to the same attribute value for another event in the pattern.
    """

    op: "Operator"
    key: str
    values: list[Any] = field(default_factory=list)
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
    max_count: int = 0
    properties: list["Property"] = field(default_factory=list)

    def __repr__(self) -> str:
        return repr_dataclass(self)


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

    def __repr__(self) -> str:
        return repr_dataclass(self)


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

    pattern_str: str = ""
    match_all: bool = True
    allow_overlaps: bool = False
    match_seq_start: bool = False
    match_seq_end: bool = False
    idx_start_event: Optional[int] = None
    idx_end_event: Optional[int] = None
    custom_names: dict[str, int] = field(default_factory=dict)
    event_patterns: list[EvtPattern] = field(default_factory=list)
    properties: list["Property"] = field(default_factory=list)

    def __repr__(self) -> str:
        return repr_dataclass(self)

    def type_properties(
        self,
        seq_prop_types: dict[str, Type[LITERAL_TYPES]],
        evt_prop_types: dict[str, Type[LITERAL_TYPES]],
    ) -> None:
        """Cast the property values to match to the appropriate types based on the dataset."""
        for prop in self.properties:
            assert (
                prop.key in seq_prop_types
            ), f"property: {prop.key} not found in dataset"
            datatyp = seq_prop_types[prop.key]
            if not datatyp in (int, float):
                assert prop.op in (
                    Operator.EQ,
                    Operator.NE,
                ), "operator: {op} is only supported for numerical types: {prop.key}"
            prop.values = [datatyp(v) for v in prop.values]

        for evt_pat in self.event_patterns:
            for prop in evt_pat.properties:
                assert (
                    prop.key in evt_prop_types
                ), f"property: {prop.key} not found in the events records of dataset"
                datatyp = evt_prop_types[prop.key]
                if not datatyp in (int, float):
                    assert prop.op in (
                        Operator.EQ,
                        Operator.NE,
                    ), "operator: {op} is only supported for numerical types: {prop.key}"
                prop.values = [datatyp(v) for v in prop.values]


@dataclass
class ReplSeqPattern:
    """Specifies how to construct the subsequence to output, after matching against a `SeqPattern`."""

    events: list[ReplEvtPattern] = field(default_factory=list)
    properties: list["Property"] = field(default_factory=list)

    def __repr__(self) -> str:
        return repr_dataclass(self)


class Event(TypedDict):
    _eventName: str


class Sequence(TypedDict):
    events: list[Event]


@runtime_checkable
class Rule(Protocol):
    """Each step of the sequence matching pipeline is specified as a `Rule` that takes a list of
    sequences as an input, and returns another list of sequences as output.
    """

    ...


class RuleRegular:
    """Matches each sequence against the `SeqPattern`, returns the list of all matched subsequences."""

    match_pattern: SeqPattern

    def __init__(self, pattern: SeqPattern):
        self.match_pattern = pattern


class RuleReplace:
    """Matches each sequence against the `SeqPattern`, and returns the list of all
    matched subsequences edited per the `ReplSeqPattern`.
    """

    match_pattern: SeqPattern
    replace_pattern: ReplSeqPattern

    def __init__(self, pattern: SeqPattern, replace_pattern: ReplSeqPattern):
        self.match_pattern = pattern
        self.replace_pattern = replace_pattern


class RuleNot:
    """Matches each sequence against the `SeqPattern`, returns all sequences that didn't have a match."""

    match_pattern: SeqPattern

    def __init__(self, pattern: SeqPattern):
        self.match_pattern = pattern


class RuleOr:
    """Matches against a list of `SeqPattern` objects, collects all the matched
    subsequences from each of them, and returns a union over them.
    """

    list_match_patterns: list[SeqPattern]

    def __init__(self, patterns: list[SeqPattern]):
        self.list_match_patterns = patterns


# -----------------------------------------------------------
# utilities
# -----------------------------------------------------------


def repr_dataclass(obj, indent=2, _indents=0) -> str:
    """Pretty print a dataclass.

    credits: https://stackoverflow.com/a/66809229/10450004
    """
    if isinstance(obj, str):
        return f"'{obj}'"

    if not is_dataclass(obj) and not isinstance(obj, (dict, list, tuple)):
        return str(obj)

    this_indent = indent * _indents * " "
    next_indent = indent * (_indents + 1) * " "
    start = f"{type(obj).__name__}("
    end = ")"

    if is_dataclass(obj):
        body = "\n".join(
            f"{next_indent}{field.name}="
            f"{repr_dataclass(getattr(obj, field.name), indent, _indents + 1)},"
            for field in fields(obj)
        )

    elif isinstance(obj, dict):
        start, end = "{}"
        if len(obj) > 0:
            body = "\n".join(
                f"{next_indent}{repr_dataclass(key, indent, _indents + 1)}: "
                f"{repr_dataclass(value, indent, _indents + 1)},"
                for key, value in obj.items()
            )
        else:
            body = ""

    else:
        if isinstance(obj, list):
            start, end = "[]"
        elif isinstance(obj, tuple):
            start, end = "()"
        if len(obj) > 0:
            body = "\n".join(
                f"{next_indent}{repr_dataclass(item, indent, _indents + 1)},"
                for item in obj
            )
        else:
            body = ""

    if body == "":
        return f"{start}{end}"
    else:
        return f"{start}\n{body}\n{this_indent}{end}"
