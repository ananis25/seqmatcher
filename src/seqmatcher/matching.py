"""
This module implements the matching routine. 
"""

import numpy as np
from typing import Union, Any, Optional, TypedDict, Iterable
from .primitives import *


class Event(TypedDict):
    _event_name: str


class Sequence(TypedDict):
    events: list[Event]


class MatchedSubSeq(TypedDict):
    seq_id: int
    evt_indices: list[int]
    evt_counts: list[int]
    start: int
    end_excl: int


class CacheMatch:
    values: np.ndarray

    def __init__(self, sz_pat: int, sz_events: int) -> None:
        self.values = np.full((sz_pat, sz_events), fill_value=-1, dtype=np.int8)

    def set(self, idx_pat: int, idx_event: int, value: bool) -> None:
        self.values[idx_pat, idx_event] = 1 if value else 0

    def get(self, idx_pat: int, idx_event: int) -> Optional[bool]:
        v = self.values[idx_pat, idx_event]
        if v == 1:
            return True
        elif v == 0:
            return False
        else:
            return None

    def reset(self):
        self.values.fill(-1)


def replace_pattern(
    match_pat: SeqPattern, replace_pat: ReplSeqPattern, sequences: list[Sequence]
) -> list[Sequence]:
    """Replace the pattern with the replacement pattern in the sequences to yield new sequences"""
    if match_pat.allow_overlaps:
        raise Exception(
            "replace semantics don't make sense when event subsequences overlap"
        )
    new_sequences = []
    for seq in sequences:
        matched_subseqs = match_pattern(match_pat, [seq])
        if len(matched_subseqs) == 0:
            continue

        # for all the subsequences matched in the same sequence, we produce a single sequence after patching the
        # matched events. So, if the original sequence had events 0-8, with a match between 3-7, the extracted
        # subsequence would be: 0-2, 3-7 (patched with the replacement pattern), 8.
        pos = 0
        events = []
        for subseq in matched_subseqs:
            # copy events before the match starts
            for i in range(pos, subseq["start"]):
                events.append(seq["events"][i])

            # copy over events from the matched subsequence
            for repl_evt in replace_pat.events:
                if repl_evt.ref_custom_name is not None:
                    ref_index = subseq["evt_indices"][repl_evt.ref_custom_name]
                    ref_count = subseq["evt_counts"][repl_evt.ref_custom_name]
                    if repl_evt.copy_ref_all:
                        events.extend(seq["events"][ref_index : ref_index + ref_count])
                    elif repl_evt.copy_ref_reverse:
                        events.extend(
                            seq["events"][
                                ref_index + ref_count - 1 : ref_index - 1 : -1
                            ]
                        )
                    else:
                        evt = seq["events"][ref_index]
                        events.append(evt)
                else:
                    evt = Event(_event_name="noname")
                    for _prop in repl_evt.properties:
                        if len(_prop.value) > 0:
                            evt[_prop.key] = _prop.value[0]
                        else:
                            ref_index = _prop.value_refs[0]
                            evt[_prop.key] = seq["events"][ref_index][_prop.key]
                    events.append(evt)

            # copy events following the match
            for i in range(subseq["end_excl"], len(seq["events"])):
                events.append(seq["events"][i])

        new_seq = Sequence(events=events)
        for prop_name, prop_val in seq.items():
            if prop_name != "events":
                new_seq[prop_name] = prop_val
        for _prop in replace_pat.properties:
            new_seq[_prop.key] = _prop.value[0]
        new_sequences.append(new_seq)

    return sequences


def extract_pattern(pat: SeqPattern, sequences: list[Sequence]) -> list[Sequence]:
    """Extract the pattern from the sequences to yield new sequences"""
    new_sequences: list[Sequence] = []
    for matched_seq in match_pattern(pat, sequences):
        seq = sequences[matched_seq["seq_id"]]
        new_seq = seq.copy()
        new_seq["events"] = seq["events"][
            matched_seq["start"] : matched_seq["end_excl"]
        ]
        new_sequences.append(new_seq)
    return new_sequences


def match_pattern(pat: SeqPattern, sequences: list[Sequence]) -> list[MatchedSubSeq]:
    matched_sequences: list[MatchedSubSeq] = []

    for seq_id, seq in enumerate(sequences):
        matched = True
        # filter by property matching
        for prop in pat.properties:
            if not match_prop(prop, seq):  # type:ignore
                matched = False
                break
        if not matched:
            continue

        pos = 0
        start_before = 1 if pat.match_seq_start else len(pat.events)
        while pos < start_before:
            match_indices: list[int] = []
            match_counts: list[int] = []
            if match_sequence_here(
                pat, 0, seq["events"], pos, match_indices, match_counts
            ):
                matched_seq = MatchedSubSeq(
                    seq_id=seq_id, evt_indices=[], evt_counts=[], start=0, end_excl=0
                )
                matched_seq["evt_indices"] = match_indices
                matched_seq["evt_counts"] = match_counts
                matched_seq["start"] = (
                    match_indices[pat.idx_start_event]  # type: ignore
                    if pat.idx_start_event is not None
                    else 0
                )
                matched_seq["end_excl"] = (
                    match_indices[pat.idx_end_event] + match_counts[pat.idx_end_event]  # type: ignore
                    if pat.idx_end_event is not None
                    else len(seq["events"])
                )
                matched_sequences.append(matched_seq)

                # conclude if we only want to match the first occurrence
                if not pat.match_all:
                    break
                if not pat.allow_overlaps:
                    pos = matched_seq["end_excl"]
                else:
                    pos = match_indices[0] + 1
            else:
                pos = pos + 1

    return matched_sequences


def match_sequence_here(
    pat: SeqPattern,
    pat_pos: int,
    events: list[Event],
    events_pos: int,
    match_indices: list[int],
    match_counts: list[int],
):
    """Match the events in the sequence starting at the given position of the pattern"""
    evt_p = pat.events[pat_pos]
    for pos in range(events_pos, events_pos + evt_p.min_count):
        # more events to match than present in the sequence
        if pos >= len(events):
            return False
        if not match_event(
            pat.events, pat_pos, events, pos, match_indices, match_counts
        ):
            return False
    match_indices.append(events_pos)
    match_counts.append(evt_p.min_count)

    pos_limit = len(events)
    if evt_p.max_count is not None:
        pos_limit = min(pos_limit, events_pos + evt_p.max_count)

    # check if the pattern has more events left to match
    if pat_pos < len(pat.events) - 1:
        for pos in range(events_pos + evt_p.min_count, pos_limit):
            # see if the rest of the pattern matches rest of the sequence
            # if it does, we have a match since we already matched min_count copies of current event
            if match_sequence_here(
                pat, pat_pos + 1, events, pos, match_indices, match_counts
            ):
                return True

            # we have some leeway to match more of the current event
            if match_event(
                pat.events, pat_pos, events, pos, match_indices, match_counts
            ):
                match_counts[-1] += 1
            else:
                match_indices.pop()
                match_counts.pop()
                return False

        # we have matched all copies of the current event, onto the next
        pos = pos_limit
        if pos < len(events):
            # the sequence has more events to match to
            if match_sequence_here(
                pat, pat_pos + 1, events, pos, match_indices, match_counts
            ):
                return True
            else:
                match_indices.pop()
                match_counts.pop()
                return False
        else:
            # no events left in the sequence but the pattern has more left
            match_indices.pop()
            match_counts.pop()
            return False
    else:
        # pattern has no more events to match

        # match as many of the current event as feasible
        pos = events_pos + evt_p.min_count
        while pos < pos_limit:
            if not match_event(
                pat.events, pat_pos, events, pos, match_indices, match_counts
            ):
                break
            else:
                pos += 1
                match_counts[-1] += 1

        # check if the pattern had to match at the end of the sequence
        if pat.match_seq_end and pos < len(events):
            match_indices.pop()
            match_counts.pop()
            return False
        else:
            return True


def match_event(
    pat_events: list[EvtPattern],
    pat_pos: int,
    seq_events: list[Event],
    seq_pos: int,
    match_indices: list[int],
    match_counts: list[int],
) -> bool:
    """Check whether the event in the sequence matches the event specified in the pattern."""
    pat_event = pat_events[pat_pos]
    seq_event = seq_events[seq_pos]

    for prop in pat_event.properties:
        lhs = seq_event[prop.key]
        rhs = []
        for v in prop.value:
            rhs.append(v)
        for idx in prop.value_refs:
            ref_event = seq_events[match_indices[idx]]
            rhs.append(ref_event[prop.key])
        if not match_prop(prop.op, lhs, rhs):
            return False
    return True


def match_prop(op, lhs, rhs) -> bool:
    if op == Operator.EQ:
        return lhs in rhs
    elif op == Operator.NE:
        return lhs not in rhs
    elif op == Operator.GT:
        return lhs > rhs
    elif op == Operator.GE:
        return lhs >= rhs
    elif op == Operator.LT:
        return lhs < rhs
    elif op == Operator.LE:
        return lhs <= rhs
    return False
