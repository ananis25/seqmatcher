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


class MatchedEvents(TypedDict):
    indices: list[list[int]]
    counts: list[list[int]]
    starts: list[int]
    ends_excl: list[int]


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


# def replace_pattern(
#     match_pat: SeqPattern, replace_pat: SeqPattern, sequences: list[Sequence]
# ) -> list[Sequence]:
#     """Replace the pattern with the replacement pattern in the sequences to yield new sequences"""
#     if match_pat.allow_overlaps:
#         raise Exception(
#             "replace semantics don't make sense when event subsequences overlap"
#         )
#     matched_mask, matched_data = match_pattern(match_pat, sequences)
#     new_sequences = []
#     for i, seq in enumerate(sequences):

#     return sequences


def extract_pattern(pat: SeqPattern, sequences: list[Sequence]) -> list[Sequence]:
    """Extract the pattern from the sequences to yield new sequences"""
    new_sequences: list[Sequence] = []
    matched_mask, matched_data = match_pattern(pat, sequences)
    for i, seq in enumerate(sequences):
        if not matched_mask[i]:
            continue
        matches = matched_data[i]
        assert matches is not None
        for st, end in zip(matches["starts"], matches["ends_excl"]):
            new_seq = seq.copy()
            new_seq["events"] = seq["events"][st:end]
            new_sequences.append(new_seq)
    return new_sequences


def match_pattern(
    pat: SeqPattern, sequences: list[Sequence]
) -> tuple[list[bool], list[Optional[MatchedEvents]]]:
    matched_mask: list[bool] = []
    matched_data: list[Optional[MatchedEvents]] = []

    max_run_events = max(len(seq["events"]) for seq in sequences)
    cache = CacheMatch(len(pat.events), max_run_events)
    for seq in sequences:
        matched = True
        cache.reset()

        # filter by property matching
        for prop in pat.properties:
            if not match_prop(prop, seq):  # type:ignore
                matched = False
                break
        if not matched:
            matched_mask.append(False)
            matched_data.append(None)
            continue

        matches = MatchedEvents(indices=[], counts=[], starts=[], ends_excl=[])
        pos = 0
        start_before = 1 if pat.match_seq_start else len(pat.events)
        while pos < start_before:
            match_indices: list[int] = []
            match_counts: list[int] = []
            if match_sequence_here(
                pat, 0, seq["events"], pos, match_indices, match_counts, cache
            ):
                matches["indices"].append(match_indices)
                matches["counts"].append(match_counts)
                # NOTE: don't know why this doesn't type correctly
                matches["starts"].append(
                    match_indices[pat.idx_start_event]  # type: ignore
                    if pat.idx_start_event is not None
                    else 0
                )
                matches["ends_excl"].append(
                    match_indices[pat.idx_end_event] + match_counts[pat.idx_end_event]  # type: ignore
                    if pat.idx_end_event is not None
                    else len(seq["events"])
                )

                # conclude if we only want to match the first occurrence
                if not pat.match_all:
                    break
                if not pat.allow_overlaps:
                    pos = matches["ends_excl"][-1]
                else:
                    pos = match_indices[0] + 1
            else:
                pos = pos + 1

        if len(matches["indices"]) > 0:
            matched_mask.append(True)
            matched_data.append(matches)
        else:
            matched_mask.append(False)
            matched_data.append(None)

    return matched_mask, matched_data


def match_sequence_here(
    pat: SeqPattern,
    pat_pos: int,
    events: list[Event],
    events_pos: int,
    match_indices: list,
    match_counts: list,
    cache: CacheMatch,
):
    """Match the events in the sequence starting at the given position of the pattern"""
    evt_p = pat.events[pat_pos]
    for pos in range(events_pos, events_pos + evt_p.min_count):
        # more events to match than present in the sequence
        if pos >= len(events):
            return False
        if not match_event(pat.events, pat_pos, events, pos, cache):
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
                pat, pat_pos + 1, events, pos, match_indices, match_counts, cache
            ):
                return True

            # we have some leeway to match more of the current event
            if match_event(pat.events, pat_pos, events, pos, cache):
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
                pat, pat_pos + 1, events, pos, match_indices, match_counts, cache
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
            if not match_event(pat.events, pat_pos, events, pos, cache):
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
    cache: CacheMatch,
) -> bool:
    """Check whether the event in the sequence matches the event specified in the pattern. Cache
    the results for if we need it again.
    """
    cached = cache.get(pat_pos, seq_pos)
    if cached is not None:
        return cached

    result = _match_event(pat_events[pat_pos], seq_events[seq_pos])
    cache.set(pat_pos, seq_pos, result)
    return result


def _match_event(pat_event: EvtPattern, seq_event: Event) -> bool:
    """Check whether the event in the sequence matches the event specified in the pattern."""
    if pat_event.flag_exclude_names:
        if seq_event["_event_name"] in pat_event.list_names:
            return False
    else:
        if seq_event["_event_name"] not in pat_event.list_names:
            return False

    for prop in pat_event.properties:
        if not match_prop(prop, seq_event):  # type:ignore
            return False
    return True


def match_prop(prop: Property, obj: dict) -> bool:
    if prop.key not in obj:
        return False
    elif prop.op == Operator.EQ:
        return obj[prop.key] in prop.value
    elif prop.op == Operator.NE:
        return obj[prop.key] not in prop.value
    elif prop.op == Operator.GT:
        return obj[prop.key] > prop.value
    elif prop.op == Operator.GE:
        return obj[prop.key] >= prop.value
    elif prop.op == Operator.LT:
        return obj[prop.key] < prop.value
    elif prop.op == Operator.LE:
        return obj[prop.key] <= prop.value
    return True
