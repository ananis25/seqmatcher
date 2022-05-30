"""
This module implements the matching routine, by JIT compiling the code 
for each input separately using Numba. 
"""

from typing import cast, Callable

import numpy as np
import numba as nb
import awkward as ak
from numba.typed import List

from .primitives import *
from .codegen import add_jit_decorator, dump_code, make_ast, pattern_match_fns


@nb.experimental.jitclass
class PatternInfo:
    length: int
    idx_start: int
    idx_end: int
    event_min_counts: nb.int64[:]  # type:ignore
    event_max_counts: nb.int64[:]  # type:ignore
    match_seq_start: bool
    match_seq_end: bool
    match_all: bool
    allow_overlaps: bool

    def __init__(
        self,
        length: int,
        idx_start: int,
        idx_end: int,
        event_min_counts: np.ndarray,
        event_max_counts: np.ndarray,
        match_seq_start: bool,
        match_seq_end: bool,
        match_all: bool,
        allow_overlaps: bool,
    ):
        self.length = length
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.event_min_counts = event_min_counts
        self.event_max_counts = event_max_counts
        self.match_seq_start = match_seq_start
        self.match_seq_end = match_seq_end
        self.match_all = match_all
        self.allow_overlaps = allow_overlaps


def match_pattern(pat: SeqPattern, sequences: ak.Array, jit: bool = True) -> ak.Array:
    pat_info = PatternInfo(
        len(pat.event_patterns),
        pat.idx_start_event if pat.idx_start_event else -1,
        pat.idx_end_event if pat.idx_end_event else -1,
        np.asarray([p.min_count for p in pat.event_patterns], dtype=np.int64),
        np.asarray([p.max_count for p in pat.event_patterns], dtype=np.int64),
        pat.match_seq_start,
        pat.match_seq_end,
        pat.match_all,
        pat.allow_overlaps,
    )

    # assemble code to pattern match
    entry_fn = _match_pattern
    list_fns = pattern_match_fns(pat)
    list_fns.append(make_ast(match_sequence_here))  # type:ignore
    if jit:
        for fn in list_fns:
            add_jit_decorator(fn)
        entry_fn = nb.njit(entry_fn)

    code_str = "\n\n".join(dump_code(f) for f in list_fns)
    code_str += "\n\n" + dump_code(entry_fn)

    ctx = {}
    for val in ("ak", "nb", "np", "PatternInfo"):
        ctx[val] = globals()[val]
    exec(code_str, ctx)

    res = entry_fn(
        pat_info, sequences, ctx["match_seq"], ctx["match_sequence_here"]
    )  # type: tuple[list[np.ndarray], ...]
    list_match_seq_ids, list_match_indices, list_match_counts = res
    match_seq_ids = np.concatenate(list_match_seq_ids)
    match_indices = np.concatenate(list_match_indices)
    match_counts = np.concatenate(list_match_counts)

    # filter down to legit matches
    valid_indices = match_seq_ids > 0
    match_seq_ids = match_seq_ids[valid_indices]
    match_indices = match_indices[valid_indices]
    match_counts = match_counts[valid_indices]

    return ak.Array(
        {"seq_id": match_seq_ids, "indices": match_indices, "counts": match_counts}
    )


def _match_pattern(
    pat: PatternInfo,
    sequences: ak.Array,
    match_seq: Callable[..., bool],
    match_seq_here: Callable[..., bool],
) -> tuple[list[np.ndarray], ...]:
    """Match the pattern against the sequences"""

    num_sequences = len(sequences)
    max_seq_length = ak.max(ak.num(sequences["events"]))

    # List of numpy arrays to output.
    # A cleaner option would be to output a list of structs instead but we want to keep allocations to
    # a minimum. So, initialize numpy arrays to record it, and when we run out of space, create a new
    # numpy array and append it to the list.
    # Another alternative would be to resize the array each time we run out of space, but numba copies
    # the array elements one at a time which is slow. Numpy likely just copies the buffer directly.
    list_match_seq_ids = List()
    list_match_indices = List()
    list_match_counts = List()

    def _gen_arrays():
        list_match_seq_ids.append(np.full(num_sequences, -1, dtype=np.int64))
        list_match_indices.append(np.zeros((num_sequences, pat.length), dtype=np.int32))
        list_match_counts.append(np.zeros((num_sequences, pat.length), dtype=np.int32))
        return (
            list_match_seq_ids[-1],
            list_match_indices[-1],
            list_match_counts[-1],
        )

    match_seq_ids, match_indices, match_counts = _gen_arrays()
    idx_match_arrays = 0

    for seq_id in range(num_sequences):
        seq = sequences[seq_id]

        # filter against the sequence properties
        if not match_seq(seq):
            continue
        events = seq["events"]  # type:ignore
        num_events = len(events)

        pos = 0
        start_before = 1 if pat.match_seq_start else pat.length
        while pos < start_before:
            i = idx_match_arrays  # save characters
            if match_seq_here(pat, 0, events, pos, match_indices[i], match_counts[i]):
                match_seq_ids[i] = seq_id
                # conclude if we only want to match the first occurrence
                if not pat.match_all:
                    break

                if pat.allow_overlaps:
                    # start matching from the next event in the sequence
                    pos += 1
                else:
                    # start matching after the current match ends
                    if pat.idx_end > 0:
                        pos = (
                            match_indices[i][pat.idx_end] + match_counts[i][pat.idx_end]
                        )
                    else:
                        pos = num_events

                idx_match_arrays += 1
                if idx_match_arrays == num_sequences:
                    # create more space for new outputs
                    match_seq_ids, match_indices, match_counts = _gen_arrays()
                    idx_match_arrays = 0

            else:
                pos += 1

    return (list_match_seq_ids, list_match_indices, list_match_counts)  # type: ignore


def match_sequence_here(
    pat: PatternInfo,
    pat_pos: int,
    events: ak.Array,
    events_pos: int,
    cut_match_indices: np.ndarray,
    cut_match_counts: np.ndarray,
) -> bool:
    """Match the events in the sequence starting at the given position of the pattern"""

    min_to_match = pat.event_min_counts[pat_pos]
    max_to_match = pat.event_max_counts[pat_pos]
    num_events = len(events)

    pos = events_pos
    while pos < events_pos + min_to_match:
        # more events to match than present in the sequence
        if pos >= num_events:
            return False
        if not match_event(events[pos], pat_pos):  # type: ignore
            return False
    cut_match_indices[pat_pos] = events_pos
    cut_match_counts[pat_pos] = min_to_match

    pos_limit = num_events
    if max_to_match > 0:
        pos_limit = min(pos_limit, events_pos + max_to_match)

    # check if the pattern has more events left to match
    assert pos == events_pos + min_to_match
    if pat_pos < pat.length - 1:
        while pos < pos_limit:
            if match_sequence_here(
                pat, pat_pos + 1, events, pos, cut_match_indices, cut_match_counts
            ):
                # if the rest of the pattern matches rest of the sequence, we have a match
                # since we already matched min_count copies of current event
                return True
            elif match_event(events[pos], pat_pos):  # type: ignore
                # we have some leeway to match more of the current event
                cut_match_counts[pat_pos] += 1
                pos += 1
            else:
                # we can't match any further of the pattern
                return False

        # we have matched all copies of the current event, onto the next
        if pos < num_events:
            # the sequence has more events to match to
            if match_sequence_here(
                pat, pat_pos + 1, events, pos, cut_match_indices, cut_match_counts
            ):
                return True
            else:
                return False
        else:
            # no events left in the sequence but the pattern has more left
            return False
    else:
        # pattern has no more events to match, so match as many of the current event as feasible
        while pos < pos_limit:
            if not match_event(events[pos], pat_pos):  # type: ignore
                break
            else:
                pos += 1
                cut_match_indices[pat_pos] += 1

        # check if the pattern needed to match at the end of the sequence
        if pat.match_seq_end and pos < num_events:
            return False
        else:
            return True
