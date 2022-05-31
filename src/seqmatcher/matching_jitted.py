"""
This module implements the matching routine, by JIT compiling the code 
for each input separately using Numba. 
"""

from typing import Callable

import numpy as np
import numba as nb
import awkward as ak
from numba.typed import List

from .primitives import *
from . import codegen as cg


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


def match_pattern(
    pat: SeqPattern, sequences: ak.Array, debug_mode: bool = False
) -> ak.Array:
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

    if debug_mode:
        cg.clear_cache()
    # check if code exists as a cached file, else generate and store it
    if not cg.present_in_cache(pat.pattern_str):
        list_fns = cg.generate_match_fns(pat)
        list_fns.append(cg.make_ast(match_sequence_here))  # type: ignore

        code_str = ""
        for _imp in (
            "import numba as nb",
            "import numpy as np",
            "import awkward as ak",
        ):
            code_str += _imp + "\n"
        for _func in list_fns:
            code_str += "\n" + cg.ast_to_code(_func) + "\n"
        cg.write_to_cache(pat.pattern_str, code_str)

    jit_mod = cg.import_from_cache(pat.pattern_str)
    res = _match_pattern(
        pat_info, sequences, jit_mod.match_seq, jit_mod.match_sequence_here
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


@nb.jit(nopython=True)
def _match_pattern(
    pat: PatternInfo,
    sequences: ak.Array,
    match_seq: Callable[..., bool],
    match_seq_here: Callable[..., bool],
) -> tuple[list[np.ndarray], ...]:
    """Match the pattern against the sequences"""
    num_sequences = len(sequences)

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
    out_idx = 0  # index into the current set of match arrays, resets each time we run out of space

    for seq_id in range(num_sequences):
        seq = sequences[seq_id]

        # filter against the sequence properties
        if not match_seq(seq):
            continue
        events = seq["events"]  # type:ignore
        num_events = len(events)

        pos = 0
        start_before = 1 if pat.match_seq_start else pat.length
        # NOTE: MAKE SURE THE LOOP VARIABLE IS INCREMENTED
        while pos < start_before:
            if match_seq_here(
                pat,
                0,
                events,
                pos,
                match_indices[out_idx],
                match_counts[out_idx],
            ):
                if pat.allow_overlaps:
                    # start matching from the next event in the sequence
                    pos += 1
                elif pat.idx_end > 0:
                    # start matching after the current match ends
                    pos = (
                        match_indices[out_idx][pat.idx_end]
                        + match_counts[out_idx][pat.idx_end]
                    )
                else:
                    # subsequences go until the end by default
                    pos = num_events

                # fill in the output arrays

                # match_indices and match_counts arrays were already mutated while matching
                match_seq_ids[out_idx] = seq_id
                # update the running output index
                out_idx += 1
                # create more space if needed
                if out_idx == num_sequences:
                    match_seq_ids, match_indices, match_counts = _gen_arrays()
                    out_idx = 0

                # conclude if we need to match only the first occurrence
                if not pat.match_all:
                    break

            else:
                pos += 1

    return (list_match_seq_ids, list_match_indices, list_match_counts)  # type: ignore


@nb.jit(nopython=True, cache=True)
def match_sequence_here(
    pat: "PatternInfo",
    pat_pos: "int",
    events: "ak.Array",
    events_pos: "int",
    cut_match_indices: "np.ndarray",
    cut_match_counts: "np.ndarray",
) -> "bool":
    """Match the events in the sequence starting at the given position of the pattern"""

    min_to_match = pat.event_min_counts[pat_pos]
    max_to_match = pat.event_max_counts[pat_pos]
    num_events = len(events)

    pos = events_pos
    # NOTE: MAKE SURE THE LOOP VARIABLE IS INCREMENTED
    while pos < events_pos + min_to_match:
        # more events to match than present in the sequence
        if pos >= num_events:
            return False
        if not match_event(pat_pos, events, pos, cut_match_indices):  # type: ignore
            return False
        pos += 1
    cut_match_indices[pat_pos] = events_pos
    cut_match_counts[pat_pos] = min_to_match

    pos_limit = num_events
    if max_to_match > 0:
        pos_limit = min(pos_limit, events_pos + max_to_match)

    # check if the pattern has more events left to match
    assert pos == events_pos + min_to_match
    if pat_pos < pat.length - 1:
        # NOTE: MAKE SURE THE LOOP VARIABLE IS INCREMENTED
        while pos < pos_limit:
            if match_sequence_here(
                pat, pat_pos + 1, events, pos, cut_match_indices, cut_match_counts
            ):
                # if the rest of the pattern matches rest of the sequence, we have a match
                # since we already matched min_count copies of current event
                return True
            elif match_event(pat_pos, events, pos, cut_match_indices):  # type: ignore
                # we have some leeway to match more of the current event
                cut_match_counts[pat_pos] += 1
                pos += 1
            else:
                # we couldn't match any further of the pattern
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
        # NOTE: MAKE SURE THE LOOP VARIABLE IS INCREMENTED
        while pos < pos_limit:
            if not match_event(pat_pos, events, pos, cut_match_indices):  # type: ignore
                break
            else:
                pos += 1
                cut_match_indices[pat_pos] += 1

        # check if the pattern needed to match at the end of the sequence
        if pat.match_seq_end and pos < num_events:
            return False
        else:
            return True
