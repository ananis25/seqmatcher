"""
This module implements the matching routine, by JIT compiling the code 
for each input separately using Numba. 
"""

import inspect
import sys
from typing import Callable, Union

import awkward as ak
import numba as nb
import numpy as np
from numba.typed import List
import awkward.layout as ak_layout

from . import codegen as cg
from .primitives import *

__all__ = ["match_pattern", "extract_pattern"]


# -----------------------------------------------------------
# matching sequences using the given pattern
# -----------------------------------------------------------


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


def type_getitem(typ):
    t = typ.type.getitem_at_check(typ)(typ, nb.int64)
    return t.return_type


_compiled_registry = {}


def match_pattern(
    pat: SeqPattern, sequences: ak.Array, debug_mode: bool = False
) -> ak.Array:
    """Wrapper that jit compiles numba functions to match the given pattern, and returns a list of
    identifiers for the matched sequences.
    """
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
        # clear file entry, unload the module, and the cached match_pattern dispatcher
        key = cg.get_cache_key(pat.pattern_str)
        cg.clear_cache(key)
        mod_name = f"code_cache.{key}"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        _compiled_registry.clear()

    # check if code exists as a cached file, else generate and store it
    if not cg.present_in_cache(pat.pattern_str):
        list_fns = cg.generate_match_fns(pat)

        code_str = ""
        for _imp in (
            "import numba as nb",
            "import numpy as np",
            "import awkward as ak",
            "from numba.typed import List",
        ):
            code_str += _imp + "\n"
        for _func in list_fns:
            code_str += "\n" + cg.ast_to_code(_func) + "\n"
        code_str += "\n" + inspect.getsource(match_sequence_here) + "\n"
        cg.write_to_cache(pat.pattern_str, code_str)

    jit_mod = cg.import_from_cache(pat.pattern_str)

    typ_seq_item = type_getitem(nb.typeof(sequences))
    typ_events_item = type_getitem(nb.typeof(sequences["events"]))
    typ_np_array = nb.typeof(np.zeros(5, np.int64))
    typ_inputs = (
        nb.typeof(pat_info),
        nb.typeof(sequences),
        nb.types.FunctionType(nb.bool_(typ_seq_item)),
        nb.types.FunctionType(
            nb.bool_(
                nb.typeof(pat_info),
                nb.int64,
                typ_events_item,
                nb.int64,
                typ_np_array,
                typ_np_array,
            )
        ),
    )
    if typ_inputs in _compiled_registry:
        fn = _compiled_registry[typ_inputs]
    else:
        fn = nb.jit(
            typ_inputs, nopython=True, locals={"pos": nb.int64, "pat_pos": nb.int64}
        )(_match_pattern)
        _compiled_registry[typ_inputs] = fn

    res = fn(
        pat_info,
        sequences,
        jit_mod.match_seq,
        jit_mod.match_sequence_here,
    )  # type: tuple[list[np.ndarray], ...]

    list_match_seq_ids, list_match_indices, list_match_counts = res
    match_seq_ids = np.concatenate(list_match_seq_ids)
    match_indices = np.concatenate(list_match_indices)
    match_counts = np.concatenate(list_match_counts)

    # filter down to legit matches
    valid_indices = match_seq_ids >= 0
    match_seq_ids = match_seq_ids[valid_indices]
    match_indices = match_indices[valid_indices]
    match_counts = match_counts[valid_indices]

    return ak.Array(
        {
            "seq_id": match_seq_ids,
            "evt_indices": match_indices,
            "evt_counts": match_counts,
        }
    )


def _match_pattern(
    pat: "PatternInfo",
    sequences: "ak.Array",
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
        list_match_indices.append(np.zeros((num_sequences, pat.length), dtype=np.int64))
        list_match_counts.append(np.zeros((num_sequences, pat.length), dtype=np.int64))
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
        if not match_seq(seq):  # type: ignore
            continue

        events = seq["events"]  # type:ignore
        num_events = len(events)

        # NOTE: this variable isn't mutated. Numba creates extra specializations, treating it as a literal in
        # addition to int64. We just want to avoid it.
        pat_pos = 0
        pos = 0
        start_before = 1 if pat.match_seq_start else num_events
        # NOTE: MAKE SURE THE LOOP VARIABLE IS INCREMENTED
        while pos < start_before:
            if match_seq_here(
                pat,
                pat_pos,
                events,
                pos,
                match_indices[out_idx],
                match_counts[out_idx],
            ):
                if pat.allow_overlaps:
                    # start matching from the next event in the sequence
                    pos += 1
                elif pat.idx_end >= 0:
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


@nb.jit(nopython=True)
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


# -----------------------------------------------------------
# matching sequences and extracting the matched subsequences
# -----------------------------------------------------------


def get_full_offsets_n_records(content: ak_layout.Content):
    """Events is an array with layout as a list-type content, or the same
    thing wrapped under one or more IndexedArray or UnmaskedArray.
    """
    from awkward._util import listtypes

    if not isinstance(content, listtypes):
        return get_full_offsets_n_records(content.content)
    else:
        if isinstance(content, ak_layout.RegularArray):
            starts = np.arange(0, len(content.content), content.size)
            return (starts, starts + content.size, content.content)
        else:
            return (
                np.asarray(content.starts),
                np.asarray(content.stops),
                content.content,
            )


def extract_pattern(pat: SeqPattern, sequences: ak.Array) -> ak.Array:
    """Extract the pattern from the sequences to yield new sequences."""

    match_res = match_pattern(pat, sequences)
    select_indices = match_res["seq_id"]

    matched_sequences: ak.Array = sequences[select_indices]  # type: ignore
    matched_events: ak.Array = matched_sequences["events"]  # type: ignore

    # we still need to subset the events for each sequence
    events_length = ak.num(matched_events)
    start_at = (
        ak.zeros_like(events_length)
        if pat.idx_start_event is None
        else match_res["evt_indices"][:, pat.idx_start_event]  # type: ignore
    )
    end_excl_at = (
        events_length
        if pat.idx_end_event is None
        else match_res["evt_indices"][:, pat.idx_end_event]  # type: ignore
        + match_res["evt_counts"][:, pat.idx_end_event]  # type: ignore
    )

    # the original events array is a ListOffsetArray or ListArray, so pull out the offsets off it
    full_starts, full_ends, full_content = get_full_offsets_n_records(
        matched_events.layout
    )
    select_starts = full_starts[select_indices] + start_at
    select_ends = full_starts[select_indices] + end_excl_at
    assert np.all(
        select_ends <= full_ends[select_indices]
    ), "shame, your math is incorrect"
    subset_matched_events = ak.Array(
        ak_layout.ListArray64(
            ak_layout.Index64(select_starts),
            ak_layout.Index64(select_ends),
            full_content,
        )
    )
    assert ak.is_valid(subset_matched_events)
    matched_sequences["events"] = subset_matched_events

    return matched_sequences
