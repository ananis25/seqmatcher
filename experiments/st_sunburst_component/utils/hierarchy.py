"""
This module hosts routines to construct a hierarchy of events with frequencies 
given by how often an event is followed by another. The dataset is a list of 
event sequences. 
"""

import awkward as ak
import pyarrow as pa
import numba as nb
import numpy as np
from numba.typed import Dict, List
from numba import int64


def make_events_categorical(
    arr_seq_event_names: pa.ListArray,
) -> tuple[pa.ListArray, list[str]]:
    """Convert the event names from an array of strings to categories. Integer comparisons are
    quicker than strings.
    """
    event_names_catg = pa.compute.dictionary_encode(arr_seq_event_names.flatten())

    # reassemble the list array
    catg_array = pa.ListArray.from_arrays(
        arr_seq_event_names.offsets, event_names_catg.indices
    )
    catg_names = event_names_catg.dictionary.tolist()
    return catg_array, catg_names


def make_event_hierarchy(arr_seq_events: ak.Array) -> pa.Table:
    """
    Go over the array of event sequences, and return a hierarchy of events. We serialize this
    data to plot it in the frontend, and creating a lot of python dicts is expensive. So, the
    output is tabular, with fields that refer to the corresponding parent and child event indices
    for each node.

    Args:
        arr_seq_events (ListArray): array where each element is a sequence of events.
    """

    (
        indices,
        nodes,
        depths,
        counts,
        parents,
        term_seq_offsets,
        term_seq_values,
        children_offsets,
        children_values,
    ) = _draw_paths(arr_seq_events)

    # arrow list-item type is nullable by default, but our arrays are not and we don't want awkward to think they are.
    term_seq_array = ak.from_arrow(
        pa.ListArray.from_arrays(children_offsets, children_values)
    )
    term_seq_array = ak.fill_none(term_seq_array, -100, axis=1)
    children_array = ak.from_arrow(
        pa.ListArray.from_arrays(children_offsets, children_values)
    )
    children_array = ak.fill_none(children_array, -100, axis=1)

    return ak.Array(
        {
            "index": indices,
            "event": nodes,
            "depth": depths,
            "count": counts,
            "parent_index": parents,
            "child_indices": children_array,
            "terminal_seq_indices": term_seq_array,
        }
    )


@nb.njit
def _make_array(list_ints: list[int]) -> np.ndarray:
    """Numba typed lists give me grief when converting to numpy, so construct it manually."""
    num = len(list_ints)
    arr = np.zeros(num, np.int64)
    for i in range(num):
        arr[i] = list_ints[i]
    return arr


@nb.njit
def _draw_paths(data: ak.Array) -> tuple[np.ndarray, ...]:
    """
    Function to do the leg work of calculating the hierarchy of events.
    """

    nodes = []  # event name ids
    depths = []  # depth of the current node
    counts = []  # num of sequences termination or passing this node
    parents = []  # index for the node previous to this one
    list_children = []
    term_sequences = []  # list of sequence ids which terminate at the current event

    def add_row(a, b, c, d):
        nodes.append(a)
        depths.append(b)
        counts.append(c)
        parents.append(d)
        list_children.append(Dict.empty(int64, int64))
        term_sequences.append(List.empty_list(int64))

    ROOT_EVENT_ID = -100  # any negative int would do
    add_row(ROOT_EVENT_ID, 0, 0, -1)

    for seq_id, seq in enumerate(data):
        idx = 0  # start from the root event each time
        depth = 0  # root event is at depth 0

        for evt in seq:
            depth += 1
            # increment the count at the current node
            counts[idx] += 1

            # check if this event was encountered following the current node before
            if evt not in list_children[idx]:
                # add a new node
                add_row(evt, depth, 1, idx)
                # add new node to the list of children of the current node
                list_children[idx][evt] = len(nodes) - 1

            # move on to the next event
            idx = list_children[idx][evt]

        # the sequence terminates at this event
        term_sequences[idx].append(seq_id)
        # increment the count for this node too; not recursed on so `add_row` calls to do it
        counts[idx] += 1

    num_nodes = len(nodes)

    term_seq_offsets = np.zeros(1 + num_nodes, np.int32)
    term_seq_values = []
    for i, evts in enumerate(term_sequences):
        term_seq_offsets[i + 1] = term_seq_offsets[i] + len(evts)
        term_seq_values.extend(evts)

    children_offsets = np.zeros(1 + num_nodes, np.int32)
    children_values = []
    for i, children in enumerate(list_children):
        children_offsets[i + 1] = children_offsets[i] + len(children)
        children_values.extend(children.values())

    return (
        np.arange(len(nodes), dtype=np.int64),
        _make_array(nodes),
        _make_array(depths),
        _make_array(counts),
        _make_array(parents),
        term_seq_offsets,
        _make_array(term_seq_values),
        children_offsets,
        _make_array(children_values),
    )


@nb.njit
def compute_coordinates(
    hierarchy: ak.Array, bound_num_children: int
) -> tuple[np.ndarray, np.ndarray]:
    """This could be delegated to d3 in the frontend instead, but we only ship enough data
    to plot the current chart, which is insufficient for transitions. Transitions are too
    cool to leave out.
    """
    counts = hierarchy["count"]
    child_indices = hierarchy["child_indices"]

    # we want to sort the children indices by their node counts
    arr_counts = np.zeros(bound_num_children, dtype=np.int64)
    arr_indices = np.zeros(bound_num_children, dtype=np.int64)
    tmp_count = 0
    tmp_idx = 0

    # add coordinates since transitions aren't possible if we leave this to the frontend
    num_nodes = len(counts)
    x0 = np.zeros(num_nodes)
    x1 = np.zeros(num_nodes)

    x0[0] = 0
    x1[0] = 1
    for node_idx in range(num_nodes):
        children = child_indices[node_idx]
        num_children = len(children)
        for i, child_idx in enumerate(children):
            arr_counts[i] = counts[child_idx]
            arr_indices[i] = child_idx

        # now we sort `arr_indices` and `arr_counts` by the count values at respective indices
        for i in range(num_children - 1):
            min_idx = i
            for j in range(i + 1, num_children):
                if arr_counts[j] < arr_counts[min_idx]:
                    min_idx = j

            if min_idx != i:
                tmp_count = arr_counts[i]
                arr_counts[i] = arr_counts[min_idx]
                arr_counts[min_idx] = tmp_count

                tmp_idx = arr_indices[i]
                arr_indices[i] = arr_indices[min_idx]
                arr_indices[min_idx] = tmp_idx

        cursor = x0[node_idx]
        incr_unit = max(0, (x1[node_idx] - x0[node_idx]) / counts[node_idx])
        for i in range(num_children):
            child_idx = arr_indices[i]
            x0[child_idx] = cursor
            cursor = min(cursor + arr_counts[i] * incr_unit, 1)
            x1[child_idx] = cursor

    return x0, x1


def convert_to_hierarchy(dataset: pa.Table) -> tuple[ak.Array, list[str]]:
    """Convert the input dataset to a hierarchy of events."""

    # compute the hierarchy of events
    events = dataset["events"].combine_chunks()
    seq_event_names = pa.ListArray.from_arrays(
        events.offsets, events.flatten().field("_eventName")
    )
    seq_event_catg, evt_catg_names = make_events_categorical(seq_event_names)
    awk_seq = ak.from_arrow(seq_event_catg)
    evt_hierarchy = make_event_hierarchy(ak.values_astype(awk_seq, np.int64))

    # calculate the x-coordinates
    bound_num_children: int = ak.max(ak.num(evt_hierarchy["child_indices"]))  # type: ignore
    x0_arr, x1_arr = compute_coordinates(evt_hierarchy, bound_num_children)

    evt_hierarchy["x0"] = x0_arr
    evt_hierarchy["x1"] = x1_arr
    return evt_hierarchy, evt_catg_names


@nb.njit
def compute_ancestry(data: ak.Array, anchor_index: int) -> np.ndarray:
    take_list = []
    anchor = data[anchor_index]
    take_list.append(anchor_index)

    parent = anchor.parent_index
    while parent >= 0:
        take_list.append(parent)
        parent = data[parent].parent_index
    return _make_array(take_list)


@nb.njit
def compute_subtree(data: ak.Array, anchor_index: int, num_levels: int) -> np.ndarray:
    take_list = []

    max_depth = data[anchor_index].depth + num_levels - 1
    min_width = (data[anchor_index].x1 - data[anchor_index].x0) / 720  # half-degree

    node_stack = []
    node_stack.append(anchor_index)
    take_list.append(anchor_index)

    # now recurse on the children nodes until max depth is reached
    while len(node_stack) > 0:
        node_idx = node_stack.pop()
        node = data[node_idx]
        if not node.depth > max_depth:
            for child_idx in node.child_indices:
                if (data[child_idx].x1 - data[child_idx].x0) > min_width:
                    node_stack.append(child_idx)
                    take_list.append(child_idx)
    return _make_array(take_list)


def prep_arrow_table(dataset: ak.Array) -> pa.Table:
    """Convert the input array to an arrow table to send to the frontend."""
    result = ak.to_arrow_table(dataset, list_to32=True)
    result = result.cast(
        pa.schema(
            [
                ("index", pa.int32()),
                ("event", pa.int32()),
                ("depth", pa.int32()),
                ("count", pa.int32()),
                ("parent_index", pa.int32()),
                ("child_indices", pa.list_(pa.int32())),
                ("terminal_seq_indices", pa.list_(pa.int32())),
                ("x0", pa.float64()),
                ("x1", pa.float64()),
                ("percent", pa.float64()),
            ]
        )
    )
    return result


def get_ancestors_n_subtree(
    hierarchy: ak.Array, anchor_idx: int, num_levels: int
) -> tuple[pa.Table, pa.Table]:
    """Get the ancestors and the subtree of the given node."""
    total_count = hierarchy["count"][0]

    # fetch the ancestors
    select_indices = compute_ancestry(hierarchy, anchor_idx)
    select_data: ak.Array = hierarchy[select_indices]  # type: ignore
    select_data["percent"] = 100 * select_data["count"] / total_count
    ancestors = prep_arrow_table(select_data)

    # fetch the subtree
    select_indices = compute_subtree(hierarchy, anchor_idx, num_levels)
    select_data: ak.Array = hierarchy[select_indices]  # type: ignore
    select_data["percent"] = 100 * select_data["count"] / total_count
    subtree = prep_arrow_table(select_data)

    return ancestors, subtree
