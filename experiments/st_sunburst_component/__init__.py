import os
import pyarrow as pa
import streamlit as st

import streamlit.components.v1 as components


def serialize_arrow_table(table: pa.Table) -> bytes:
    """Copied off Streamlit repository."""
    sink = pa.BufferOutputStream()
    writer = pa.RecordBatchStreamWriter(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return sink.getvalue().to_pybytes()


_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "sunburst_chart",
        url="http://localhost:3000",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/dist")
    _component_func = components.declare_component("sunburst_chart", path=build_dir)


def sunburst_chart(all_event_names, anchor_index, ancestors, subtree):
    return _component_func(
        key="sunburst_chart",
        anchorIndex=anchor_index,
        eventNames=all_event_names,
        ancestors=serialize_arrow_table(ancestors),
        subtree=serialize_arrow_table(subtree),
    )
