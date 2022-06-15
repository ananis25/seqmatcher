"""
Streamlit app to display a Sunburst chart using the custom component.

* Run it as: `streamlit run app_sunburst.py`
* Before first run, compile the frontend component by navigating to the `frontend` folder, 
and running: `pnpm run build`. 
"""

import os
import pyarrow as pa
import pyarrow.parquet
import awkward as ak
import streamlit as st

from st_sunburst_component.utils import hierarchy as h
from st_sunburst_component import sunburst_chart


st.set_page_config(layout="wide")


@st.experimental_memo
def read_hierarchy_table():
    """This routine takes a couple seconds to run. When running multiple times, store
    the hierarchy table one time, and load it there on.
    """
    dataset = pa.parquet.read_table("../datasets/tennis_shot_by_shot.parquet")
    evt_hierarchy, evt_names = h.convert_to_hierarchy(dataset)
    return evt_hierarchy, evt_names


num_levels_to_plot = 10
evt_hierarchy, event_names = read_hierarchy_table()

if "anchor_index" not in st.session_state:
    st.session_state["anchor_index"] = 0

st.title("Sunburst Chart")
# print("RENDERING anchor index: ", st.session_state["anchor_index"])

ancestors, subtree = h.get_ancestors_n_subtree(
    evt_hierarchy, st.session_state.anchor_index, num_levels_to_plot - 1
)

new_anchor_index = sunburst_chart(
    event_names, st.session_state.anchor_index, ancestors, subtree
)
# print("DESIRED anchor index: ", new_anchor_index)

if new_anchor_index is not None and new_anchor_index != st.session_state.anchor_index:
    st.session_state["anchor_index"] = new_anchor_index
    st.experimental_rerun()
