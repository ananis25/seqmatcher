This folder hosts more explorations around analyzing sequences of events. 

* `vmsp-mining.ipynb` - This notebook implements the Verical Sequential Pattern Mining algorithm. When exploring a large list of sequences of events, it helps find the most frequent event subssequences (non contiguous). 

* `app_sunburst.py` - Streamlit app to visualize event sequences as an interactive, zoomable, Sunburst chart. The idea is copied off this observable [notebook](https://observablehq.com/@mikpanko/tennis-rallies-sunburst-chart) and uses the dataset of tennis shot sequences as a sample. By mapping event types to concentric arcs, this chart lets you quickly see how frequently a particular event is followed by another. For ex - the frequency of `serve_fault-ace` plays, or how often rallies start with `serve_middle-forehand-forehand-...`. 

    Refer to the original link for a description of how the data is encoded. The implementation here handles preparing the dataset and the `zoom` interaction server side, so the full dataset is never shipped to the browser. Thus, this should scale well to larger datasets. 

