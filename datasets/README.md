This folder hosts datasets recording sequences of events. The original data sources are often csv/json, which we convert to the parquet format for persistence. 

1. Foursquare check-in dataset
* source - https://sites.google.com/site/yangdingqi/home/foursquare-dataset#h.p_ID_46
* summary - "This dataset contains check-ins in NYC and Tokyo collected for about 10 month (from 12 April 2012 to 16 February 2013). It contains 227,428 check-ins in New York city and 573,703 check-ins in Tokyo. Each check-in is associated with its time stamp, its GPS coordinates and its semantic meaning (represented by fine-grained venue-categories)."

2. Tennis shot by shot data
* source - https://observablehq.com/@mikpanko/tennis-shot-by-shot-data
* summary - "MCP match records contain shot-by-shot data for every point of a match, including the type of shot, direction of shot, depth of returns, types of errors, and more."
* notes - The dataset is originally from [link](https://github.com/JeffSackmann/tennis_MatchChartingProject). The dataset in the observable notebooks cleans it up and organizes it. 