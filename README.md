### Download data
Place the following files into `data/raw/`:

- connections_princeton.csv
- names.csv
- labels.csv



### Step 1 — Preprocessing
Run the notebook:
notebooks/preprocessing.ipynb

This loads the raw data from `data/raw/`, cleans it, normalizes IDs, and saves:
- data/preprocessed/processed_edges.csv
- data/preprocessed/processed_nodes.csv

### Step 2 — Network Analysis
Run the notebook:
notebooks/network_analysis.ipynb

This builds the directed graph, extracts the subgraph, computes centrality measures,
community detection, and saves:
- data/preprocessed/subgraph.pkl
- data/preprocessed/node_centralities.csv

### Step 3 — Visualization
Run the notebook:
notebooks/visualization.ipynb

This loads the processed subgraph + centralities and generates:
- degree distribution plots
- centrality comparison plots
- community visualizations
- heatmaps, etc.
