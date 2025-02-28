# [Learning to Explore and Exploit with GNNs For Unsupervised Combinatorial Optimization](https://openreview.net/forum?id=vaJ4FObpXN)


# Data
This folder contains datasets used for training and evaluating the machine learning models in this repository.

The codebase assumes that graphs are represented as [NetworkX](https://networkx.org/) graphs.

## Data Format

- The `load_data('path')` method expects a directory containing:
  - `train_graphs.pkl` – A list of NetworkX graphs for training.
  - `test_graphs.pkl` – A list of NetworkX graphs for evaluation.

## Available Datasets

- **Small Datasets**: Training and evaluation data for *RB200-300* and *BA200-300* are included in the repository.
- **Large Datasets**: For larger datasets, please download them from the Google Drive link below:

📂 **[Google Drive Link](#)** (Replace `#` with the actual link)
