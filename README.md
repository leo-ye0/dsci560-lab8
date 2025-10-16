# DSCI 560 Lab 8 - Word2Vec BOW Implementation

This project implements a Word2Vec-based Bag of Words (BOW) representation using K-means clustering.

## Setup

### Prerequisites
- Python 3.11 (automatically managed by uv)
- uv package manager

### Installation

1. Install uv:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

2. Create Python 3.11 virtual environment:
```bash
uv venv --python 3.11
```

3. Activate virtual environment:
```bash
source .venv/bin/activate
```

4. Install dependencies:
```bash
uv pip install -r Word2Vec_BOW/requirements.txt
```

## Usage

1. Ensure your virtual environment is activated:
```bash
source .venv/bin/activate
```

2. Run the Word2Vec BOW implementation:
```bash
python Word2Vec_BOW/word2vec_bow.py
```

## Features

- **Word2Vec Training**: Trains Skip-gram Word2Vec model on text corpus
- **Optimal K Selection**: Uses elbow method to find optimal number of clusters
- **Word Clustering**: Groups semantically similar words using K-means
- **BOW Vector Generation**: Creates normalized bag-of-words vectors for documents
- **Visualization**: Generates elbow plot for cluster analysis

## Output Files

- `Word2Vec_BOW/bow_vectors.npy`: Generated BOW vectors
- `Word2Vec_BOW/elbow_plot.png`: Elbow method visualization
- `Word2Vec_BOW/elbow_inertias.npy`: Inertia values for different k values

## Data

The script expects a CSV file at `data/posts.csv` with columns:
- `title`: Post titles
- `keywords`: Post keywords (optional)