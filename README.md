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

- **Word2Vec Training**: Trains Skip-gram Word2Vec model with configurable parameters
- **Three Configuration Testing**: Tests different vector sizes, epochs, and cluster counts
- **Optimal K Selection**: Uses elbow method to find optimal number of word clusters
- **Word Clustering**: Groups semantically similar words using K-means
- **BOW Vector Generation**: Creates normalized bag-of-words vectors for documents
- **Document Clustering**: Clusters documents using cosine distance metric
- **Visualization**: t-SNE and PCA plots of word clusters, elbow curves
- **Document Analysis**: Detailed examination of clustered documents with keywords

## Three Test Configurations

- **Config 1**: 10D vectors, 40 epochs, 5 word clusters (small)
- **Config 2**: 20D vectors, 50 epochs, 10 word clusters (medium)
- **Config 3**: 30D vectors, 60 epochs, 15 word clusters (large)

## Output Files

- `Word2Vec_BOW/bow_vectors_config{1-3}.npy`: Generated BOW vectors for each config
- `Word2Vec_BOW/document_clusters_config{1-3}.npy`: Document cluster labels
- `Word2Vec_BOW/elbow_inertias_config{1-3}.npy`: Inertia values for different k values
- `Word2Vec_BOW/word_clusters_tsne.png`: t-SNE word cluster visualization
- `Word2Vec_BOW/word_clusters_pca.png`: PCA word cluster visualization
- `Word2Vec_BOW/elbow_plot.png`: Elbow method visualization

## Data

The script expects a CSV file at `data/posts.csv` with columns:
- `title`: Post titles
- `keywords`: Post keywords (optional)