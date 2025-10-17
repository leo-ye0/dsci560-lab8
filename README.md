# DSCI 560 Lab 8 - Word2Vec BOW vs Doc2Vec Comparison

This project implements and compares Word2Vec-based Bag of Words (BOW) and Doc2Vec approaches for document embedding and clustering.

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

2. Run Word2Vec BOW implementation:
```bash
python Word2Vec_BOW/word2vec_bow.py
```

3. Run Doc2Vec implementation:
```bash
python Doc2Vec/doc2vec.py
```

## Features

### Core Implementation
- **Word2Vec Training**: Trains Skip-gram Word2Vec model with configurable parameters
- **Doc2Vec Training**: Trains Doc2Vec model for direct document embeddings
- **Document Clustering**: Clusters documents using cosine distance for k=3,4,5,6
- **Quality Metrics**: Silhouette score evaluation for clustering performance

### Analysis & Visualization
- **Document Cluster Visualization**: PCA plots of document embeddings for all k values
- **Document Cluster Analysis**: Text files containing all documents in each cluster
- **Multi-Configuration Comparison**: Tests three different embedding dimensions
- **Direct Comparison**: Same parameters and evaluation metrics for both approaches

## Three Test Configurations

Both Word2Vec BOW and Doc2Vec use identical configurations:
- **Config 1**: 10D vectors, 40 epochs, window=5, min_count=2
- **Config 2**: 20D vectors, 50 epochs, window=5, min_count=2  
- **Config 3**: 30D vectors, 60 epochs, window=5, min_count=2

## Output Files

### Word2Vec BOW Results
- `Word2Vec_BOW/clusters/`: PCA visualizations for each config and k value
- `Word2Vec_BOW/results/`: Document cluster text files (config{1-3}_k{3-6}_cluster_{0-n}.txt)

### Doc2Vec Results
- `Doc2Vec/clusters/`: PCA visualizations for each model and k value
- `Doc2Vec/results/`: Document cluster text files (model{1-3}_k{3-6}_cluster_{0-n}.txt)

## Methodology Comparison

### Word2Vec BOW Approach
1. Train Word2Vec on word tokens
2. Cluster words into semantic groups using optimal k (silhouette score)
3. Create document vectors as normalized word cluster frequencies
4. Cluster documents using cosine distance for k=3,4,5,6

### Doc2Vec Approach
1. Train Doc2Vec directly on documents
2. Extract document embeddings
3. Cluster documents using cosine distance for k=3,4,5,6

## Evaluation

Both approaches generate:
- **PCA visualizations** showing document clusters in 2D space
- **Silhouette scores** measuring cluster quality
- **Document cluster files** for qualitative analysis
- **Identical parameters** for fair comparison

## Data Requirements

The script expects a CSV file at `data/posts.csv` with columns:
- `title`: Post titles (required)
- `keywords`: Post keywords (optional, enhances analysis)