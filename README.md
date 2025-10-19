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

4. Run analysis implementation:
```bash
python analysis.py
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
- 
### Analysis Result
- `analysis/analysis_summary.csv`: store the evaluations

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

To quantitatively compare **Word2Vec-BOW** and **Doc2Vec**, we evaluated their clustering results on the original dataset (`posts.csv`) using three standard metrics:

| Metric | Description |
|---------|--------------|
| **Purity** | Measures how homogeneous each cluster is with respect to the true topic labels. |
| **NMI (Normalized Mutual Information)** | Measures the similarity between two clustering assignments, normalized between 0 and 1. |
| **ARI (Adjusted Rand Index)** | Evaluates the degree of agreement between two clusterings after adjusting for chance. |

### Results Overview
| Config | k | n_overlap | Purity_W2V | Purity_D2V | NMI | ARI |
|:------:|:--:|:----------:|:------------:|:------------:|:----:|:----:|
| 1 | 3–6 | 100 | **1.0000** | **1.0000** | 0.02–0.07 | ≈0.00 |
| 2 | 3–6 | 100 | **1.0000** | **1.0000** | 0.01–0.09 | ≈0.00 |
| 3 | 3–6 | 100 | **1.0000** | **1.0000** | 0.06–0.10 | ≈0.03–0.09 |

### Interpretation

- **Perfect Purity (1.0)** — Both Word2Vec-BOW and Doc2Vec successfully separate documents by their true topics.  
  Each cluster contains posts from a single topic category.

- **Low NMI / ARI values (<0.1)** — Despite identical topic separation quality, the two models organize documents differently in semantic space.  
  Their internal clustering structures are only weakly correlated.

- **Conclusion** —  
  Both embedding methods are effective for macro-level topic separation.  
  However, **Doc2Vec** and **Word2Vec-BOW** encode intra-topic relationships differently, reflecting distinct document-level semantics.  
  For topic identification, either model is suitable; for fine-grained semantic similarity, Doc2Vec tends to capture more individualized representations.


## Data Requirements

The script expects a CSV file at `data/posts.csv` with columns:
- `title`: Post titles (required)
- `keywords`: Post keywords (optional, enhances analysis)
