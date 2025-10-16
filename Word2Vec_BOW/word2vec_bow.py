import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class Word2VecBOW:
    def __init__(self, k_bins=15, vector_size=300):
        self.k_bins = k_bins
        self.vector_size = vector_size
        self.word2vec_model = None
        self.kmeans = None
        self.word_to_cluster = {}
        self.optimal_k = None
        
    def preprocess_text(self, text):
        """Clean and tokenize text"""
        if pd.isna(text):
            return []
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        return tokens
    
    def prepare_corpus(self, df):
        """Prepare corpus from DataFrame"""
        # Combine title and keywords for richer text representation
        corpus = []
        for _, row in df.iterrows():
            title_tokens = self.preprocess_text(row['title'])
            keyword_tokens = self.preprocess_text(row['keywords']) if 'keywords' in df.columns else []
            
            # Combine tokens
            document_tokens = title_tokens + keyword_tokens
            if document_tokens:
                corpus.append(document_tokens)
        
        return corpus
    
    def train_word2vec(self, corpus):
        """Train Word2Vec model"""
        print(f"Training Word2Vec model with {len(corpus)} documents...")
        
        self.word2vec_model = Word2Vec(
            sentences=corpus,
            vector_size=self.vector_size,
            window=5,
            min_count=2,
            workers=4,
            sg=1
        )
        
        print(f"Word2Vec model trained with {len(self.word2vec_model.wv)} words")
        return self.word2vec_model
    
    def cluster_words(self):
        """Cluster words using K-means"""
        if not self.word2vec_model:
            raise ValueError("Word2Vec model not trained yet")
        
        # Get word vectors
        words = list(self.word2vec_model.wv.index_to_key)
        word_vectors = np.array([self.word2vec_model.wv[word] for word in words])
        
        print(f"Clustering {len(words)} words into {self.k_bins} bins...")
        
        # Perform K-means clustering
        self.kmeans = KMeans(n_clusters=self.k_bins, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(word_vectors)
        
        # Create word to cluster mapping
        self.word_to_cluster = {words[i]: cluster_labels[i] for i in range(len(words))}
        
        print(f"Words clustered into {self.k_bins} bins")
        return self.word_to_cluster
    
    def create_bow_vector(self, document_tokens):
        """Create bag-of-words vector for a document"""
        bow_vector = np.zeros(self.k_bins)
        
        for token in document_tokens:
            if token in self.word_to_cluster:
                cluster_id = self.word_to_cluster[token]
                bow_vector[cluster_id] += 1
        
        # Normalize by document length
        total_words = len(document_tokens)
        if total_words > 0:
            bow_vector = bow_vector / total_words
            
        return bow_vector
    
    def transform_corpus(self, corpus):
        """Transform entire corpus to BOW vectors"""
        print(f"Transforming {len(corpus)} documents to BOW vectors...")
        
        bow_matrix = []
        for document_tokens in corpus:
            bow_vector = self.create_bow_vector(document_tokens)
            bow_matrix.append(bow_vector)
        
        return np.array(bow_matrix)
    
    def analyze_clusters(self):
        """Analyze word clusters"""
        if not self.word_to_cluster:
            raise ValueError("Words not clustered yet")
        
        cluster_analysis = {}
        for word, cluster_id in self.word_to_cluster.items():
            if cluster_id not in cluster_analysis:
                cluster_analysis[cluster_id] = []
            cluster_analysis[cluster_id].append(word)
        
        print("\nCluster Analysis (showing first 10 clusters):")
        for cluster_id in sorted(cluster_analysis.keys())[:10]:
            words = cluster_analysis[cluster_id][:10]  # Show first 10 words
            print(f"Cluster {cluster_id}: {', '.join(words)}")
        
        return cluster_analysis
    
    def find_optimal_k(self, k_range=None):
        """Find optimal k using elbow method"""
        if not self.word2vec_model:
            raise ValueError("Word2Vec model not trained yet")
        
        words = list(self.word2vec_model.wv.index_to_key)
        word_vectors = np.array([self.word2vec_model.wv[word] for word in words])
        
        if k_range is None:
            k_range = range(5, min(26, len(words)//3))
        
        inertias = []
        k_values = list(k_range)
        
        print(f"Testing k values: {k_values}")
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(word_vectors)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        elbow_idx = np.argmax(diffs2) + 1
        self.optimal_k = k_values[elbow_idx]
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertias, 'bo-')
        plt.axvline(x=self.optimal_k, color='r', linestyle='--', label=f'Optimal k={self.optimal_k}')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.legend()
        plt.grid(True)
        plt.savefig('/home/yutaoye/Desktop/dsci560-lab8/Word2Vec_BOW/elbow_plot.png')
        plt.show()
        
        print(f"Optimal k found: {self.optimal_k}")
        return self.optimal_k, inertias
    
    def visualize_word_clusters(self, method='both', max_words=500):
        """Visualize word clusters using t-SNE and/or PCA"""
        if not self.word2vec_model or not self.word_to_cluster:
            raise ValueError("Word2Vec model not trained or words not clustered yet")
        
        # Get word vectors and labels
        words = list(self.word2vec_model.wv.index_to_key)[:max_words]
        word_vectors = np.array([self.word2vec_model.wv[word] for word in words])
        cluster_labels = [self.word_to_cluster[word] for word in words]
        
        if method in ['tsne', 'both']:
            self._plot_tsne(word_vectors, cluster_labels, words)
        
        if method in ['pca', 'both']:
            self._plot_pca(word_vectors, cluster_labels, words)
    
    def _plot_tsne(self, word_vectors, cluster_labels, words):
        """Create t-SNE visualization of word clusters"""
        print("Creating t-SNE visualization...")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        word_vectors_2d = tsne.fit_transform(word_vectors)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
        
        # Add word labels for a subset of points
        for i in range(0, len(words), max(1, len(words)//50)):
            plt.annotate(words[i], (word_vectors_2d[i, 0], word_vectors_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f'Word Clusters Visualization (t-SNE)\nk={self.k_bins} clusters')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        
        plt.savefig('/home/yutaoye/Desktop/dsci560-lab8/Word2Vec_BOW/word_clusters_tsne.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_pca(self, word_vectors, cluster_labels, words):
        """Create PCA visualization of word clusters"""
        print("Creating PCA visualization...")
        
        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        word_vectors_2d = pca.fit_transform(word_vectors)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
        
        # Add word labels for a subset of points
        for i in range(0, len(words), max(1, len(words)//50)):
            plt.annotate(words[i], (word_vectors_2d[i, 0], word_vectors_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f'Word Clusters Visualization (PCA)\nk={self.k_bins} clusters\n'
                 f'Explained variance: {pca.explained_variance_ratio_[0]:.2%} + {pca.explained_variance_ratio_[1]:.2%}')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.grid(True, alpha=0.3)
        
        plt.savefig('/home/yutaoye/Desktop/dsci560-lab8/Word2Vec_BOW/word_clusters_pca.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def cluster_documents_cosine(self, bow_matrix, n_clusters=5):
        """Cluster documents using cosine distance metric"""
        print(f"Clustering {len(bow_matrix)} documents into {n_clusters} clusters using cosine distance...")
        
        # Use KMeans with cosine distance (equivalent to using normalized vectors with euclidean)
        normalized_vectors = normalize(bow_matrix, norm='l2')
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        doc_cluster_labels = kmeans.fit_predict(normalized_vectors)
        
        return doc_cluster_labels
    
    def analyze_document_clusters(self, df, doc_cluster_labels):
        """Analyze document clusters"""
        print(f"\nDocument Cluster Analysis:")
        
        for cluster_id in range(max(doc_cluster_labels) + 1):
            cluster_docs = df[doc_cluster_labels == cluster_id]
            print(f"\nCluster {cluster_id} ({len(cluster_docs)} documents):")
            
            # Show sample titles
            sample_titles = cluster_docs['title'].head(5).tolist()
            for i, title in enumerate(sample_titles):
                print(f"  {i+1}. {title}")
            
            # Show common keywords if available
            if 'keywords' in df.columns:
                all_keywords = ' '.join(cluster_docs['keywords'].dropna().astype(str))
                if all_keywords:
                    keywords_list = [kw.strip().lower() for kw in all_keywords.split(',') if kw.strip()]
                    common_keywords = Counter(keywords_list).most_common(5)
                    print(f"  Common keywords: {[kw for kw, count in common_keywords]}")
            
            # Show cluster characteristics
            print(f"  Avg title length: {cluster_docs['title'].str.len().mean():.1f} chars")
            if len(cluster_docs) > 1:
                print(f"  Title diversity: {len(set(cluster_docs['title'].str.lower()))/len(cluster_docs):.2f}")
        
        return doc_cluster_labels

def main():
    # Load data
    print("Loading posts.csv...")
    df = pd.read_csv('/home/yutaoye/Desktop/dsci560-lab8/data/posts.csv')
    print(f"Loaded {len(df)} posts")
    
    # Initialize Word2Vec BOW (k_bins will be optimized)
    w2v_bow = Word2VecBOW(vector_size=100)
    
    # Prepare corpus
    corpus = w2v_bow.prepare_corpus(df)
    print(f"Prepared corpus with {len(corpus)} documents")
    
    # Train Word2Vec
    w2v_bow.train_word2vec(corpus)
    
    # Find optimal k using elbow method
    optimal_k, inertias = w2v_bow.find_optimal_k()
    w2v_bow.k_bins = optimal_k
    
    # Cluster words with optimal k
    w2v_bow.cluster_words()
    
    # Analyze clusters
    w2v_bow.analyze_clusters()
    
    # Visualize word clusters
    w2v_bow.visualize_word_clusters(method='both')
    
    # Transform corpus to BOW vectors
    bow_matrix = w2v_bow.transform_corpus(corpus)
    print(f"\nBOW matrix shape: {bow_matrix.shape}")
    
    # Cluster documents using cosine distance
    doc_clusters = w2v_bow.cluster_documents_cosine(bow_matrix, n_clusters=5)
    w2v_bow.analyze_document_clusters(df, doc_clusters)
    
    # Save results
    np.save('/home/yutaoye/Desktop/dsci560-lab8/Word2Vec_BOW/bow_vectors.npy', bow_matrix)
    np.save('/home/yutaoye/Desktop/dsci560-lab8/Word2Vec_BOW/elbow_inertias.npy', np.array(inertias))
    np.save('/home/yutaoye/Desktop/dsci560-lab8/Word2Vec_BOW/document_clusters.npy', doc_clusters)
    
    # Display sample vectors
    print("\nSample BOW vectors (first 3 documents):")
    for i in range(min(3, len(bow_matrix))):
        print(f"Document {i}: {bow_matrix[i][:10]}...")  # Show first 10 dimensions
    
    print(f"\nResults saved:")
    print(f"- BOW vectors: Word2Vec_BOW/bow_vectors.npy")
    print(f"- Document clusters: Word2Vec_BOW/document_clusters.npy")
    print(f"- Elbow plot: Word2Vec_BOW/elbow_plot.png")
    print(f"- Word clusters (t-SNE): Word2Vec_BOW/word_clusters_tsne.png")
    print(f"- Word clusters (PCA): Word2Vec_BOW/word_clusters_pca.png")
    print(f"- Optimal k: {optimal_k}")

if __name__ == "__main__":
    main()