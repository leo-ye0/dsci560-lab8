import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from collections import Counter
import os

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
    def __init__(self, k_bins=15, vector_size=300, min_count=2, epochs=40, window=5):
        self.k_bins = k_bins
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.window = window
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
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=4,
            sg=1
        )
        
        print(f"Word2Vec model trained with {len(self.word2vec_model.wv)} words")
        return self.word2vec_model
    
    def bin_words_by_distance(self):
        """Bin words based on distance metric"""
        if not self.word2vec_model:
            raise ValueError("Word2Vec model not trained yet")
        
        words = list(self.word2vec_model.wv.index_to_key)
        word_vectors = np.array([self.word2vec_model.wv[word] for word in words])
        
        print(f"Binning {len(words)} words into {self.k_bins} bins using distance metric...")
        
        # Create reference points using K-means centroids
        kmeans = KMeans(n_clusters=self.k_bins, random_state=42, n_init=10)
        kmeans.fit(word_vectors)
        centroids = kmeans.cluster_centers_
        
        # Assign words to bins based on distance to closest centroid
        self.word_to_cluster = {}
        for word in words:
            word_vector = self.word2vec_model.wv[word]
            distances = [np.linalg.norm(word_vector - centroid) for centroid in centroids]
            bin_id = np.argmin(distances)
            self.word_to_cluster[word] = bin_id
        
        print(f"Word binning completed. {len(self.word_to_cluster)} words assigned to bins.")
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
            words = cluster_analysis[cluster_id][:10]
            print(f"Cluster {cluster_id}: {', '.join(words)}")
        
        return cluster_analysis
    
    def find_optimal_k(self, k_range=None):
        """Find optimal k using elbow method"""
        if not self.word2vec_model:
            raise ValueError("Word2Vec model not trained yet")
        
        words = list(self.word2vec_model.wv.index_to_key)
        word_vectors = np.array([self.word2vec_model.wv[word] for word in words])
        
        if k_range is None:
            k_range = range(3, 7)
        
        
        silhouette_scores = []
        k_values = list(k_range)
        
        print(f"Testing k values: {k_values}")
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(word_vectors)
            score = silhouette_score(word_vectors, cluster_labels)
            silhouette_scores.append(score)
        
        best_idx = np.argmax(silhouette_scores)
        self.optimal_k = k_values[best_idx]
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, silhouette_scores, 'bo-')
        plt.axvline(x=self.optimal_k, color='r', linestyle='--', label=f'Optimal k={self.optimal_k}')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Optimal k')
        plt.legend()
        plt.grid(True)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(os.path.join(base_dir, 'silhouette_plot.png'))
        plt.show()
        
        print(f"Optimal k found: {self.optimal_k}")
        return self.optimal_k, silhouette_scores
    
    def visualize_word_clusters(self, max_words=500):
        """Visualize word clusters using t-SNE and/or PCA"""
        if not self.word2vec_model or not self.word_to_cluster:
            raise ValueError("Word2Vec model not trained or words not clustered yet")
        
        pass
    

    
    def _plot_document_pca(self, reduced, cluster_labels, config_name, k_value, score):
        """Create PCA visualization of document clusters"""
        print("Creating document PCA visualization...")
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.7, s=30)
        
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f'{config_name} | k={k_value} | Silhouette={score:.4f}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.tight_layout()
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        clusters_dir = os.path.join(base_dir, 'clusters')
        os.makedirs(clusters_dir, exist_ok=True)
        
        filename = f'{config_name}_k{k_value}_sil{score:.3f}.png'
        plt.savefig(f'{clusters_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_k_visualizations(self, config_name, bow_matrix, df):
        """Generate PCA visualizations for document clusters with all k values"""
        normalized_vectors = normalize(bow_matrix, norm='l2')
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(normalized_vectors)
        
        for k in range(3, 7):
            print(f"Generating document cluster visualization for k={k}...")
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_vectors)
            score = silhouette_score(normalized_vectors, cluster_labels, metric='cosine')
            
            self._plot_document_pca(reduced, cluster_labels, config_name, k, score)
            self.save_document_clusters(df, cluster_labels, config_name, k)
    
    def cluster_documents_cosine(self, bow_matrix, n_clusters=5):
        """Cluster documents using cosine distance metric"""
        print(f"Clustering {len(bow_matrix)} documents into {n_clusters} clusters using cosine distance...")
        
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
            
            sample_titles = cluster_docs['title'].head(5).tolist()
            for i, title in enumerate(sample_titles):
                print(f"  {i+1}. {title}")
            if 'keywords' in df.columns:
                all_keywords = ' '.join(cluster_docs['keywords'].dropna().astype(str))
                if all_keywords:
                    keywords_list = [kw.strip().lower() for kw in all_keywords.split(',') if kw.strip()]
                    common_keywords = Counter(keywords_list).most_common(5)
                    print(f"  Common keywords: {[kw for kw, count in common_keywords]}")
            
            print(f"  Avg title length: {cluster_docs['title'].str.len().mean():.1f} chars")
            if len(cluster_docs) > 1:
                print(f"  Title diversity: {len(set(cluster_docs['title'].str.lower()))/len(cluster_docs):.2f}")
        
        return doc_cluster_labels
    
    def save_document_clusters(self, df, doc_cluster_labels, config_name, k):
        """Save document clusters to text files"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        for cluster_id in range(max(doc_cluster_labels) + 1):
            cluster_docs = df[doc_cluster_labels == cluster_id]
            
            filename = f'{results_dir}/{config_name}_k{k}_cluster_{cluster_id}.txt'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"{config_name} | k={k} | Cluster {cluster_id} - {len(cluster_docs)} documents\n")
                f.write("=" * 60 + "\n\n")
                
                for idx, (_, row) in enumerate(cluster_docs.iterrows()):
                    f.write(f"{idx+1}. {row['title']}\n")
                    if 'keywords' in df.columns and pd.notna(row['keywords']):
                        f.write(f"   Keywords: {row['keywords']}\n")
                    f.write("\n")

def main():
    print("Loading posts.csv...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'posts.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} posts")
    configs = [
        {'vector_size': 10, 'min_count': 2, 'epochs': 40, 'window': 5},
        {'vector_size': 20, 'min_count': 2, 'epochs': 50, 'window': 5},
        {'vector_size': 30, 'min_count': 2, 'epochs': 60, 'window': 5}
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"CONFIGURATION {i}: {config}")
        print(f"{'='*60}")
        
        w2v_bow = Word2VecBOW(**config)
        
        corpus = w2v_bow.prepare_corpus(df)
        print(f"Prepared corpus with {len(corpus)} documents")
        
        w2v_bow.train_word2vec(corpus)
        
        optimal_k, silhouette_scores = w2v_bow.find_optimal_k()
        w2v_bow.k_bins = optimal_k
        
        w2v_bow.bin_words_by_distance()
        w2v_bow.analyze_clusters()
        
        bow_matrix = w2v_bow.transform_corpus(corpus)
        print(f"\nBOW matrix shape: {bow_matrix.shape}")
        
        config_name = f"config{i}"
        w2v_bow.generate_all_k_visualizations(config_name, bow_matrix, df)
        
        print(f"\nConfig {i} results saved:")
        print(f"- Visualizations: clusters/ folder")
        print(f"- Document clusters: results/ folder")
        print(f"- Optimal k: {optimal_k}")

if __name__ == "__main__":
    main()