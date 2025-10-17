import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.doc2vec import Doc2Vec
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

def preprocessing(text):
    text = remove_stopwords(text.lower())
    return simple_preprocess(text)

def train(documents, configs):

    models = []

    for i, cfg in enumerate(configs):
        print(f'Training Model {i+1} with {cfg}')
        model = Doc2Vec(documents, **cfg, workers=4)
        models.append(model)
    
    return models
    
def return_embeddings(models):
    embeddings = [[model.dv[i] for i in range(len(documents))] for model in models]
    return embeddings

def save_document_clusters(df, labels, model_idx, k):
    """Save document clusters to text files"""
    results_dir = 'Doc2Vec/results'
    os.makedirs(results_dir, exist_ok=True)
    
    for cluster_id in range(max(labels) + 1):
        cluster_docs = df[labels == cluster_id]
        
        filename = f'{results_dir}/model{model_idx}_k{k}_cluster_{cluster_id}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Model {model_idx} | k={k} | Cluster {cluster_id} - {len(cluster_docs)} documents\n")
            f.write("=" * 60 + "\n\n")
            
            for idx, (_, row) in enumerate(cluster_docs.iterrows()):
                f.write(f"{idx+1}. {row['title']}\n")
                if 'keywords' in df.columns and pd.notna(row['keywords']):
                    f.write(f"   Keywords: {row['keywords']}\n")
                f.write("\n")

def cluster(embeddings, configs, output_dir, df, k_range=(3, 6)):
    i = 0

    for model_idx, emb in enumerate(embeddings, start=1):
        emb_norm = normalize(emb)
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(emb_norm)

        print(f'\nModel {model_idx} Results:')
        print(f'Configs: {configs[i]}')
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(emb_norm)
            score = silhouette_score(emb_norm, labels, metric='cosine')

            plt.figure(figsize=(8, 6))
            plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=30, alpha=0.7)
            plt.title(f'Model {model_idx} | k={k} | Silhouette={score:.4f}')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.tight_layout()

            filename = f'model{model_idx}_k{k}_sil{score:.3f}.png'
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

            print(f'k={k}: silhouette={score:.4f}')
            
            save_document_clusters(df, labels, model_idx, k)
        i += 1
    
if __name__ == '__main__':
    path = 'data/posts.csv'
    df = pd.read_csv(path)
    titles = df.title.tolist()

    documents = [gensim.models.doc2vec.TaggedDocument(preprocessing(doc), [i]) for i, doc in enumerate(titles)]
    configs = [
        {'vector_size': 10, 'min_count': 2, 'epochs': 40, 'window': 5},
        {'vector_size': 20, 'min_count': 2, 'epochs': 50, 'window': 5},
        {'vector_size': 30, 'min_count': 2, 'epochs': 60, 'window': 5}
    ]
    models = train(documents, configs)
    embeddings = return_embeddings(models)    

    output_dir = 'Doc2Vec/clusters'
    os.makedirs(output_dir, exist_ok=True)
    cluster(embeddings, configs, output_dir, df)