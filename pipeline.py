# pipeline.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def run_pipeline(df, k=3):
    df = df.select_dtypes(include=["number"])
    df = df.fillna(df.mean())

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(pca_result)

    return {
        "pca": pca_result.tolist(),
        "clusters": clusters.tolist(),
        "gene_means": df.mean().to_dict()
    }