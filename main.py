import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Cancer Gene Expression Dashboard",
    layout="wide"
)

st.title("Cancer Gene Expression Explorer")


# ==========================
# LOAD DATA
# ==========================

file_path = "data/large_cancer_gene_expression.csv"
df = pd.read_csv(file_path)


# ==========================
# CLEAN DATA
# ==========================

if "Sample" in df.columns:
    df = df.set_index("Sample")
else:
    df = df.set_index(df.columns[0])

df = df.select_dtypes(include=["number"])
df = df.fillna(df.mean())


# ==========================
# SIDEBAR CONTROLS
# ==========================

st.sidebar.title("Controls")

selected_genes = st.sidebar.multiselect(
    "Select genes to view",
    df.columns,
    default=list(df.columns[:5])
)

k = st.sidebar.slider(
    "Number of clusters (KMeans)",
    min_value=2,
    max_value=6,
    value=3
)


# ==========================
# DATA PREVIEW
# ==========================

st.subheader("Dataset Preview")
st.dataframe(df[selected_genes].head())


# ==========================
# GENE STATISTICS
# ==========================

st.subheader("Gene Expression Statistics")

gene_means = df.mean(axis=0)

col1, col2 = st.columns(2)

with col1:
    st.write("Top expressed genes")
    st.write(gene_means.sort_values(ascending=False).head(10))

with col2:
    st.write("Lowest expressed genes")
    st.write(gene_means.sort_values(ascending=True).head(10))


# ==========================
# CORRELATION HEATMAP
# ==========================

st.subheader("Gene Correlation Heatmap")

corr = df[selected_genes].corr()

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(corr, cmap="coolwarm", ax=ax)

st.pyplot(fig)


# ==========================
# PCA + CLUSTERING
# ==========================

st.subheader("PCA + Clustering")

scaled = StandardScaler().fit_transform(df)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)

pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])


kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(pca_df)

pca_df["Cluster"] = clusters


fig2, ax2 = plt.subplots()

scatter = ax2.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    c=pca_df["Cluster"],
    cmap="viridis",
    alpha=0.7
)

ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_title("PCA with KMeans Clusters")

st.pyplot(fig2)


# ==========================
# FULL DATA
# ==========================

st.subheader("Full Dataset")

st.dataframe(df[selected_genes])