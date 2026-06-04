import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================
# LOAD DATA SAFELY
# ==========================

file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "large_cancer_gene_expression.csv"   
)

df = pd.read_csv(file_path)

print("Original shape:", df.shape)
print(df.head())


# ==========================
# AUTO-DETECT FORMAT
# ==========================

# Case A: Sample column exists
if "Sample" in df.columns:
    df = df.set_index("Sample")

# Case B: first column is genes or IDs
elif df.columns[0].lower() in ["gene", "id", "unnamed: 0"]:
    df = df.set_index(df.columns[0])


# ==========================
# KEEP ONLY NUMERIC DATA
# ==========================

df = df.apply(pd.to_numeric, errors="coerce")

print("\nCleaned data shape:", df.shape)


# ==========================
# BASIC ANALYSIS
# ==========================

gene_means = df.mean(axis=0)

print("\nTop expressed genes:")
print(gene_means.sort_values(ascending=False).head())

print("\nLowest expressed genes:")
print(gene_means.sort_values().head())


# ==========================
# MOST ACTIVE GENE PER SAMPLE
# ==========================

if df.shape[0] > 1:

    most_active = df.idxmax(axis=1)

    result = pd.DataFrame({
        "Most_Active_Gene": most_active
    })

    print("\nMost active gene per sample:")
    print(result.head())


# ==========================
# BAR PLOT (GENE MEANS)
# ==========================

plt.figure(figsize=(10, 4))

gene_means.sort_values(ascending=False).head(10).plot(kind="bar")

plt.title("Top 10 Expressed Genes")
plt.ylabel("Expression Level")
plt.tight_layout()
plt.show()


# ==========================
# HEATMAP (REAL DATA SAFE)
# ==========================

plt.figure(figsize=(10, 6))

sns.heatmap(
    df.iloc[:, :20],   # limit for readability
    cmap="viridis"
)

plt.title("Gene Expression Heatmap (Subset)")
plt.tight_layout()
plt.show()