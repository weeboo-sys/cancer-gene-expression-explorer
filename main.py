import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================
# LOAD DATA (ROBUST PATH)
# ==========================

file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "sample_data.csv"
)

df = pd.read_csv(file_path)


# ==========================
# BASIC OVERVIEW
# ==========================

print("\nDATA PREVIEW")
print(df.head())

print("\nSHAPE (rows, columns)")
print(df.shape)

print("\nCOLUMN INFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nSUMMARY STATISTICS")
print(df.describe())


# ==========================
# GENE ANALYSIS
# ==========================

gene_means = df.drop(columns=["Sample"]).mean()

print("\nAVERAGE GENE EXPRESSION")
print(gene_means)


def most_active_gene(row):
    genes = row.drop("Sample")
    return genes.idxmax()


df["Most_Active_Gene"] = df.apply(most_active_gene, axis=1)

print("\nMOST ACTIVE GENE PER SAMPLE")
print(df[["Sample", "Most_Active_Gene"]])


print("\nGENE INSIGHTS")
print("Highest average expression gene:", gene_means.idxmax())
print("Lowest average expression gene:", gene_means.idxmin())


# ==========================
# BAR PLOT
# ==========================

df_plot = df.set_index("Sample")

ax = df_plot.drop(
    columns=["Most_Active_Gene"],
    errors="ignore"
).plot(kind="bar")

ax.set_title("Gene Expression Across Samples")
ax.set_ylabel("Expression Level")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ==========================
# HEATMAP 
# ==========================

df_heat = df.set_index("Sample")

# keep only numeric columns (IMPORTANT FIX)
df_heat = df_heat.select_dtypes(include="number")

plt.figure(figsize=(8, 5))

sns.heatmap(
    df_heat,
    annot=True,
    cmap="viridis"
)

plt.title("Cancer Gene Expression Heatmap")
plt.tight_layout()
plt.show()