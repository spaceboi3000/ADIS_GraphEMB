import pandas as pd
from io import StringIO

df1 = pd.read_csv('cluster_results_gin.csv')

# Read the second CSV (GIN data)
df2 = pd.read_csv('clustering_total.csv')

# Select only the relevant columns from df2 to match df1
df2_filtered = df2[['dataset', 'method', 'dim', 'n_graphs', 'n_clusters', 'cluster_alg', 'nmi', 'ari', 'acc', 'silhouette']]

# Concatenate the two dataframes
merged_df = pd.concat([df1, df2_filtered], ignore_index=True)

# Sort by dataset, method, dim, cluster_alg for better organization
merged_df = merged_df.sort_values(['dataset', 'method', 'dim', 'cluster_alg']).reset_index(drop=True)

# Save to new CSV
merged_df.to_csv('clustering_merged.csv', index=False)

print(f"Merged CSV saved as 'clustering_merged.csv'")
print(f"\nTotal rows: {len(merged_df)}")
print(f"\nMethods included: {merged_df['method'].unique()}")
print(f"\nDatasets included: {merged_df['dataset'].unique()}")
print(f"\nRows per method:")
print(merged_df['method'].value_counts())