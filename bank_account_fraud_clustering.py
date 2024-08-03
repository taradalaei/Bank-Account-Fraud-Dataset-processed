import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
def visualize_important_feature_pairs(df, features, cluster_labels):
    num_features = len(features)
    num_plots = num_features * (num_features - 1) // 2  # Number of pairwise plots
    num_rows = (num_plots // 2) + (num_plots % 2)
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 4 * num_rows))
    axes = axes.flatten()

    idx = 0
    for i in range(num_features):
        for j in range(i + 1, num_features):
            sns.scatterplot(data=df, x=features[i], y=features[j], hue=cluster_labels, ax=axes[idx])
            axes[idx].set_title(f'{features[i]} vs {features[j]}')
            idx += 1

    for ax in axes[idx:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()



df = pd.read_csv("D:/uni_st/term 6/AI/hws/hw5/cleaned_preprocessed_dataset.csv")


##Clustering##

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X_scaled)

# Check the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

# Number of components
n_components = pca.n_components_
print("Number of Components:", n_components)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_pca)

# Get the cluster labels
cluster_labels = kmeans.labels_

# Add cluster labels to the original DataFrame
df['cluster'] = cluster_labels

# Analyze the clusters
cluster_analysis = df.groupby('cluster').mean()

print(cluster_analysis)

# Select important features for visualization
important_features = ['income', 'name_email_similarity', 'prev_address_months_count', 'current_address_months_count', 'customer_age']
visualize_important_feature_pairs(df, important_features, kmeans.labels_)

##########################################################################

#split the labels from our dataset
target = 'fraud_bool'
X = df.drop(target,axis=1)
y = df[target]
print(X.head())

# Train/test split with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
train = pd.concat([X_train, y_train], axis=1).copy()
train_copy = pd.concat([X_train, y_train], axis=1).copy()


pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_train_pca)

# Predict the clusters for train and test data
train_clusters = kmeans.predict(X_train_pca)
test_clusters = kmeans.predict(X_test_pca)

# Add the cluster labels to the dataset for visualization
X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(n_components)])
X_train_pca_df['Cluster'] = train_clusters
X_train_pca_df['fraud_bool'] = y_train.values

X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(n_components)])
X_test_pca_df['Cluster'] = test_clusters
X_test_pca_df['fraud_bool'] = y_test.values

# Plot the clusters for training data
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X_train_pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', alpha=0.6)
plt.title('PCA Clusters (Training Data)')
plt.show()

# Plot the clusters for test data
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X_test_pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', alpha=0.6)
plt.title('PCA Clusters (Test Data)')
plt.show()