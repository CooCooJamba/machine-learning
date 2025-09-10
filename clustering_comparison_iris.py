#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets

# Visualization settings
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_and_preprocess_data():
    """
    Load and preprocess Iris dataset.
    Returns features, target labels, and DataFrame.
    """
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Create DataFrame for convenience
    df = pd.DataFrame(data, columns=feature_names)
    df['target'] = target
    df['species'] = df['target'].apply(lambda x: target_names[x])
    
    return data, target, df

def apply_scaler(data, scaler_type='minmax'):
    """
    Apply scaling to the data.
    
    Parameters:
    data: input data
    scaler_type: type of scaler ('minmax' or 'standard')
    
    Returns:
    scaled_data: scaled data
    scaler: scaler object
    """
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be 'minmax' or 'standard'")
    
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def find_optimal_k(data, max_k=8):
    """
    Find optimal number of clusters using silhouette score and elbow method.
    
    Parameters:
    data: scaled data
    max_k: maximum number of clusters to test
    
    Returns:
    optimal_k_silhouette: optimal k based on silhouette score
    silhouette_scores: list of silhouette scores
    inertia: list of inertia values
    """
    k_range = range(2, max_k + 1)
    silhouette_scores = []
    inertia = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        inertia.append(kmeans.inertia_)
    
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    
    return optimal_k_silhouette, silhouette_scores, inertia

def plot_clustering_results(data, true_labels, algorithms, algorithm_names, scaler_name):
    """
    Visualize clustering results.
    
    Parameters:
    data: scaled data
    true_labels: true labels
    algorithms: list of trained clustering models
    algorithm_names: names of algorithms
    scaler_name: name of the scaler used
    """
    n_algorithms = len(algorithms)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # True clusters
    scatter0 = axes[0].scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis', s=50)
    axes[0].set_title(f'True Classes (scaler: {scaler_name})')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    plt.colorbar(scatter0, ax=axes[0])
    
    # Clustering results
    for i, (algo, name) in enumerate(zip(algorithms, algorithm_names), 1):
        labels = algo.labels_
        scatter = axes[i].scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
        ari_score = adjusted_rand_score(true_labels, labels)
        axes[i].set_title(f'{name}\nARI: {ari_score:.3f}')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=axes[i])
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to perform clustering analysis.
    """
    # Load data
    data, target, df = load_and_preprocess_data()
    print("First 5 rows of data:")
    print(df.head())
    print(f"\nData shape: {data.shape}")
    
    scalers = ['minmax', 'standard']
    
    for scaler_type in scalers:
        print(f"\n{'='*50}")
        print(f"ANALYSIS USING {scaler_type.upper()} SCALER")
        print(f"{'='*50}")
        
        # Scale data
        scaled_data, scaler = apply_scaler(data, scaler_type)
        
        # Find optimal k
        optimal_k, silhouette_scores, inertia = find_optimal_k(scaled_data)
        print(f"Optimal number of clusters (silhouette): {optimal_k}")
        
        # Visualize k selection methods
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Silhouette plot
        ax1.plot(range(2, 9), silhouette_scores, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title(f'Silhouette Method ({scaler_type} scaler)')
        ax1.grid(True, alpha=0.3)
        
        # Elbow plot
        ax2.plot(range(2, 9), inertia, marker='o', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of clusters')
        ax2.set_ylabel('Sum of Squared Distances')
        ax2.set_title(f'Elbow Method ({scaler_type} scaler)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Apply clustering algorithms
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        
        hierarchical = AgglomerativeClustering(n_clusters=3)
        hierarchical.fit(scaled_data)
        
        # DBSCAN parameter tuning
        if scaler_type == 'minmax':
            dbscan_eps, dbscan_min_samples = 0.195, 5
        else:
            dbscan_eps, dbscan_min_samples = 0.75, 10
        
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        dbscan.fit(scaled_data)
        
        algorithms = [kmeans, hierarchical, dbscan]
        algorithm_names = ['K-Means', 'Hierarchical', 'DBSCAN']
        
        # Visualize results
        plot_clustering_results(scaled_data, target, algorithms, algorithm_names, scaler_type.upper())
        
        # Dendrogram for hierarchical clustering
        plt.figure(figsize=(12, 6))
        linked = linkage(scaled_data, 'ward')
        dendrogram(linked, orientation='top', 
                  distance_sort='descending', 
                  show_leaf_counts=True,
                  truncate_mode='level',
                  p=3)
        plt.title(f'Hierarchical Clustering Dendrogram ({scaler_type} scaler)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()
        
        # Print cluster labels
        print("\nCluster labels:")
        print(f"K-Means: {np.unique(kmeans.labels_, return_counts=True)}")
        print(f"Hierarchical: {np.unique(hierarchical.labels_, return_counts=True)}")
        print(f"DBSCAN: {np.unique(dbscan.labels_, return_counts=True)}")

if __name__ == "__main__":
    main()

