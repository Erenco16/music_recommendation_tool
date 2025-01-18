import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from db_operations import readData

# this is a test commit
def find_optimal_k(scaled_features, max_k=10):
    """
    Determine the optimal number of clusters (k) using the Elbow Method and Silhouette Score.

    Parameters:
    - scaled_features: Array-like, shape (n_samples, n_features)
        The scaled feature set for clustering.
    - max_k: int
        The maximum number of clusters to test.

    Returns:
    - optimal_k: int
        The optimal number of clusters based on the highest silhouette score.
    """

    # Lists to hold WCSS and silhouette scores
    wcss = []
    silhouette_scores = []

    # Calculate WCSS and silhouette scores for each k
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

        if k > 1:  # Silhouette score is undefined for k=1
            score = silhouette_score(scaled_features, kmeans.labels_)
            silhouette_scores.append(score)

    # Plotting the Elbow Method graph
    plt.figure(figsize=(12, 5))

    # Elbow Method plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS (Within-cluster sum of squares)')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)

    # Silhouette Score plot
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, max_k + 1))
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Determine optimal k based on maximum silhouette score
    if silhouette_scores:
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 due to index offset
    else:
        optimal_k = 1  # If no silhouette scores are calculated

    return optimal_k


def df_operations(data):
    # Create a DataFrame to hold user features
    user_features = pd.DataFrame()

    # Calculate track count
    track_count = data.groupby('user_id')['track'].nunique().reset_index()
    track_count.columns = ['user_id', 'track_count']

    # Calculate playlist count
    playlist_count = data.groupby('user_id')['playlist'].nunique().reset_index()
    playlist_count.columns = ['user_id', 'playlist_count']

    # Calculate artist diversity
    artist_diversity = data.groupby('user_id')['artist'].nunique().reset_index()
    artist_diversity.columns = ['user_id', 'artist_diversity']

    # Merge all features into a single DataFrame
    user_features = (track_count
                     .merge(playlist_count, on='user_id')
                     .merge(artist_diversity, on='user_id'))

    # Calculate user_activity_metric - this is a separate df
    user_activity_metric = pd.DataFrame({
        'user_id': user_features['user_id'],
        'user_activity_metric': (
                user_features['track_count'] +
                user_features['playlist_count'] +
                user_features['artist_diversity']
        )
    })

    # Merge the user_activity_metric DataFrame with user_features
    user_features = user_features.merge(user_activity_metric, on='user_id')

    return user_features


def main():
    # read_csv function randomly picks 50% of the data that's in the csv file
    data = df_operations(readData.read_csv('../data/spotify_million_playlists_dataset/spotify_dataset.csv'))
    # Example feature engineering
    features = data[['track_count', 'playlist_count', 'artist_diversity', 'user_activity_metric']]  # Adjust as needed

    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply K-means clustering
    optimal_k = find_optimal_k(scaled_features)
    kmeans = KMeans(n_clusters=optimal_k)
    data['cluster'] = kmeans.fit_predict(scaled_features)

    # 1. Visualize with PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)

    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    pca_df['cluster'] = data['cluster']

    # Plot PCA results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='cluster', palette='viridis', alpha=0.7)
    plt.title('PCA of Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.show()

    # 2. Visualize with t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(scaled_features)

    # Create a DataFrame for t-SNE results
    tsne_df = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])
    tsne_df['cluster'] = data['cluster']

    # Plot t-SNE results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='cluster', palette='viridis', alpha=0.7)
    plt.title('t-SNE of Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Cluster')
    plt.show()


if __name__ == '__main__':
    main()