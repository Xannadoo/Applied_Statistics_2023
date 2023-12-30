import pandas as pd
import numpy as np


## Visulaisation
import matplotlib.pyplot as plt
import seaborn as sns


## analysis
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
import statsmodels.api as sm

from scipy.stats import anderson


# Load the dataset
df = pd.read_csv("drugsData.csv")

# Save features
features = list(df.columns)
personality_traits = features[6:13]  # list of personality traits
drugs = features[13:]  # list of drug types

# Extract the selected features relevant for GMM
data = df[personality_traits]


# Check if personality traits are normaly distributed
def checkNormal(df, variables):
    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Set up subplots
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 7))
    fig.subplots_adjust(hspace=0.5)

    # Flatten the 2D array of subplots for easier iteration
    axes = axes.flatten()

    # Common title for all Q-Q plots
    fig.suptitle("Q-Q Plots for Personality Traits", fontsize=16)

    # Loop over variables and create Q-Q plots
    for i, variable in enumerate(variables):
        ax = axes[i]

        # Shapiro-Wilk test for normality
        stat, p_value = stats.shapiro(df[variable])
        print(f"{variable}, Shapiro-Wilk Test - Statistic: {stat}, p-value: {p_value}")

        # Q-Q plot using statsmodels
        sm.qqplot(df[variable], line="q", ax=ax, color="skyblue")

        # Annotate p-value on the plot
        ax.annotate(
            f"Shapiro-Wilk Test, p-value: {p_value:.4f}",
            xy=(0.5, 0.02),
            xycoords="axes fraction",
            ha="center",
            fontsize=8,
            color="darkred",
        )

        # Customize axis labels
        ax.set_title(f"{variable}", fontsize=12)
        ax.set_xlabel("Theoretical Quantiles", fontsize=10)
        ax.set_ylabel("Ordered Values", fontsize=10)

    # Remove the empty subplot
    for i in range(len(variables), len(axes)):
        fig.delaxes(axes[i])

    # Show the plots
    plt.tight_layout()
    plt.show()


checkNormal(df, variables=personality_traits)


# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


# Determine the optimal number of clusters using silhouette score
def get_optimal_clusters(data):
    scores = []
    for n_clusters in range(2, 11):
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        scores.append(silhouette_avg)
    optimal_clusters = np.argmax(scores) + 2  # Add 2 because we started from 2 clusters
    return optimal_clusters


# Get the optimal number of clusters
n_clusters = get_optimal_clusters(data_scaled)


# Fit GMM model
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
clusters = gmm.fit_predict(data_scaled)

# Add cluster labels to the original DataFrame
df["cluster"] = clusters

# Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)


# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap="viridis", s=50)
plt.title("Gaussian Mixture Analysis - Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# Display cluster means and covariances
for i in range(n_clusters):
    print(f"Cluster {i + 1} Mean:")
    print(scaler.inverse_transform(gmm.means_[i].reshape(1, -1)))
    print(f"Cluster {i + 1} Covariance:")
    print(np.sqrt(np.diag(scaler.inverse_transform(gmm.covariances_[i]))))
    print("=" * 40)


# Selects relevant features for GMM (neuroticism, extraversion, openness, agreeableness).
# Standardizes the data.
# Determines the optimal number of clusters using silhouette score.
# Fits a GMM model with the optimal number of clusters.
# Adds cluster labels to the original DataFrame.
# Reduces dimensionality using PCA for visualization.
# Plots the clusters in a 2D space.
# Prints cluster means and covariances.
# %%
data["impulsive"].unique()
# %%
