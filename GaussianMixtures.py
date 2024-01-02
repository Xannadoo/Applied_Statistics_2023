import pandas as pd
import numpy as np


## Visulaisation
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.sandbox.stats.multicomp import multipletests

# save data path and file name
data_path = "../data/raw/"
file_name = "data.csv"

# Load the dataset
df = pd.read_csv(data_path + file_name)
df.info()

# Save features
features = list(df.drop("Semer", axis=1).columns)
personality_traits = features[6:13]  # list of personality traits
drugs = features[13:]  # list of drug types


# Check if personality traits are normaly distributed
import statsmodels.api as sm
from scipy import stats


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


# Extract the selected features relevant for GMM
data = df[personality_traits]

# Pairplot for selected personality traits
sns.pairplot(data=data)
plt.suptitle("Pairplot of Selected Personality Traits", y=1.02)
plt.show()

# Visualize the distribution of each personality trait
plt.figure(figsize=(14, 8))
for i, trait in enumerate(personality_traits, 1):
    plt.subplot(3, 3, i)  # Adjusted to 2 rows, 2 columns
    sns.histplot(df[trait], kde=True)
    plt.title(f"Distribution of {trait}")

plt.tight_layout()
plt.show()

# GMM Analysis

# Import scaler
from sklearn.preprocessing import StandardScaler

# Scale numerical variables
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled

# Import model
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


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
print(f"Optimal number of clusters: {n_clusters}")

# Fit GMM model
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
cluster_labels = gmm.fit_predict(data_scaled)

# Add cluster labels to the original DataFrame
df["cluster"] = cluster_labels


# Visualize clusters using PCA or t-SNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_clusters(data, labels, method="PCA"):
    if method == "PCA":
        reduced_data = PCA(n_components=2).fit_transform(data)
    elif method == "t-SNE":
        reduced_data = TSNE(n_components=2, random_state=42).fit_transform(data)
    else:
        raise ValueError("Invalid visualization method. Use 'PCA' or 't-SNE'.")

    # Create a DataFrame for the reduced data
    reduced_df = pd.DataFrame(data=reduced_data, columns=["Component 1", "Component 2"])
    reduced_df["cluster"] = labels

    # Plot clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Component 1",
        y="Component 2",
        hue="cluster",
        palette="viridis",
        data=reduced_df,
    )
    plt.title(f"Clusters of Participants - {method}")
    plt.show()


# Visualize clusters using PCA
visualize_clusters(data_scaled, cluster_labels, method="PCA")
visualize_clusters(data_scaled, cluster_labels, method="t-SNE")


# Assess the relevance of identified clusters in explaining variations in drug use behaviors
# You can analyze drug use patterns within each cluster or perform further statistical tests
print("Cluster Characteristics:")
cluster_stats = df[personality_traits + drugs + ["cluster"]].groupby("cluster").mean()
print(cluster_stats)


# Assess drug use behaviors
for drug in drugs:
    drug_cluster_stats = (
        df.groupby("cluster")[drug].value_counts(normalize=True).unstack().fillna(0)
    )
    print(f"\nDrug Use Patterns for {drug}:")
    print(drug_cluster_stats)


# Statistical testing (if needed)
# Example: t-test for comparing means of a specific variable between clusters
from scipy.stats import ttest_ind

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=["object"]).columns
df[categorical_columns] = df[categorical_columns].apply(
    lambda col: label_encoder.fit_transform(col)
)


for drug in drugs:
    variable_of_interest = drug
    for i in range(n_clusters):
        cluster_data = df[df["cluster"] == i][variable_of_interest]
        other_clusters = df[df["cluster"] != i][variable_of_interest]

        stat, p_value = ttest_ind(cluster_data, other_clusters)
        print(
            f"\nT-test for {variable_of_interest} between Cluster {i} and Other Clusters:"
        )
        print(f"Statistic: {stat}, p-value: {p_value}")
        if p_value > 0.05:
            print("No Evidence for Statistical diff")
        else:
            print(f"Use of {drug} differs in the clusters")
            
        #Using Bonferroni correction 
        #Getting Bonferroni corrected P-value, which is 0.0001
        bf_p = p_value/len(drugs)


        count_below_threshold = len(p_value[p_value < bf_p])


        #Display number of False Positives 
        print(f"Number of false positives after Bonferroni Correction: {count_below_threshold}")


        #Benjamini and Hochberg method for controlling FDR at alpha=0.05
        y = multipletests(pvals = p_value, alpha=0.05, method="fdr_bh")


        #Display number of False Positives 
        count_BH = len(y[1][np.where(y[1]<0.05)])  # y[1] returns corrected P-vals (array)    
        print(f"Number of false positives after Benjamini Hochberg Correction: {count_BH}")
        
# Code reference for Bonferroni Correction and Benjamini Hochberg: 
#https://www.reneshbedre.com/blog/multiple-hypothesis-testing-corrections.html
