import dask.dataframe as dd
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from rdkit.Chem import Butina
import matplotlib.pyplot as plt

# Load data
train_data = dd.read_parquet('dataset/train.parquet', engine="pyarrow", columns=["molecule_smiles"])
test_data = dd.read_parquet('dataset/test.parquet', engine="pyarrow", columns=["molecule_smiles"])

print("dataloader")

# Combine the data for clustering
combined_data = dd.concat([train_data, test_data])

print("concatenated")

# Function to compute the fingerprint
def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()


# Parallel fingerprint calculation with progress bar
smiles_list = combined_data['molecule_smiles'].compute().tolist()
fps = Parallel(n_jobs=-1, backend='multiprocessing')(
    delayed(get_fingerprint)(smiles) for smiles in tqdm(smiles_list, desc="Calculating fingerprints")
)

print("done")

# Create a DataFrame with the SMILES and fingerprints
df_fps = pd.DataFrame({'molecule_smiles': smiles_list, 'fingerprint': fps})

# Save the DataFrame to a Parquet file
output_path = 'molecules_and_fingerprints.parquet'
df_fps.to_parquet(output_path, engine='pyarrow')

print(f"Fingerprints saved to {output_path}")

print("fps")

# Calculate Tanimoto similarity matrix
def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

size = len(fps)
similarity_matrix = np.zeros((size, size))
for i in range(size):
    for j in range(i + 1, size):
        similarity = tanimoto_similarity(fps[i], fps[j])
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

# Butina clustering
def butina_cluster(fps, cutoff=0.2):
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1-x for x in sims])
    clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return clusters

clusters = butina_cluster(fps)

# Assign cluster IDs
cluster_ids = np.zeros(len(fps))
for cluster_idx, cluster in enumerate(clusters):
    for member_idx in cluster:
        cluster_ids[member_idx] = cluster_idx

# Use PCA for visualization
pca = PCA(n_components=2)
pca_results = pca.fit_transform(1 - similarity_matrix)

# Separate PCA results for train and test
train_pca = pca_results[:len(train_sample)]
test_pca = pca_results[len(train_sample):]

# Plotting train and test set
plt.figure(figsize=(12, 8))
plt.scatter(train_pca[:, 0], train_pca[:, 1], c='blue', label='Train', alpha=0.5)
plt.scatter(test_pca[:, 0], test_pca[:, 1], c='red', label='Test', alpha=0.5)
plt.legend()
plt.title('PCA of Train and Test SMILES')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.savefig('smiles_pca_plot_train_test.png')
plt.show()

# Plotting clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=cluster_ids, cmap='tab20', alpha=0.6)
plt.legend(handles=scatter.legend_elements()[0], title="Clusters")
plt.title('PCA of SMILES Clusters')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.savefig('smiles_pca_plot_clusters.png')
plt.show()
