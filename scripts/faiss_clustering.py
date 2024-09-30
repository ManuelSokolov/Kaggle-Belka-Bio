import boto3
import torch
import io
import faiss
import os
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt

'''
Script to test faiss for clustering dataset

'''
# Initialize S3 client
s3 = boto3.client('s3')

def load_batch_to_tensor(bucket_name, prefix, batch_index, split):
    file_name = f"batch_{split}_{batch_index}.pt"
    key = f"{prefix}/{split}/{file_name}" 
    # Create an in-memory bytes buffer
    buffer = io.BytesIO()    
    # Download the file content to the buffer
    s3.download_fileobj(bucket_name, key, buffer)    
    # Set the buffer's position to the beginning
    buffer.seek(0)    
    # Load the tensor from the buffer
    tensor = torch.load(buffer)
    tensor = tensor["embeddings"]
    
    return tensor

def list_batches_in_s3(bucket_name, prefix, split):
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=f"{prefix}/{split}/")
    
    batch_files = []
    for page in page_iterator:
        batch_files.extend([obj['Key'] for obj in page.get('Contents', []) if obj['Key'].endswith('.pt')])
    
    return batch_files

def load_and_append(batch_file, bucket_name, prefix, split):
    # Extract batch index from the file name
    batch_index = int(os.path.basename(batch_file).split('_')[-1].split('.')[0])
    batch_tensor = load_batch_to_tensor(bucket_name, prefix, batch_index, split)
    return batch_tensor

# Function to handle parallel loading of batches
def parallel_load_batches(batch_files, split):
    embeddings = []
    with ThreadPoolExecutor() as executor:
        future_to_batch = {executor.submit(load_and_append, batch_file, bucket_name, prefix, split): batch_file for batch_file in batch_files}
        for future in tqdm(as_completed(future_to_batch), total=len(batch_files), desc=f"Loading batches {split}"):
            batch_tensor = future.result()
            embeddings.append(batch_tensor)
    return embeddings


# Example usage
bucket_name = 'leash-belka'
prefix = 'molformer_embeddings'
splits = ["train", "val", "test"]

# List all batch files in the train, val, and test directories
batch_files_train = list_batches_in_s3(bucket_name, prefix, "train")
batch_files_val = list_batches_in_s3(bucket_name, prefix, "val")
batch_files_test = list_batches_in_s3(bucket_name, prefix, "test")

# Load and concatenate all batches
embeddings_train = []
embeddings_test = []


# Load and concatenate all batches in parallel
embeddings_train = parallel_load_batches(batch_files_train, "train")
embeddings_train += parallel_load_batches(batch_files_val, "val")
embeddings_test = parallel_load_batches(batch_files_test, "test")

size = int(len(embeddings_train)/2)

embeddings_train = embeddings_train[:size]
embeddings_train += embeddings_test

# Concatenate all embeddings into a single tensor
all_embeddings_tensor = torch.cat(embeddings_train, dim=0)


# Convert the concatenated tensor to a NumPy array
all_embeddings_np = all_embeddings_tensor.detach().cpu().numpy()

# Initialize the GPU resources (already in GPU)
#res = faiss.StandardGpuResources()

print(all_embeddings_np.shape[0])
# Dimension of the embeddings
d = all_embeddings_np.shape[1]

# Define a range of cluster numbers to test
cluster_numbers = [10, 100, 1000, 10000, 100000, 1000000]

# Store metrics for each cluster count
results = []

for num_clusters in cluster_numbers:
    print(f"\nTesting with {num_clusters} clusters...")
    
    # Create a k-means clustering index on the GPU
    kmeans = faiss.Kmeans(d=d, k=num_clusters, niter=20, verbose=True, gpu=True)
    
    # Track the time taken for k-means clustering
    print("Training k-means clustering...")
    start_time = time.time()
    kmeans.train(all_embeddings_np)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"K-means training completed in {training_time:.2f} seconds.")
    
    # Get cluster assignments for each embedding
    print("Assigning clusters...")
    start_time = time.time()
    _, cluster_assignments = kmeans.index.search(all_embeddings_np, 1)
    end_time = time.time()
    assignment_time = end_time - start_time
    cluster_assignments = cluster_assignments.flatten()
    print(f"Cluster assignments completed in {assignment_time:.2f} seconds.")
    
    # Calculate clustering metrics
    print("Calculating clustering metrics...")
    
    # Inertia (Within-Cluster Sum of Squares)
    inertia = kmeans.obj[-1]
    
    # Silhouette Score
    if num_clusters < len(all_embeddings_np):
        silhouette_avg = silhouette_score(all_embeddings_np, cluster_assignments)
    else:
        silhouette_avg = None
    
    # Calinski-Harabasz Index
    calinski_harabasz_avg = calinski_harabasz_score(all_embeddings_np, cluster_assignments)
    
    # Davies-Bouldin Index
    davies_bouldin_avg = davies_bouldin_score(all_embeddings_np, cluster_assignments)
    
    # Store the results
    results.append({
        "num_clusters": num_clusters,
        "inertia": inertia,
        "silhouette_score": silhouette_avg,
        "calinski_harabasz_index": calinski_harabasz_avg,
        "davies_bouldin_index": davies_bouldin_avg,
        "training_time": training_time,
        "assignment_time": assignment_time
    })
    
    # Print the results for the current number of clusters
    print(f"Inertia: {inertia}")
    if silhouette_avg is not None:
        print(f"Silhouette Score: {silhouette_avg}")
    else:
        print("Silhouette Score: Not applicable (number of clusters >= number of samples)")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin_avg}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Assignment Time: {assignment_time:.2f} seconds")

# Plotting the results
cluster_counts = [result["num_clusters"] for result in results]
inertia_values = [result["inertia"] for result in results]
silhouette_scores = [result["silhouette_score"] for result in results if result["silhouette_score"] is not None]
calinski_harabasz_values = [result["calinski_harabasz_index"] for result in results]
davies_bouldin_values = [result["davies_bouldin_index"] for result in results]
training_times = [result["training_time"] for result in results]
assignment_times = [result["assignment_time"] for result in results]

fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Plot Inertia
axs[0, 0].plot(cluster_counts, inertia_values, marker='o')
axs[0, 0].set_title('Inertia (Within-Cluster Sum of Squares)')
axs[0, 0].set_xlabel('Number of Clusters')
axs[0, 0].set_ylabel('Inertia')
axs[0, 0].set_xscale('log')

# Plot Silhouette Score
axs[0, 1].plot(cluster_counts[:len(silhouette_scores)], silhouette_scores, marker='o')
axs[0, 1].set_title('Silhouette Score')
axs[0, 1].set_xlabel('Number of Clusters')
axs[0, 1].set_ylabel('Silhouette Score')
axs[0, 1].set_xscale('log')

# Plot Calinski-Harabasz Index
axs[1, 0].plot(cluster_counts, calinski_harabasz_values, marker='o')
axs[1, 0].set_title('Calinski-Harabasz Index')
axs[1, 0].set_xlabel('Number of Clusters')
axs[1, 0].set_ylabel('Calinski-Harabasz Index')
axs[1, 0].set_xscale('log')

# Plot Davies-Bouldin Index
axs[1, 1].plot(cluster_counts, davies_bouldin_values, marker='o')
axs[1, 1].set_title('Davies-Bouldin Index')
axs[1, 1].set_xlabel('Number of Clusters')
axs[1, 1].set_ylabel('Davies-Bouldin Index')
axs[1, 1].set_xscale('log')

# Plot Training Time
axs[2, 0].plot(cluster_counts, training_times, marker='o')
axs[2, 0].set_title('Training Time')
axs[2, 0].set_xlabel('Number of Clusters')
axs[2, 0].set_ylabel('Time (seconds)')
axs[2, 0].set_xscale('log')

# Plot Assignment Time
axs[2, 1].plot(cluster_counts, assignment_times, marker='o')
axs[2, 1].set_title('Assignment Time')
axs[2, 1].set_xlabel('Number of Clusters')
axs[2, 1].set_ylabel('Time (seconds)')
axs[2, 1].set_xscale('log')

plt.tight_layout()
plt.show()
