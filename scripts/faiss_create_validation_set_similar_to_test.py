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
Idea behind this approach

For each embedding in test, get one similar from train to create validation set
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
#batch_files_val = list_batches_in_s3(bucket_name, prefix, "val")
batch_files_test = list_batches_in_s3(bucket_name, prefix, "test")


# Load and concatenate all batches
embeddings_train = []
embeddings_test = []

# Load and concatenate all batches in parallel
embeddings_train = parallel_load_batches(batch_files_train, "train")
#embeddings_train += parallel_load_batches(batch_files_val, "val")
embeddings_test = parallel_load_batches(batch_files_test, "test")

# Currently testing for 50% of train set
size = int(len(embeddings_train)/3)

embeddings_train = embeddings_train[:size]

# Concatenate all embeddings into a single tensor
embeddings_train = torch.cat(embeddings_train, dim=0)
embeddings_test = torch.cat(embeddings_test, dim=0)

# vary this number (cells are the clusters)
n_cells = 100000
dimensions = embeddings_train.shape[1]
quantitizer = faiss.IndexFlatL2(dimensions)
index = faiss.IndexIVFFlat(quantitizer, dimensions, n_cells)
index.train(embeddings_train)
index.add(embeddings_train)

# Initialize the result DataFrame with the specified columns
# Here we create a dataframe with for each test sample the most similar sample of train
result = pd.DataFrame(columns=["test_index", "train_most_similar_mol_index"])

for idx, emb in enumerate(embeddings_test):
    D, I = index.search(emb, 1)
    train_index = I[0][0]  # Extracting the train index from the result
    test_index = idx  # Using the loop index as the test index 
    # Append the new row to the result DataFrame
    result = result.append({"test_index": test_index, "train_most_similar_mol_index": train_index}, ignore_index=True)

# Save the resulting DataFrame to a Parquet file
file_path = 'mapping_test_train.parquet'
result.to_parquet(file_path, index=False)

# Display a message to the user with the file path
print(f"The DataFrame has been saved to {file_path}.")



