"""Calculate circular fingerprints for each molecule in the dataset."""

import os

import polars as pl
import torch
from molfeat.trans.fp import FPVecTransformer
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool

def calculate_fingerprint(smiles):
    transformer = FPVecTransformer(kind="fcfp", dtype=float)
    fingerprint = transformer(smiles)
    return fingerprint

def process_batch(batch,batch_n, features_dir):

    batch = batch.with_columns(
        (
            pl.struct(["molecule_smiles"]).map_batches(
                lambda x: calculate_fingerprint(x.struct.field("molecule_smiles"))
            )
        ).alias("fingerprint"))

    data = {"id": batch["id"].to_list(), 
           "fingerprint":  batch["fingerprint"].to_numpy(),  
           "protein_names": batch["protein_name"].to_list()}
    torch.save(data,os.path.join(features_dir,  f"batch_{batch_n}.pt"))
    
def calculate_circular_fingerprint(df, features_dir, batch_size=512, num_processes=None):
    """Calculate circular fingerprint for each molecule in the dataframe
    and save it to the features_dir as a pytorch file."""
    try:
        os.mkdir(features_dir)
    except FileExistsError:
        print("Folder already exists.")

    batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
    for i, batch in tqdm(enumerate(batches), total= len(batches)): 
        process_batch(batch,i, features_dir)

if __name__ == "__main__":

    dataset_dir = "/home/ec2-user/SageMaker/dataset/"

    train = pl.scan_parquet(os.path.join(dataset_dir, "train.parquet"))
    train = train.select(["id", "molecule_smiles", "protein_name", "binds"]).collect()

    test = pl.scan_parquet(os.path.join(dataset_dir, "test.parquet"))
    test = test.select(["id", "molecule_smiles", "protein_name"]).collect()

    train_features_dir = os.path.join(dataset_dir, "circular_fingerprint", "train")
    test_features_dir = os.path.join(dataset_dir, "circular_fingerprint", "test")

    calculate_circular_fingerprint(train, train_features_dir)
    calculate_circular_fingerprint(test, train_features_dir)
