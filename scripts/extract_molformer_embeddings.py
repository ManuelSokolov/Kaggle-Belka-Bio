"Extract embeddings from pre-trained MoLFormer model."
import os
import polars as pl
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

STAGE = "test"
BATCH_SIZE = 64
DATA_PATH = "/home/ec2-user/SageMaker/dataset/"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

# Set up paths and configurations
molformer_path = os.path.join(DATA_PATH, "molformer", STAGE)
os.makedirs(molformer_path, exist_ok=True)
file_path = os.path.join(DATA_PATH, "test.parquet")

# Load data
df = pl.scan_parquet(file_path).collect()
df = df.sample(fraction=1, shuffle=True)  # shuffle to insert randomness in the batches
df.write_parquet(file_path)

n_batches = (len(df)) // BATCH_SIZE

# Load model
model = AutoModel.from_pretrained(
    MODEL_NAME, deterministic_eval=True, trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if torch.cuda.is_available():
    model = model.to("cuda")
    print("Transferred model to GPU")


def encode_protein(protein):
    "Encode protein in one-hot format."
    possible_proteins = ["BRD4", "sEH", "HSA"]
    protein_index = possible_proteins.index(protein)
    one_hot_vector = torch.eye(len(possible_proteins))[protein_index]
    return one_hot_vector


# Process batches: saving in batches to be more efficient
for idx in tqdm(range(n_batches)):
    start_idx = idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(df))
    batch_df = df[start_idx:end_idx]
    inputs = dict()
    smiles_list = batch_df["molecule_smiles"].to_list()
    batch_tokens = tokenizer(smiles_list, padding=True, return_tensors="pt")
    if torch.cuda.is_available():
        batch_tokens.to("cuda")
    outputs = model(**batch_tokens)
    batch_embeddings = outputs.last_hidden_state
    mean_embeddings = torch.mean(batch_embeddings, axis=1)
    inputs["embeddings"] = mean_embeddings
    protein_list = batch_df["protein_name"].to_list()
    protein_encoded = torch.stack([encode_protein(p) for p in protein_list])
    if STAGE == "test":
        label = torch.full((len(protein_list),), -1)
    else:
        label = torch.tensor(batch_df["binds"].to_numpy())
    inputs["label"] = label
    inputs["protein"] = protein_encoded

    inputs["label"] = label
    inputs["protein"] = protein_encoded

    torch.save(inputs, os.path.join(molformer_path, f"batch_{STAGE}_{idx}.pt"))
