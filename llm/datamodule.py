""" Data Module for Fine-Tuning Molformer."""

import os
import numpy as np
import polars as pl
import pytorch_lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time 

class EmbeddingsDataset(Dataset):
    """Dataset for fine-tuning Molformer on ligand binding dataset."""

    def __init__(
        self,
        dataset_file: str,
        batch_size: int = 64,  # batch size with which the dataset was extracted (extract molformer embeddings script)
        stage: str = "train",
        dataset_dir: str = "/home/ec2-user/SageMaker/dataset/molformer",
    ):
        self.dataset_dir = os.path.join(dataset_dir, stage)
        self.stage = stage
        self.dataset = pl.scan_parquet(dataset_file).collect()
        self.batch_size = batch_size

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.stage == "test": 
            length = len(self.dataset) // self.batch_size + 1 
        else: 
            length = len(self.dataset) // self.batch_size
        return length

    def __getitem__(self, idx: int):
        filename = os.path.join(self.dataset_dir, f"batch_{self.stage}_{idx}.pt")
        inputs = torch.load(filename)
        return inputs



class EmbeddingsDataModule(L.LightningDataModule):
    "Data module for ligand binding dataset."

    def __init__(
        self,
        train_df_file,
        val_df_file,
        test_df_file,
        batch_size=64,
        dataset_dir="/home/ec2-user/SageMaker/dataset/molformer",
    ):
        super().__init__()
        self.train_df_file = train_df_file
        self.val_df_file = val_df_file
        self.test_df_file = test_df_file
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir

    def _generate_dataset(self, stage):
        if stage == "train":
            df = self.train_df_file

        elif stage == "val":
            df = self.val_df_file

        else:
            df = self.test_df_file

        dataset = EmbeddingsDataset(df, self.batch_size, stage, self.dataset_dir)

        return dataset

    def _generate_dataloader(self, stage):
        dataset = self._generate_dataset(stage)
        if stage == "test":
            n_batches = 1
        else:
            n_batches = 16
        return DataLoader(
            dataset,
            batch_size=n_batches,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            collate_fn=collate_embeddings,
            num_workers=os.cpu_count(),
            persistent_workers=True,
        )

    def train_dataloader(self):
        "Get train dataloader"
        return self._generate_dataloader("train")

    def val_dataloader(self):
        "Get validation dataloader"
        return self._generate_dataloader("val")

    def test_dataloader(self):
        "Get test dataloader"
        return self._generate_dataloader("test")

class BatchFineTuneDataset(Dataset):
    """Dataset for fine-tuning Molformer on ligand binding dataset."""

    def __init__(
        self,
        dataset_file: str,
        tokenizer,
        batch_size: int = 64,
        stage: str = "train",
        pad_max_length: int = 150,
        truncation: bool = True,
        save_dir: str = "/home/ec2-user/SageMaker/dataset/",
        preprocess_dataset: bool = True,
        shuffle=True,
    ):
        self.stage = stage
        if stage == "test":
            columns = ["id", "molecule_smiles", "protein_name"]
        else:
            columns = ["id", "molecule_smiles", "binds", "protein_name"]
        df = pl.scan_parquet(dataset_file)
        self.dataset = (
            df.select(columns).collect().sample(fraction=1, shuffle=shuffle)
        )  # add randomness to batches
        self.tokenizer = tokenizer
        self.pad_max_length = pad_max_length
        self.truncation = truncation
        self.possible_proteins = ["BRD4", "sEH", "HSA"]
        self.save_dir = save_dir
        self.batch_size = batch_size

        if preprocess_dataset:
            self.__process_dataset__()

    def __process_dataset__(self):
        for idx in tqdm(range(self.__len__())):
            batch = self.__process_batch__(idx)
            filename = os.path.join(self.save_dir, f"batch_{self.stage}_{idx}.pt")
            torch.save(batch, filename)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset) // self.batch_size

    def __process_batch__(self, idx: int):
        """Return a batch of samples at the given index."""
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        batch_df = self.dataset[start_idx:end_idx]

        smiles_list = batch_df["molecule_smiles"].to_list()

        inputs = self.tokenizer(
            smiles_list,
            padding="max_length",
            max_length=self.pad_max_length,
            truncation=self.truncation,
            return_tensors="pt",
        )

        protein_list = batch_df["protein_name"].to_list()
        protein_encoded = torch.stack(
            [self.__encode_protein__(p) for p in protein_list]
        )

        if self.stage == "test":
            label = torch.full((self.batch_size,), -1, dtype=torch.long)
        else:
            label = torch.tensor(batch_df["binds"].to_numpy())

        inputs["label"] = label
        inputs["protein"] = protein_encoded

        return inputs

    def __getitem__(self, idx: int):
        filename = os.path.join(self.save_dir, f"batch_{self.stage}_{idx}.pt")
        if os.path.exists(filename):
            inputs = torch.load(filename)
        else:
            inputs = self.__process_batch__(idx)
        return inputs

    def __encode_protein__(self, protein):
        if protein not in self.possible_proteins:
            raise ValueError(
                f"Invalid protein: {protein}. Possible proteins are {self.possibsle_proteins}"
            )
        protein_index = self.possible_proteins.index(protein)
        one_hot_vector = torch.eye(len(self.possible_proteins))[protein_index]

        return one_hot_vector


class FineTuneDataset(Dataset):
    """Dataset for fine-tuning Molformer on ligand binding dataset."""

    def __init__(
        self,
        dataset_file: str,
        tokenizer,
        batch_size: int = 64,
        stage: str = "train",
        pad_max_length: int = 150,
        truncation: bool = True,
        save_dir: str = "/home/ec2-user/SageMaker/dataset/",
    ):
        self.stage = stage
        if stage == "test":
            columns = ["id", "molecule_smiles", "protein_name"]
        else:
            columns = ["id", "molecule_smiles", "binds", "protein_name"]
        df = pl.scan_parquet(dataset_file)
        self.dataset = df.select(columns).collect()
        self.tokenizer = tokenizer
        self.pad_max_length = pad_max_length
        self.truncation = truncation
        self.possible_proteins = ["BRD4", "sEH", "HSA"]
        self.save_dir = save_dir

    def __process_dataset__(self):
        for idx in tqdm(range(self.__len__())):
            batch = self.__process_batch__(idx)
            filename = os.path.join(self.save_dir, f"batch_{self.stage}_{idx}.pt")
            torch.save(batch, filename)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """Return the sample at the given index."""
        smiles = self.dataset[idx]["molecule_smiles"].to_list()
        inputs = self.tokenizer(
            smiles,
            padding="max_length",
            max_length=self.pad_max_length,
            truncation=self.truncation,
            return_tensors="pt",
        )
        protein = self.dataset[idx]["protein_name"].to_list()[0]
        protein_encoded = self.__encode_protein__(protein)

        if self.stage == "test":
            label = np.array(-1)
        else:
            label = self.dataset[idx]["binds"].to_numpy()
        inputs["label"] = label
        inputs["protein"] = protein_encoded
        return inputs

    def __encode_protein__(self, protein):
        if protein not in self.possible_proteins:
            raise ValueError(
                f"Invalid protein: {protein}. Possible proteins are {self.possible_proteins}"
            )
        protein_index = self.possible_proteins.index(protein)
        one_hot_vector = torch.eye(len(self.possible_proteins))[protein_index]

        return one_hot_vector


class LBDataModule(L.LightningDataModule):
    "Data module for ligand binding dataset."

    def __init__(
        self,
        train_df_file,
        val_df_file,
        test_df_file,
        tokenizer,
        batch_size=16,
    ):
        super().__init__()
        self.train_df_file = train_df_file
        self.val_df_file = val_df_file
        self.test_df_file = test_df_file
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def _generate_dataset(self, stage):
        if stage == "train":
            df = self.train_df_file
            dataset = BatchFineTuneDataset(
                df, self.tokenizer, stage=stage, preprocess_dataset=False, shuffle=True
            )

        elif stage == "val":
            df = self.val_df_file
            dataset = BatchFineTuneDataset(
                df, self.tokenizer, stage=stage, preprocess_dataset=False, shuffle=True
            )

        elif stage == "test":
            df = self.test_df_file
            dataset = FineTuneDataset(
                df, self.tokenizer, stage=stage
            )  # For test it needs to be the sample loader, otherwise some samples might be discarded

        else:
            raise NotImplementedError

        return dataset

    def _generate_dataloader(self, stage):
        dataset = self._generate_dataset(stage)
        if stage == "train":
            shuffle = True
            drop_last = True
            collate = collate_batches

        elif stage == "val":
            shuffle = False
            drop_last = False
            collate = collate_batches

        elif stage == "test":
            shuffle = False
            drop_last = False
            collate = collate_samples

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
            collate_fn=collate,
            num_workers=os.cpu_count(),
            persistent_workers=True,
        )

    def train_dataloader(self):
        "Get train dataloader"
        return self._generate_dataloader("train")

    def val_dataloader(self):
        "Get validation dataloader"
        return self._generate_dataloader("val")

    def test_dataloader(self):
        "Get test dataloader"
        return self._generate_dataloader("test")

    
def collate_samples(batch):
    "Format samples for LLM training."
    # process inputs and attention masks
    input_ids_batch = [sample["input_ids"] for sample in batch]
    attention_mask_batch = [sample["attention_mask"] for sample in batch]
    padded_input_ids = pad_sequence(input_ids_batch, batch_first=True, padding_value=0)
    padded_attention_mask = pad_sequence(
        attention_mask_batch, batch_first=True, padding_value=0
    )
    padded_input_ids = padded_input_ids.squeeze(1)
    padded_attention_mask = padded_attention_mask.squeeze(1)
    # process labels
    labels_batch = [torch.tensor(sample["label"]) for sample in batch]
    labels = torch.stack(labels_batch)
    protein_batch = [sample["protein"] for sample in batch]
    proteins = torch.stack(protein_batch)
    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": labels,
        "protein": proteins,
    }


def collate_batches(batch):
    "Format batches for LLM training."
    input_ids_batch = [sample["input_ids"] for sample in batch]
    attention_mask_batch = [sample["attention_mask"] for sample in batch]
    labels_batch = [sample["label"] for sample in batch]
    protein_batch = [sample["protein"] for sample in batch]

    inputs_stack = torch.stack(input_ids_batch)
    new_shape = (inputs_stack.shape[0] * inputs_stack.shape[1], inputs_stack.shape[2])
    inputs = inputs_stack.reshape(new_shape)

    attention_stack = torch.stack(attention_mask_batch)
    new_shape = (
        attention_stack.shape[0] * attention_stack.shape[1],
        attention_stack.shape[2],
    )
    attention = attention_stack.reshape(new_shape)

    proteins_stack = torch.stack(protein_batch)
    new_shape = (
        proteins_stack.shape[0] * proteins_stack.shape[1],
        proteins_stack.shape[2],
    )
    proteins = proteins_stack.reshape(new_shape)
    labels = torch.cat(labels_batch).unsqueeze(1)
    return {
        "input_ids": inputs,
        "attention_mask": attention,
        "labels": labels,
        "protein": proteins,
    }

def collate_embeddings(batch):
    "Format batches for LLM training."
    embeddings_batch = [sample["embeddings"] for sample in batch]
    labels_batch = [sample["label"] for sample in batch]
    protein_batch = [sample["protein"] for sample in batch]

    embeddings_stack = torch.stack(embeddings_batch)
    new_shape = (
        embeddings_stack.shape[0] * embeddings_stack.shape[1],
        embeddings_stack.shape[2],
    )
    embeddings = embeddings_stack.reshape(new_shape)

    proteins_stack = torch.stack(protein_batch)
    new_shape = (
        proteins_stack.shape[0] * proteins_stack.shape[1],
        proteins_stack.shape[2],
    )
    proteins = proteins_stack.reshape(new_shape)
    labels = torch.cat(labels_batch).unsqueeze(1)
    end_time = time.time()

    return {
        "embeddings": embeddings.detach(),
        "labels": labels,
        "protein": proteins,
    }


