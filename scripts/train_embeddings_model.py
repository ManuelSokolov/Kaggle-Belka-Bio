"""
This script trains an embeddings model using PyTorch Lightning framework.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytorch_lightning as L
import torch
from pytorch_lightning.loggers import NeptuneLogger

from llm.datamodule import EmbeddingsDataModule
from llm.modelmodule import ModelModule, EmbeddingsModel


def main():
    "Train model."
    # Set the hyperparameters
    batch_size = 64
    learning_rate = 1e-5
    num_epochs = 100
    data_path = "/home/ec2-user/SageMaker/dataset/"    
    loss_function = "bce"
    hidden_dim1 = 512
    hidden_dim2 = 256
    dropout = 0.5

    optimizer = "Adam"

    # Create the data module
    datamodule = EmbeddingsDataModule(
        os.path.join(data_path, "train_subsampled_50_50.parquet"),
        os.path.join(data_path, "val_subsampled_50_50.parquet"),
        os.path.join(data_path, "test.parquet"),
        dataset_dir=os.path.join(data_path, "molformer"),
        batch_size=batch_size,
    )

    # Create the model module
    model = EmbeddingsModel(
        input_dim=768
        + 3,  # 768 for the molformer embeddings and 3 for the protein encoded features
        loss="bce",
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout=dropout,
    )
    modelmodule = ModelModule(
        model, learning_rate, loss=loss_function, optimizer=optimizer
    )

    # Set the floating-point precision
    torch.set_float32_matmul_precision("medium")

    # Create the Neptune logger for logging the training progress
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODY3NzNiZi1jYTM1LTRkM2MtYmE1Ny1hODRjN2UzNjBmMTQifQ==",
        project="cristiana.carpinteiro/kaggle-challenge",
        tags=["embeddings", "bigger_dims"],
    )

    # Set the multiprocessing start method
    torch.multiprocessing.set_start_method('spawn')

    # Set the training parameters
    params = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "loss_function": loss_function,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
    }

    # Log the training parameters to Neptune
    neptune_logger.experiment["params"] = params

    # Create the checkpoint callback for saving the model
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        mode="min",
        monitor="val_loss",
    )

    # Create the early stopping callback for preventing overfitting
    early_stopping_callback = L.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )

    # Create the PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=neptune_logger,
        log_every_n_steps=500,
        profiler="simple",
    )

    # Train the model
    trainer.fit(modelmodule, datamodule)


if __name__ == "__main__":
    main()
