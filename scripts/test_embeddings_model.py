"Test script for the embeddings model."
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import polars as pl
import pytorch_lightning as L
import torch

from llm.datamodule import EmbeddingsDataModule
from llm.modelmodule import ModelModule, EmbeddingsModel


def main():
    # Set up paths and configurations
    checkpoint_dir = "checkpoints"
    data_path = "/home/ec2-user/SageMaker/dataset/"
    checkpoint_path = os.path.join(checkpoint_dir, "model-epoch=25-val_loss=0.38.ckpt")
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision("medium")
    # Create the data module
    datamodule = EmbeddingsDataModule(
        os.path.join(data_path, "train_subsampled_50_50.parquet"),
        os.path.join(data_path, "val_subsampled_50_50.parquet"),
        os.path.join(data_path, "test.parquet"),
        dataset_dir=os.path.join(data_path, "molformer"),
    )

    # Create the model module
    model = EmbeddingsModel()
    modelmodule = ModelModule.load_from_checkpoint(checkpoint_path, model=model)

    # Define Lightning Trainer
    trainer = L.Trainer()
    # Make predictions
    predictions = trainer.predict(modelmodule, datamodule.test_dataloader())

    test_df = pl.scan_parquet(os.path.join(data_path, "test.parquet")).collect()
    all_probabilities = torch.cat(predictions, dim=0)
    probabilities_series = pl.Series("binds", all_probabilities.cpu().numpy().ravel())
    test_df = test_df.with_columns(probabilities_series)
    test_df = test_df.select(["id", "binds"])
    test_df.write_csv("test_predictions.csv")


if __name__ == "__main__":
    main()
