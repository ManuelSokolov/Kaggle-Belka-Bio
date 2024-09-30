"Test script."
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import polars as pl
from transformers import AutoTokenizer
import torch
import pytorch_lightning as L

from llm.modelmodule import ModelModule, LMModel
from llm.datamodule import LBDataModule


def main():
    # Set up paths and configurations
    checkpoint_dir = "checkpoints"
    batch_size = 16
    data_path = "/home/ec2-user/SageMaker/dataset/"
    checkpoint_path = os.path.join(checkpoint_dir, "model-epoch=02-val_loss=0.68.ckpt")
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    lora = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding=True, return_tensors="pt"
    )
    model = LMModel(model_name=model_name, lora=lora)
    modelmodule = ModelModule.load_from_checkpoint(checkpoint_path, model=model)

    datamodule = LBDataModule(
        os.path.join(data_path, "train_subsampled_bb_split.parquet"),
        os.path.join(data_path, "val_subsampled_bb_split.parquet"),
        os.path.join(data_path, "test.parquet"),
        tokenizer,
        batch_size=batch_size,
    )

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
