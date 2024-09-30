"Fine-tuning Molformer on ligand binding dataset."
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.datamodule import LBDataModule
from llm.modelmodule import ModelModule, LMModel
from transformers import AutoTokenizer
import pytorch_lightning as L
from pytorch_lightning.loggers import NeptuneLogger
import torch


def main():
    batch_size = 16
    learning_rate = 1e-5
    num_epochs = 100
    data_path = "/home/ec2-user/SageMaker/dataset/" 
    model_name = "DeepChem/ChemBERTa-10M-MTR"
    loss_function = "bce"
    optimizer = "Adam"
    lora = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding=True, return_tensors="pt"
    )

    datamodule = LBDataModule(
        os.path.join(data_path, "train_subsampled_50_50.parquet"),
        os.path.join(data_path, "val_subsampled_50_50.parquet"),
        os.path.join(data_path, "test.parquet"),
        tokenizer,
        batch_size=batch_size,
    )
    model = LMModel(model_name=model_name, loss=loss_function, lora=lora)

    modelmodule = ModelModule(
        model, learning_rate, loss=loss_function, optimizer=optimizer
    )
    torch.set_float32_matmul_precision("medium")
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODY3NzNiZi1jYTM1LTRkM2MtYmE1Ny1hODRjN2UzNjBmMTQifQ==",
        project="cristiana.carpinteiro/kaggle-challenge",
        tags=["lora", "chemberta", "50_50"],
    )

    params = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "loss_function": loss_function,
        "learning_rate": learning_rate,
        "model_name": model_name,
        "optimizer": optimizer,
        "lora": lora,
    }

    neptune_logger.experiment["params"] = params

    checkpoint_callback = L.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        mode="min",
        monitor="val_loss",
    )

    early_stopping_callback = L.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )

    trainer = L.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=neptune_logger,
        log_every_n_steps=500,
        profiler="simple",
    )

    trainer.fit(modelmodule, datamodule)


if __name__ == "__main__":
    main()
