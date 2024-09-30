"""Module for the MoLFormer model."""

import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import AveragePrecision
from transformers import AutoConfig, AutoModel
from torchmetrics import AveragePrecision, Precision, Recall, AUROC
from torchvision.ops import sigmoid_focal_loss
from peft import (
    LoftQConfig,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import time 


class EmbeddingsModel(nn.Module):

    def __init__(
        self,
        input_dim=768 + 3,
        loss="bce",
        hidden_dim1=256,
        hidden_dim2=128,
        dropout=0.5,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.loss = loss

    def forward(self, batch):
        "Forward pass for the given batch."
        embeddings = batch["embeddings"]
        protein_features = batch["protein"]
        concatenated_features = torch.cat((embeddings, protein_features), dim=1)
        x = F.relu(self.fc1(concatenated_features))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def calculate_loss(self, batch, logits):
        if self.loss == "focal_loss":
            loss = sigmoid_focal_loss(logits, batch["labels"].float(), reduction="mean")
        elif self.loss == "bce_weights":
            labels = batch["labels"].float()
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["labels"].float(), reduction="none"
            )
            weighted_loss = 5 * labels * bce_loss + (1 - labels) * bce_loss
            loss = torch.mean(weighted_loss)
        else:
            labels = batch["labels"].float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["labels"].float(), reduction="mean"
            )
        return loss


class LMModel(nn.Module):

    def __init__(
        self, model_name="ibm/MoLFormer-XL-both-10pct", loss="focal_loss", lora=True
    ):
        "Initialize the model."
        super().__init__()
        print(model_name)
        self.config = AutoConfig.from_pretrained(
            model_name, num_labels=1, trust_remote_code=True
        )
        self.num_proteins = 3
        self.loss = loss

        base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        if lora:
            base_model = prepare_model_for_kbit_training(base_model)
            loftq_config = LoftQConfig(loftq_bits=4)
            lora_config = LoraConfig(
                r=8,
                init_lora_weights="loftq",
                target_modules=["query", "key"],  # attention mechanism
                loftq_config=loftq_config,
                task_type="FEATURE_EXTRACTION",
            )
            self.lm = get_peft_model(base_model, lora_config)
        else:
            self.lm = base_model

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        print(self.config.hidden_size)
        self.linear = nn.Linear(
            self.config.hidden_size + self.num_proteins, self.config.num_labels
        )

    def forward(self, batch):
        "Forward pass for the given batch."
        output = self.lm(batch["input_ids"], attention_mask=batch["attention_mask"])
        protein_features = batch["protein"]
        concatenated_features = torch.cat(
            (output.last_hidden_state[:, 0], protein_features), dim=1
        )
        x = self.dropout(concatenated_features)
        x = self.linear(x)
        return x

    def calculate_loss(self, batch, logits):
        if self.loss == "focal_loss":
            loss = sigmoid_focal_loss(logits, batch["labels"].float(), reduction="mean")
        elif self.loss == "bce_weights":
            labels = batch["labels"].float()
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["labels"].float(), reduction="none"
            )
            weighted_loss = 5 * labels * bce_loss + (1 - labels) * bce_loss
            loss = torch.mean(weighted_loss)
        else:
            labels = batch["labels"].float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["labels"].float(), reduction="mean"
            )
        return loss


class ModelModule(L.LightningModule):
    "Lightning module for the ligand binding model."

    def __init__(
        self,
        model,
        learning_rate=0.0001,
        loss="focal_loss",
        optimizer="Adam",
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model
        self.loss = loss
        self.optimizer = optimizer


        self.val_map = AveragePrecision(task="binary")
        self.val_precision = Precision(num_classes=1, average="binary", task="binary")
        self.val_recall = Recall(num_classes=1, average="binary", task="binary")
        self.val_auroc = AUROC(task="binary")
        
        self.train_map = AveragePrecision(task="binary")
        self.train_precision = Precision(num_classes=1, average="binary", task="binary")
        self.train_recall = Recall(num_classes=1, average="binary", task="binary")
        self.train_auroc = AUROC(task="binary")

    def forward(self, batch):
        "forward pass for the given batch."
        logits = self.model(batch)
        return logits

    def calculate_loss(self, batch, logits):
        "Calculate loss for the given batch."
        return self.model.calculate_loss(batch, logits)

    def training_step(self, batch):
        """Training step for the given batch."""
        logits = self.forward(batch)
        train_loss = self.calculate_loss(batch, logits)
        probas = F.sigmoid(logits)

        self.train_map.update(probas, batch["labels"].long())
        self.train_precision.update(probas, batch["labels"].long())
        self.train_recall.update(probas, batch["labels"].long())
        self.train_auroc.update(probas, batch["labels"].long())

        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        return train_loss
    
    def on_train_epoch_end(self):
        self.log(
            "train_map_epoch",
            self.train_map.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "train_precision_epoch",
            self.train_precision.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "train_recall_epoch",
            self.train_recall.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "train_auroc_epoch",
            self.train_auroc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.train_map.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_auroc.reset()
        
    def validation_step(self, batch):
        """Validation step for the given batch."""
        logits = self.forward(batch)
        val_loss = self.calculate_loss(batch, logits)
        probas = F.sigmoid(logits)

        self.val_map.update(probas, batch["labels"].long())
        self.val_precision.update(probas, batch["labels"].long())
        self.val_recall.update(probas, batch["labels"].long())
        self.val_auroc.update(probas, batch["labels"].long())
  
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        return val_loss

    def on_validation_epoch_end(self):
        self.log(
            "val_map",
            self.val_map.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "val_precision",
            self.val_precision.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "val_recall",
            self.val_recall.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "val_auroc",
            self.val_auroc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.val_map.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_auroc.reset()
        
    def predict_step(self, batch):
        "Predict step for the given batch."
        logits = self.forward(batch)
        probs = F.sigmoid(logits)
        return probs

    def configure_optimizers(self):
        "Configure the optimizer."
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
        }
