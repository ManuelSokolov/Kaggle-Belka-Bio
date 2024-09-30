import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import pytorch_lightning as L
from gnn_lightning import GNNModel  # Assuming GNNModel is defined correctly with validation in gnn_lightning
from gnn_utils import DatasetLoader
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, precision_recall_curve, auc, average_precision_score
from pytorch_lightning.loggers import NeptuneLogger
import os 


# Track things
from tqdm import tqdm

# Constants
train_path = "dataset/train_subsampled_bb_split.parquet"
val_path = "dataset/val_subsampled_bb_split.parquet"
test_path = "dataset/test.parquet"
proteins = ['sEH', 'BRD4', 'HSA']
batch_size = 2**8  # 256

# Hyperparameters
hidden_dim = 128
num_epochs = 5
num_layers = 8
dropout_rate = 0.3
lr = 0.1

all_predictions = []

#run = neptune.init_run(project='manuel-loka/belka')

neptune_logger = NeptuneLogger( api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZWUzMmZhOS0zMzA3LTRiZjMtOTdiMi0zZmFhZjgxYTJiZTEifQ==',
        project="manuel-loka/belka",
)

for protein in proteins:
    train_loader = DatasetLoader(train_path, protein=protein)
    train_dataloader = DataLoader(train_loader, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True, pin_memory=True)
    input_dim = train_dataloader.dataset[0].num_node_features
    
    val_loader = DatasetLoader(val_path, protein=protein, stage="val")
    val_dataloader = DataLoader(val_loader, batch_size=batch_size, num_workers=os.cpu_count(), shuffle = True)
    
    test_loader = DatasetLoader(test_path, protein=protein, stage="test")
    test_dataloader = DataLoader(test_loader, batch_size=batch_size, num_workers=os.cpu_count())
    
    model = GNNModel(protein, input_dim, hidden_dim, num_layers, dropout_rate, learning_rate=lr, loss="focal_loss")
    
    trainer = L.Trainer(
        max_epochs=num_epochs, 
        logger=neptune_logger, 
        log_every_n_steps=10,  # Log every 10 steps
        enable_progress_bar=True,  # Show progress bar
        enable_checkpointing=False,  # Disable checkpointing to avoid too many logs
    )

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    predictions = trainer.predict(model, dataloaders=test_dataloader)
    predictions = torch.cat(predictions, dim=0).squeeze().numpy()
    
    protein_predictions = pd.DataFrame({
        'id': test_loader.dataset['id'],
        'binds': predictions
    })
    all_predictions.append(protein_predictions)

    torch.save(model.state_dict(), f'{protein}_model_state.pth')
    print("Model training and prediction for", protein, "completed.")

final_predictions = pd.concat(all_predictions, ignore_index=True)
final_predictions = final_predictions.set_index('id')
final_predictions.to_csv("dataset/sample_submission.csv")

print("All predictions saved.")

