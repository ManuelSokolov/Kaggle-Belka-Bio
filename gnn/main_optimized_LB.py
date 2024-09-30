import subprocess
import sys
import argparse
import os

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main(train_path, val_path, test_path, proteins, config_path, model_type, api_key_path, packages_file):
    # Read required packages from file
    with open(packages_file, "r") as f:
        required_packages = [line.strip() for line in f.readlines()]

    # Install required packages if not already installed
    installed_packages = subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode("utf-8")
    for package in required_packages:
        if package not in installed_packages:
            print(f"Installing {package}...")
            install_package(package)

    # Now import the necessary modules after installing the packages
    import pandas as pd
    import torch
    from torch_geometric.loader import DataLoader
    import pytorch_lightning as L
    from gnn_lightning import GNNModel 
    from gat_lightning import GATModel
    from gnn_utils import DatasetLoader
    from pytorch_lightning.loggers import NeptuneLogger
    from pytorch_lightning.callbacks import EarlyStopping
    from tqdm import tqdm
    import yaml

    # Rest of the main function remains unchanged
    # Load model parameters from the YAML file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    hidden_dim = config["hidden_dim"]
    num_epochs = config["num_epochs"]
    num_layers = config["num_layers"]
    dropout_rate = config["dropout_rate"]
    lr = config["learning_rate"]
    batch_size = config["batch_size"]
    heads = config["heads"]

    # Load API key from the file
    with open(api_key_path, "r") as f:
        api_key = f.read().strip()

    all_predictions = []

    neptune_logger = NeptuneLogger(
        api_key=api_key,
        project="manuel-loka/belka",
    )

    for protein in proteins:
        train_loader = DatasetLoader(train_path, protein=protein)
        positives = train_loader.get_number_of_positives()
        print(positives)
        train_loader = DatasetLoader(train_path, protein=protein, positives=True, number=positives)
        train_dataloader = DataLoader(train_loader, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True, pin_memory=True)
        input_dim = train_dataloader.dataset[0].num_node_features

        val_loader = DatasetLoader(val_path, protein=protein, stage="val")
        val_dataloader = DataLoader(val_loader, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True)

        test_loader = DatasetLoader(test_path, protein=protein, stage="test")
        test_dataloader = DataLoader(test_loader, batch_size=batch_size, num_workers=os.cpu_count())
        
        if model_type == 'gat':
            model = GATModel(protein, input_dim, hidden_dim, num_layers, heads, dropout_rate, learning_rate=lr, loss="bce")
        else:   
            model = GNNModel(protein, input_dim, hidden_dim, num_layers, dropout_rate, learning_rate=lr, loss="bce")

        early_stopping_callback = EarlyStopping(
            monitor=f'val_loss_{protein}',  # Monitor validation loss
            patience=3,  # Number of epochs with no improvement to wait
            verbose=True,
            mode='min'
        )

        trainer = L.Trainer(
            max_epochs=num_epochs, 
            logger=neptune_logger, 
            log_every_n_steps=10,  # Log every 10 steps
            enable_progress_bar=True,  # Show progress bar
            enable_checkpointing=False, 
            callbacks=[early_stopping_callback]
        )

        trainer.fit(model=model, train_dataloaders=train_dataloader) #val_dataloaders=val_dataloader)

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
    final_predictions.to_csv("sample_submission.csv")

    print("All predictions saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN model training and evaluation.")
    parser.add_argument("--train_path", type=str, default="../dataset/train_subsampled_50_50.parquet", help="Path to the training dataset")
    parser.add_argument("--val_path", type=str, default="../dataset/val_subsampled_bb_split.parquet", help="Path to the validation dataset")
    parser.add_argument("--test_path", type=str, default="../dataset/test.parquet", help="Path to the test dataset")
    parser.add_argument("--proteins", nargs='+', default=['sEH', 'BRD4', 'HSA'], help="List of proteins to train and evaluate on")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--model_type", type=str, choices=['gnn', 'gat'], default="gnn", help="Type of model to use: 'gnn' or 'gat'")
    parser.add_argument("--api_key_path", type=str, default="neptune_api_key.txt", help="Path to the file containing Neptune API key")
    parser.add_argument("--packages_file", type=str, default="required_packages.txt", help="Path to the file containing required packages")

    args = parser.parse_args()

    main(
        args.train_path,
        args.val_path,
        args.test_path,
        args.proteins,
        args.config_path,
        args.model_type,
        args.api_key_path,
        args.packages_file
    )
