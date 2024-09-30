# GNN
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as L
from torch_geometric.nn import MessagePassing, global_max_pool
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from torchmetrics import AveragePrecision, Precision, Recall, AUROC
from torchvision.ops import sigmoid_focal_loss
#from torch_scatter import scatter

class CustomGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomGNNLayer, self).__init__(aggr='max')
        self.lin = nn.Linear(in_channels + 6, out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        combined = torch.cat((x_j, edge_attr), dim=1)
        return combined

    def update(self, aggr_out):
        return self.lin(aggr_out)

class GNNModel(L.LightningModule):
    def __init__(self, protein, input_dim, hidden_dim, num_layers, dropout_rate, learning_rate=1e-3, loss='bce'):
        super(GNNModel, self).__init__()
        self.save_hyperparameters()
        self.protein = protein

        self.convs = nn.ModuleList([
            CustomGNNLayer(input_dim if i == 0 else hidden_dim, hidden_dim) 
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.lin = nn.Linear(hidden_dim, 1)
        self.criterion = BCEWithLogitsLoss()
        self.focal_loss = sigmoid_focal_loss
        self.loss = loss
        
        self.map = AveragePrecision(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.auroc = AUROC(task='binary')

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.hparams.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_max_pool(x, data.batch)
        x = self.lin(x)
        return x

    def step(self, batch, stage):
        y_hat = self(batch)
        if self.loss == "focal_loss":
            loss = self.focal_loss(y_hat, batch.y.view(-1, 1), reduction="mean")
        else:
            loss = self.criterion(y_hat, batch.y.view(-1, 1))

        probas = torch.sigmoid(y_hat)        
        map_score = self.map(probas, batch.y.view(-1, 1).long())
        precision = self.precision(probas, batch.y.view(-1, 1).long())
        recall = self.recall(probas, batch.y.view(-1, 1).long())
        auroc = self.auroc(probas, batch.y.view(-1, 1).long())
        
        self.log(f"{stage}_loss_{self.protein}", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_map_{self.protein}", map_score, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_precision_{self.protein}", precision, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_recall_{self.protein}", recall, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_auroc_{self.protein}", auroc, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_graphs)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch)
        probs = torch.sigmoid(logits)
        return probs
