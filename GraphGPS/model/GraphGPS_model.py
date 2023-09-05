"""
[Recipe for a General, Powerful, Scalable Graph Transformer]
(http://arxiv.org/abs/2205.12454)
"""

import dgl.nn as dglnn
import torch
import torch.nn as nn
from util.config import cfg

from GraphGPS.encoder import FeatureEncoder
from GraphGPS.layer import GraphGPSLayer


class GraphGPSModel(nn.Module):
    def __init__(
        self,
        out_size,
        hidden_size=64,
        pos_enc_size=2,
        num_layers=10,
        num_heads=4,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(hidden_size, pos_enc_size)
        local_gnn, global_attn = cfg.gt.layer_type.split("+")
        self.layers = nn.ModuleList(
            [
                GraphGPSLayer(
                    hidden_size,
                    num_heads,
                    local_gnn,
                    global_attn,
                    dropout=cfg.gt.dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.pooler = dglnn.SumPooling()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, out_size),
        )

    def forward(self, g):
        h, e = self.encoder(g)
        for layer in self.layers:
            h, e = layer(g, h, e)
        h = self.pooler(g, h)

        return self.predictor(h)


@torch.no_grad()
def evaluate(model, dataloader, evaluator, device):
    model.eval()
    y_true = []
    y_pred = []
    for batched_g, labels in dataloader:
        batched_g, labels = batched_g.to(device), labels.to(device)
        y_hat = model(batched_g)
        y_true.append(labels.view(y_hat.shape).detach().cpu())
        y_pred.append(y_hat.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)["rocauc"]
