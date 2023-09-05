import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv, GINEConv

from GraphGPS.layer.GatedGCN import GatedGCNConv
from GraphGPS.layer.SparseMHA import SparseMHA
from GraphGPS.layer.utils import to_dense_batch


class GraphGPSLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(
        self, hidden_size, num_heads, local_gnn, attn_type, dropout=0, batch_norm=True
    ):
        super().__init__()
        self.activation = nn.ReLU()

        # Local GNN model
        self.local_gnn_with_edge_attr = True
        self.local_gnn = local_gnn
        if self.local_gnn == "None":
            self.local_model = None
        elif self.local_gnn == "GIN":
            gin_nn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                self.activation,
                nn.Linear(hidden_size, hidden_size),
            )
            self.local_model = GINConv(gin_nn)
            self.local_gnn_with_edge_attr = False
        elif self.local_gnn == "GINE":
            gin_nn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                self.activation,
                nn.Linear(hidden_size, hidden_size),
            )
            self.local_model = GINEConv(gin_nn)
        elif self.local_gnn == "GatedGCN":
            self.local_model = GatedGCNConv(
                hidden_size,
                hidden_size,
                hidden_size,
                dropout,
                batch_norm=True,
                residual=True,
            )
        else:
            raise ValueError(f"Unsupported Local GNN model {self.local_gnn}")

        # Global attention transformer model
        self.attn_type = attn_type
        if attn_type == None:
            self.global_attn = None
        elif attn_type == "Transformer":
            self.global_attn = torch.nn.MultiheadAttention(
                hidden_size, num_heads, dropout, batch_first=True
            )
        elif attn_type == "SPTransformer":
            self.global_attn = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        else:
            raise ValueError(
                f"Unsupported Global Attention Transformer model {attn_type}"
            )

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.norm_local = nn.BatchNorm1d(hidden_size)
            self.norm_attn = nn.BatchNorm1d(hidden_size)
            self.norm_out = nn.BatchNorm1d(hidden_size)

        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)

        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, g, h, e):
        h_in = h

        # Local MPNN
        if self.local_model is not None:
            if self.local_gnn == "GatedGCN":
                h_local, e = self.local_model(g, h, e)
            elif self.local_gnn_with_edge_attr:
                h_local = self.local_model(g, h, e)
            else:
                h_local = self.local_model(g, h)
            h_local = self.dropout_local(h_local)
            h_local = h_in + h_local
            if self.batch_norm:
                h_local = self.norm_local(h_local)

        # Multi-head attention
        h_attn = self.attn_block(g, h)
        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in + h_attn
        if self.batch_norm:
            h_attn = self.norm_attn(h_attn)

        # Combine the local and global outputs
        h = h_local + h_attn
        h = h + self.FFN2(F.relu(self.FFN1(h)))
        if self.batch_norm:
            h = self.norm_out(h)
        return h, e

    def attn_block(self, g, h):
        if self.attn_type == "Transformer":
            h_dense, mask = to_dense_batch(
                h, g.batch_num_nodes(), batch_size=g.batch_size, num_nodes=g.num_nodes()
            )
            x = self.global_attn(
                h_dense, h_dense, h_dense, key_padding_mask=~mask, need_weights=False
            )[0]
            h_attn = x[mask]
        elif self.attn_type == "SPTransformer":
            indices = torch.stack(g.edges())
            N = g.num_nodes()
            A = dglsp.spmatrix(indices, shape=(N, N))
            h_attn = self.global_attn(A, h)
        return h_attn
