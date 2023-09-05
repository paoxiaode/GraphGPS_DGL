import dgl
import torch

import util.register as register
from ogb.graphproppred.mol_encoder import BondEncoder
from util import cfg

from .NodeEncoder import ConcatNodeEncoder

register.edge_encoder_dict["Bond"] = BondEncoder
__all__ = ["FeatureEncoder"]


class FeatureEncoder(torch.nn.Module):
    def __init__(self, dim_hidden, dim_pe, batch_norm=True) -> None:
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_pe = dim_pe
        self.node_encoder = ConcatNodeEncoder(dim_hidden, dim_pe, batch_norm)
        if cfg.dataset.edge_encoder_name in register.edge_encoder_dict:
            self.edge_encoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name
            ](dim_hidden)
        else:
            raise ValueError(f"Unsupport edge encoder {cfg.dataset.edge_encoder_name}")

    def forward(self, g):
        if "PE" not in g.ndata:
            if cfg.dataset.pos_encoder == "RWSE":
                pos_encoder = dgl.random_walk_pe
            elif cfg.dataset.pos_encoder == "LAP":
                pos_encoder = dgl.laplacian_pe
            else:
                raise ValueError(
                    f"Unsupprot Positional Encoding {cfg.dataset.pos_encoder}"
                )
            g.ndata["PE"] = pos_encoder(g, k=self.dim_pe).to("cuda:0")
        h = self.node_encoder(g.ndata["feat"], g.ndata["PE"])
        edge_attr = self.edge_encoder(g.edata["feat"])
        return h, edge_attr
