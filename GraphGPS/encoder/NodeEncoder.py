import torch
import torch.nn as nn

import util.register as register
from ogb.graphproppred.mol_encoder import AtomEncoder
from util import cfg
from util.register import register_node_encoder

register.node_encoder_dict["Atom"] = AtomEncoder
__all__ = ["ConcatNodeEncoder"]


@register_node_encoder("RWSE")
class RWSENodeEncoder(torch.nn.Module):
    def __init__(self, dim_pe, batch_norm=True) -> None:
        super().__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm = nn.BatchNorm1d(dim_pe)
        self.pe_encoder = nn.Linear(dim_pe, dim_pe)

    def forward(self, batch):
        if self.batch_norm:
            batch = self.norm(batch)
        batch = self.pe_encoder(batch)
        return batch


class ConcatNodeEncoder(torch.nn.Module):
    def __init__(self, dim_hidden, dim_pe, batch_norm=True) -> None:
        super().__init__()
        encoder1, encoder2 = cfg.dataset.node_encoder_name.split("+")
        if encoder1 in register.node_encoder_dict:
            self.encoder1 = register.node_encoder_dict[encoder1](dim_hidden - dim_pe)
        else:
            raise ValueError(f"Unsupported encoder1 {encoder1}")

        if encoder2 in register.node_encoder_dict:
            self.encoder2 = register.node_encoder_dict[encoder2](
                dim_pe, batch_norm=batch_norm
            )
        else:
            raise ValueError(f"Unsupported encoder2 {encoder2}")

    def forward(self, feature, pe):
        h = torch.concat((self.encoder1(feature), self.encoder2(pe)), dim=1)
        return h
