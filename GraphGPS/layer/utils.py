from typing import Optional, Tuple

import torch
from torch import Tensor


def to_dense_batch(
    x: Tensor,
    graph_nodes: Optional[Tensor] = None,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_nodes: int = 0,
) -> Tuple[Tensor, Tensor]:
    batch = torch.zeros((num_nodes), dtype=torch.int64, device="cuda:0")
    batch[graph_nodes[:-1].cumsum(0)] = 1
    batch = batch.cumsum(0)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    cum_nodes = torch.cat([batch.new_zeros(1), graph_nodes.cumsum(dim=0)])

    if max_num_nodes is None:
        max_num_nodes = int(graph_nodes.max())

    # cal the index of each nodes
    idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
    # the output size
    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = x.new_full(size, 0)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool, device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask
