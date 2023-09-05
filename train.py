"""
[Recipe for a General, Powerful, Scalable Graph Transformer]
(http://arxiv.org/abs/2205.12454)
"""

import time

import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.dataloading import GraphDataLoader

from GraphGPS.model import GraphGPSModel
from GraphGPS.optimizer import get_cosine_schedule_with_warmup
from ogb.graphproppred import collate_dgl
from tqdm import tqdm
from util import load_dataset
from util.config import cfg, load_cfg, parse_args, set_cfg


def train(model, dataset, evaluator, device, train_idx, val_idx, test_idx):
    # Set dataloader
    train_dataloader = GraphDataLoader(
        dataset[train_idx],
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_dgl,
    )
    valid_dataloader = GraphDataLoader(
        dataset[val_idx],
        batch_size=cfg.train.batch_size,
        collate_fn=collate_dgl,
    )
    test_dataloader = GraphDataLoader(
        dataset[test_idx],
        batch_size=cfg.train.batch_size,
        collate_fn=collate_dgl,
    )

    # Set optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    num_epochs = cfg.optim.max_epoch
    if cfg.optim.scheduler == "cosine_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, cfg.optim.num_warmup_epochs, num_epochs
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=num_epochs, gamma=0.5
        )
    # loss_fcn = nn.BCEWithLogitsLoss()
    loss_fcn = nn.L1Loss()

    epochs_time = []

    # Start training
    for epoch in range(num_epochs):
        time_start = time.perf_counter()
        torch.cuda.synchronize(0)
        model.train()
        total_loss = 0.0
        for batched_g, labels in train_dataloader:
            batched_g, labels = batched_g.to(device), labels.to(device)
            logits = model(batched_g)
            labels = labels.view(-1, 1)
            loss = loss_fcn(logits, labels.float())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        torch.cuda.synchronize(0)
        total_time = time.perf_counter() - time_start
        epochs_time.append(total_time)
        avg_loss = total_loss / len(train_dataloader)
        # val_metric = evaluate(model, valid_dataloader, evaluator, device)
        # test_metric = evaluate(model, test_dataloader, evaluator, device)
        print(
            f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, "
            f"Time used: {total_time:.1f} "
            f"Batch num {len(train_dataloader)}, avg time {total_time/len(train_dataloader):.6f}"
        )
    print(f"Average epoch time {sum(epochs_time)/len(epochs_time)}")


if __name__ == "__main__":
    args = parse_args()
    set_cfg(cfg)
    load_cfg(cfg, args)
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    pos_enc_size = cfg.posenc_RWSE.dim_pe
    dataset, evaluator, train_idx, val_idx, test_idx = load_dataset(cfg.dataset.name)

    if args.train_subset:
        subset_ratio = 0.1
        subtrain_idx = train_idx[: int(subset_ratio * len(train_idx))]
        subvalid_idx = val_idx[:5000]
        subtest_idx = val_idx[:5000]  # The original 'valid' as testing set.
        dataset = dataset[torch.cat([subtrain_idx, subvalid_idx, subtest_idx])]
        n1, n2, n3 = len(subtrain_idx), len(subvalid_idx), len(subtest_idx)
        print(f"test set {n1}, val set {n2}, test set {n3}")
        train_idx = list(range(n1))
        val_idx = list(range(n1, n1 + n2))
        test_idx = list(range(n1 + n2, n1 + n2 + n3))

    # Positional encoding
    if not args.test and cfg.dataset.pos_encoder is not None:
        if cfg.dataset.pos_encoder == "RWSE":
            pos_encoder = dgl.random_walk_pe
        elif cfg.dataset.pos_encoder == "LAP":
            pos_encoder = dgl.laplacian_pe
        else:
            raise ValueError(f"Unsupprot Positional Encoding {cfg.dataset.pos_encoder}")
        time_start = time.perf_counter()
        for g, _ in tqdm(dataset, desc="Computing PE", position=0, leave=False):
            g.ndata["PE"] = pos_encoder(g, k=pos_enc_size)
        print(f"PE precompute time {time.perf_counter()-time_start}")
    else:
        for g, _ in tqdm(dataset, position=0, leave=False):
            g.ndata["PE"] = torch.zeros((g.num_nodes(), pos_enc_size))

    # Create model.
    out_size = 1
    model = GraphGPSModel(
        out_size=out_size,
        hidden_size=cfg.gt.dim_hidden,
        pos_enc_size=pos_enc_size,
        num_layers=cfg.gt.layers,
        num_heads=cfg.gt.n_heads,
    ).to(dev)
    print("batch_size", cfg.train.batch_size)
    print(model)

    # Kick off training.
    train(model, dataset, evaluator, dev, train_idx, val_idx, test_idx)
