import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from ogb.lsc import DglPCQM4Mv2Dataset
from tqdm import tqdm


def load_data(dataset_name, batch_size):
    if dataset_name == "PCQM4Mv2-full":
        dataset = DglPCQM4Mv2Dataset(root="./data")
    else:
        dataset = DglGraphPropPredDataset(dataset_name, "./data/OGB")
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    train_dataloader = GraphDataLoader(
        dataset[train_idx], batch_size=batch_size, shuffle=False, collate_fn=collate_dgl
    )
    return train_dataloader


def dump_batch(dataset_name, train_dataloader, batch_size, num_avg, std_avg):
    if train_dataloader is None:
        train_dataloader = load_data(dataset_name, batch_size)

    def error(val, mean):
        return abs(val - mean) / mean < 0.01

    for batched_g, labels in tqdm(train_dataloader):
        num = batched_g.num_nodes()
        std = torch.std(batched_g.batch_num_nodes().float())
        if error(num, num_avg) and error(std, std_avg):
            print(f"num {num}, num_avg {num_avg}")
            print(f"std {std}, num_avg {std_avg}")
            with open(f"data/dump/{dataset_name}_graph_{batch_size}.pickle", "wb") as f:
                pickle.dump(batched_g, f)
            with open(f"data/dump/{dataset_name}_label_{batch_size}.pickle", "wb") as f:
                pickle.dump(labels, f)
            print("dump successful")
            break


def figure_num_std(
    dataset_name, batch_size, num, num_avg, num_edges, num_edges_avg, std, std_avg
):
    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.boxplot(num)
    plt.title(
        f"{dataset_name} # of nodes per batch, mean {num_avg:.2f}",
        fontdict={"fontsize": 20},
    )
    plt.savefig(f"figure/{dataset_name}_num_nodes_{batch_size}.png")
    fig.clear()

    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.boxplot(num_edges)
    plt.title(
        f"{dataset_name} # of edges per batch, mean {num_edges_avg:.2f}",
        fontdict={"fontsize": 20},
    )
    plt.savefig(f"figure/{dataset_name}_num_edges_{batch_size}.png")
    fig.clear()

    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.boxplot(std)
    plt.title(
        f"{dataset_name} std of nodes per batch, mean {std_avg:.2f}",
        fontdict={"fontsize": 20},
    )
    # plt.ylim(0, 60)
    plt.savefig(f"figure/{dataset_name}_std_nodes_{batch_size}.png")


if __name__ == "__main__":

    # load dataset
    # dataset_name = "ogbg-molhiv"
    dataset_name = "PCQM4Mv2-full"
    batch_size = 256 if dataset_name == "PCQM4Mv2-full" else 32

    train_dataloader = None
    if not os.path.exists(f"test/{dataset_name}_num_edges_{batch_size}.npy"):
        train_dataloader = load_data(dataset_name, batch_size)
        num_nodes_batch = []
        num_edges_batch = []
        std_nodes_batch = []
        for batched_g, _ in tqdm(train_dataloader):
            num_nodes_batch.append(batched_g.num_nodes())
            num_edges_batch.append(batched_g.num_edges())
            std_nodes_batch.append(torch.std(batched_g.batch_num_nodes().float()))
        num_nodes_batch = np.array(num_nodes_batch)
        np.save(f"test/{dataset_name}_num_nodes_{batch_size}.npy", num_nodes_batch)
        num_edges_batch = np.array(num_edges_batch)
        np.save(f"test/{dataset_name}_num_edges_{batch_size}.npy", num_edges_batch)
        std_nodes_batch = np.array(std_nodes_batch)
        np.save(f"test/{dataset_name}_std_nodes_{batch_size}.npy", std_nodes_batch)
    else:
        num_nodes_batch = np.load(f"test/{dataset_name}_num_nodes_{batch_size}.npy")
        num_edges_batch = np.load(f"test/{dataset_name}_num_edges_{batch_size}.npy")
        std_nodes_batch = np.load(f"test/{dataset_name}_std_nodes_{batch_size}.npy")

    num_nodes_avg = np.mean(num_nodes_batch)
    num_edges_avg = np.mean(num_edges_batch)
    std_nodes_avg = np.mean(std_nodes_batch)

    figure_num_std(
        dataset_name,
        batch_size,
        num_nodes_batch,
        num_nodes_avg,
        num_edges_batch,
        num_edges_avg,
        std_nodes_batch,
        std_nodes_avg,
    )

    dump_batch(dataset_name, train_dataloader, batch_size, num_nodes_avg, std_nodes_avg)
