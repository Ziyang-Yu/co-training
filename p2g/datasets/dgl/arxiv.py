import pandas as pd
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset


def get_raw_text_arxiv(dataset_root, use_text=False, seed=0):

    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=dataset_root)
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits["train"]] = True
    val_mask[idx_splits["valid"]] = True
    test_mask[idx_splits["test"]] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        dataset_root + "/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz", compression="gzip"
    )

    raw_text = pd.read_csv(
        dataset_root + "/ogbn_arxiv_orig/titleabs.tsv",
        sep="\t",
        header=None,
        names=["paper id", "title", "abs"],
    )
    raw_text["paper id"] = raw_text["paper id"]
    df = pd.merge(nodeidx2paperid, raw_text, on="paper id")
    text = []
    for ti, ab in zip(df["title"], df["abs"]):
        t = "Title: " + ti + "\n" + "Abstract: " + ab
        text.append(t)
    return data, text
