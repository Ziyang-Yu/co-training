
#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import concurrent.futures
import tqdm
from typing import Optional
from argparse import Namespace
import json
import types

import torch
import torch.nn.functional as F
import sklearn
import numpy as np
import dgl

from cotraining import load_data, graphsage, opt_1_3b, init_dataloader, seed, save_exp



def dict_to_namespace(d):
    """
    Recursively converts a dictionary to a SimpleNamespace.
    
    Args:
        d (dict): The dictionary to convert.
        
    Returns:
        SimpleNamespace: The converted namespace.
    """
    if isinstance(d, dict):
        # Convert sub-dictionaries to SimpleNamespace recursively
        return types.SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        # Return non-dictionary values as-is
        return d

# In[ ]:


# training params:
config = {
    "seed": 42,
    "device": 'cuda',
    "epoch": 2000,

    "lm_type": 'facebook/opt-1.3b',
    "lm_lr": 0,
    "lm_max_length": 512,
    "lm_weight_decay": 1e-4,
    "lm_padding": True,
    "lm_truncation": True,
    "lm_requires_grad": True,
    "pooler_hidden_size": 768, 
    "pooler_dropout": 0.5,
    "pooler_hidden_act": 'relu',

    "num_nodes": 169343,
    "num_node_features": 768,
    "gnn_h_feats": 768,
    "gnn_lr": 0.0005,
    "gnn_weight_decay": 0,
    "gnn_dropout": 0.5,
    "gnn_requires_grad": True,
    "gnn_num_layers":7,
    "gnn_use_residual": True,

    "once_batch_size": 1024,
    "once_shuffle": True,
    "once_drop_last": True,

    "train_batch_size": 16,
    "train_shuffle": True,
    "train_drop_last": True,

    "valid_batch_size": 16,
    "valid_shuffle": True,
    "valid_drop_last": True,

    "test_batch_size": 16,
    "test_shuffle": True,
    "test_drop_last": True,

    "use_param_free_pooler": True,
    "leading_alpha": 0.9,
    "use_node_cache": True,
    "node_cache": "cache/cache_emb.pth",

    "log_dir": "log/exp-name", 
    "save_interval": 5,
    "save_latest": True,
    "resume": False,
}
config = dict_to_namespace(config)

writer, saver, loader = save_exp(config)

seed(config.seed)

graph, num_classes, text = load_data('ogbn-arxiv', use_dgl=True, use_text=True)
# graph.ndata['x'] = torch.load('arxiv_deberta.pt').squeeze()
graph = dgl.to_bidirected(graph, copy_ndata=True)
graph = dgl.remove_self_loop(graph)
graph = dgl.add_self_loop(graph)

lm = opt_1_3b(config=config).to(config.device)
res = []
for i, t in enumerate(text):
    if i % 50 == 0:
        print(i)
    with torch.no_grad():
        res.append(lm(t).detach().cpu())
res = torch.stack(res)
torch.save(res, "cache_emb_opt_1_3b.pth")
