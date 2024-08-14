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

from cotraining import *

# In[ ]:

123456


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

    "lm_type": 'deberta-base',
    "lm_lr": 0.,
    "lm_max_length": 512,
    "lm_weight_decay": 1e-4,
    "lm_padding": True,
    "lm_truncation": True,
    "lm_requires_grad": False,
    "pooler_hidden_size": 768, 
    "pooler_dropout": 0.5,
    "pooler_hidden_act": 'relu',

    "num_nodes": 169343,
    "num_node_features": 768,
    "gnn_h_feats": 256,
    "gnn_lr": 0.0005,
    "gnn_weight_decay": 0,
    "gnn_dropout": 0.5,
    "gnn_requires_grad": True,
    "gnn_num_layers":2,

    "once_batch_size": 64,
    "once_shuffle": True,
    "once_drop_last": True,

    "train_batch_size": 4,
    "train_shuffle": True,
    "train_drop_last": True,

    "valid_batch_size": 1024,
    "valid_shuffle": True,
    "valid_drop_last": True,

    "test_batch_size": 1024,
    "test_shuffle": True,
    "test_drop_last": True,
}

config = dict_to_namespace(config)

lm = bert(config=config).to(config.device)

graph, num_classes, text = load_data('ogbn-arxiv', use_dgl=True, use_text=True)


def get_cuda_memory_info():
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        cached_memory = torch.cuda.memory_reserved(0)

        print(f"Total Memory: {total_memory / (1024 ** 3):.2f} GB")
        print(f"Allocated Memory: {allocated_memory / (1024 ** 3):.2f} GB")
        print(f"Cached Memory: {cached_memory / (1024 ** 3):.2f} GB")
    else:
        print("CUDA is not available.")

import time
time0 = time.time()
get_cuda_memory_info()
# with torch.no_grad():
inputs = lm(text[:64])
get_cuda_memory_info()
time1 = time.time()
print(time1 - time0)