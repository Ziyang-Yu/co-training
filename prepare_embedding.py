import types

import dgl
import tqdm

import json
import os
from cotraining import deberta, init_dataloader, load_data

import torch


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

graph, num_classes, text = load_data('ogbn-arxiv', use_dgl=True, use_text=True)
# graph.ndata['x'] = torch.load('arxiv_deberta.pt').squeeze()
graph = dgl.to_bidirected(graph, copy_ndata=True)
graph = dgl.remove_self_loop(graph)
graph = dgl.add_self_loop(graph)

config = {
    "seed": 42,
    "device": 'cuda',
    "epoch": 2000,

    "lm_type": 'deberta-base',
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
    "gnn_h_feats": 256,
    "gnn_lr": 0.0005,
    "gnn_weight_decay": 0,
    "gnn_dropout": 0.5,
    "gnn_requires_grad": True,
    "gnn_num_layers":7,

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
}
config = dict_to_namespace(config)
config.epoch

train_dataloader, valid_dataloader, test_dataloader = init_dataloader(graph, 'train', config), init_dataloader(graph, 'val', config), init_dataloader(graph, 'test', config)

seen_ids = []


save_dir = './cache'
flag_file = os.path.join(save_dir, 'last_run.json')
if os.path.exists(flag_file):
    with open(flag_file, 'r') as f:
        data = json.load(f)
else:
    data = {}
finished = data.get('finished', 0)

lm = deberta(config=config).to(config.device)
lm.eval()
batch_size = 128 
for i in tqdm.tqdm(range(finished, len(text), batch_size)):
    batch = text[i:i+batch_size]
    save_name = os.path.join(save_dir, f'batch_{i // batch_size}.pth')
    with torch.no_grad():
        outputs = lm(batch).to(torch.device('cpu'))
        torch.save(outputs, save_name)
        with open(flag_file, 'w') as f:
            finished = i + batch_size
            data['finished'] = finished
            json.dump(data, f)

emb_list = []
for b in range((len(text) + batch_size - 1) // batch_size):
    save_name = os.path.join(save_dir, f'batch_{b}.pth')
    emb = torch.load(save_name)
    emb = emb.view(-1, emb.shape[-1])
    emb_list.append(emb)
emb = torch.concat(emb_list)
save_name = os.path.join(save_dir, f'cache_emb.pth')
torch.save(emb, save_name)
print(emb.shape, len(text), finished)

