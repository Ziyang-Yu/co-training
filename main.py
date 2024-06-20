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

from cotraining import *

# In[ ]:


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
    "lm_lr": 1e-4,
    "lm_max_length": 512,
    "lm_weight_decay": 1e-4,
    "lm_padding": True,
    "lm_truncation": True,
    "lm_requires_grad": False,
    "pooler_hidden_size": 128, 
    "pooler_dropout": 0.5,
    "pooler_hidden_act": 'relu',

    "num_nodes": 169343,
    "num_node_features": 128,
    "gnn_h_feats": 256,
    "gnn_lr": 0.0005,
    "gnn_weight_decay": 0,
    "gnn_dropout": 0.5,
    "gnn_requires_grad": True,
    "gnn_num_layers":7,

    "once_batch_size": 64,
    "once_shuffle": True,
    "once_drop_last": True,

    "train_batch_size": 1024,
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
config.epoch

# In[ ]:


seed(config.seed)


# with open('config/arxiv.json') as file:
#     config = json.loads(file.read())
# config = dict_to_namespace(config)


# In[ ]:


lm = deberta(config).to(config.device)

# In[ ]:


graph, num_classes, text = load_data('ogbn-arxiv', use_dgl=True, use_text=True)
graph = dgl.to_bidirected(graph, copy_ndata=True)
graph = dgl.remove_self_loop(graph)
graph = dgl.add_self_loop(graph)


# In[ ]:


len(text)

# In[ ]:


# features = torch.load('arxiv_deberta.pt')
# graph.ndata['x'] = torch.squeeze(features)

# In[ ]:


graph.ndata['x'].shape

# In[ ]:


# model = graphsage(num_nodes=graph.num_nodes(), in_feats=lm.__num_node_features__, h_feats=64, num_classes=num_classes).to(config.device)
model = graphsage(num_layers=config.gnn_num_layers, num_nodes=config.num_nodes, in_feats=config.num_node_features, h_feats=config.gnn_h_feats, num_classes=num_classes, dropout=config.gnn_dropout).to(config.device)

# In[ ]:


for param in lm.parameters():
    param.requires_grad = config.lm_requires_grad
for param in model.parameters():
    param.requires_grad = config.gnn_requires_grad

# In[ ]:


# opt = torch.optim.Adam(list(model.parameters())+list(lm.parameters())) # 
opt = torch.optim.Adam([
    {'params': lm.parameters(), 'lr': config.lm_lr, "weight_decay": config.lm_weight_decay},
    {'params': model.parameters(), 'lr': config.gnn_lr, "weight_decay": config.gnn_weight_decay}])

train_dataloader, valid_dataloader, test_dataloader = init_dataloader(graph, 'once', config)

# In[ ]:





# forward_once(train_dataloader, model)
# forward_once(valid_dataloader, model)
# forward_once(test_dataloader, model)
# torch.cuda.empty_cache()

# In[ ]:


train_dataloader, valid_dataloader, test_dataloader = init_dataloader(graph, 'train', config), init_dataloader(graph, 'val', config), init_dataloader(graph, 'test', config)

# In[ ]:



best_val_accuracy = 0.
best_model_path = 'model.pt'

for epoch in range(100):
    model.train()

    with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
            # print(output_nodes)
            # inputs = [text[i] for i in output_nodes]
            labels = mfgs[-1].dstdata['y']
            
            # inputs = lm(inputs).to(config.device)
            inputs = mfgs[0].srcdata['x']

            # print(inputs.shape, input_nodes.shape, output_nodes.shape, labels.shape)

            # predictions = model(mfgs=mfgs, x=inputs, batch_size=config.train_batch_size)
            predictions = model(mfgs=mfgs, x=inputs, batch_size=config.train_batch_size)
            labels = torch.flatten(labels)
            # print(predictions.device, labels.device)
            loss = F.cross_entropy(predictions, labels)
            # loss = torch.tensor(0.)

            opt.zero_grad()
            loss.backward()
            opt.step()

            accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())

            tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)

            del input_nodes, output_nodes, mfgs, inputs, labels, predictions, loss
            torch.cuda.empty_cache()
            # print(torch.cuda.mem_get_info())
    model.eval()

    predictions = []
    labels = []
    with torch.no_grad() and tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = mfgs[0].srcdata['x']
            labels.append(mfgs[-1].dstdata['y'].cpu().numpy())
            # predictions.append(model(mfgs=mfgs, x=inputs, batch_size=config.valid_batch_size).argmax(1).cpu().numpy())
            predictions.append(model(mfgs=mfgs, x=inputs, batch_size=config.valid_batch_size).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        val_accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        if best_val_accuracy <= val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model, best_model_path)

    best_model = torch.load(best_model_path)
    predictions = []
    labels = []
    with torch.no_grad() and tqdm.tqdm(test_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            # inputs = [text[i] for i in input_nodes]
            # print(type(mfgs[0]))
            inputs = mfgs[0].srcdata['x']
            labels.append(mfgs[-1].dstdata['y'].cpu().numpy())
            # inputs = lm(inputs).to(device)
            predictions.append(model(mfgs=mfgs, x=inputs, batch_size=config.test_batch_size).argmax(1).cpu().numpy())
            # predictions.append(model(mfgs=mfgs, x=inputs, batch_size=config.test_batch_size).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        # print(predictions)
        labels = np.concatenate(labels)
        test_accuracy = sklearn.metrics.accuracy_score(labels, predictions)

        # with open('log.txt', 'a') as file:
        #     file.write('Epoch {} Valid Accuracy {}  Best Accuracy {} Test Accuracy {}\n'.format(epoch, val_accuracy, best_val_accuracy, test_accuracy))

        print('Epoch {} Valid Accuracy {}  Best Accuracy {} Test Accuracy {}'.format(epoch, val_accuracy, best_val_accuracy, test_accuracy))
        # Note that this tutorial do not train the whole model to the end.
        # break
