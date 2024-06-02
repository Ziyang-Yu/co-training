#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from cotraining import *


# In[2]:


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


# In[3]:


seed(42)


with open('config/arxiv.json') as file:
    config = json.loads(file.read())
config = dict_to_namespace(config)


# In[4]:


lm = deberta().to(config.device)


# In[5]:


# from ogb.nodeproppred import DglNodePropPredDataset

graph, num_classes, text = load_data('ogbn-arxiv', use_dgl=True, use_text=True)


# In[6]:


features = torch.load('dataset/arxiv.pt')
graph.ndata['x'] = features


# In[7]:


model = graphsage(num_nodes= graph.num_nodes(), in_feats=lm.__num_node_features__, h_feats=64, num_classes=num_classes).to(config.device)


# In[8]:


# opt = torch.optim.Adam(list(model.parameters())+list(lm.parameters())) # 
opt = torch.optim.Adam([
    {'params': lm.parameters(), 'lr': 1e-4},
    {'params': model.parameters(), 'lr': 0.1}])

train_dataloader, valid_dataloader, test_dataloader = init_dataloader(graph, 'train', config), init_dataloader(graph, 'val', config), init_dataloader(graph, 'test', config)


# In[12]:


best_val_accuracy = 0
best_model_path = 'model.pt'


print('Initializing historical emb...')


forward_once(train_dataloader, model)
forward_once(valid_dataloader, model)
forward_once(test_dataloader, model)
torch.cuda.empty_cache()

print('Finish Initializing historical emb...')

# In[13]:


# with tqdm.tqdm(train_dataloader) as tq:
#     for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
#         # inputs = [text[i] for i in input_nodes]
#         with torch.no_grad():
#             # x = lm(inputs)
#             x = mfgs[0].srcdata['x']
#             model.forward_once(mfgs, x)

# with tqdm.tqdm(valid_dataloader) as tq:
#     for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
#         # inputs = [text[i] for i in input_nodes]
#         with torch.no_grad():
#             # x = lm(inputs)
#             x = mfgs[0].srcdata['x']
#             model.forward_once(mfgs, x)

# with tqdm.tqdm(test_dataloader) as tq:
#     for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
#         # inputs = [text[i] for i in input_nodes]
#         with torch.no_grad():
#             # x = lm(inputs)
#             x = mfgs[0].srcdata['x']
#             model.forward_once(mfgs, x)

for epoch in range(100):
    model.train()

    with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
            # print(output_nodes)
            inputs = [text[i] for i in output_nodes]
            labels = mfgs[-1].dstdata['y']
            
            inputs = lm(inputs).to(config.device)

            predictions = model(mfgs=mfgs, x=inputs, batch_size=config.train_loader.batch_size)
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
            predictions.append(model(mfgs=mfgs, x=inputs, batch_size=config.train_loader.batch_size).argmax(1).cpu().numpy())
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
            predictions.append(model(mfgs=mfgs, x=inputs, batch_size=config.train_loader.batch_size).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        # print(predictions)
        labels = np.concatenate(labels)
        test_accuracy = sklearn.metrics.accuracy_score(labels, predictions)

        print('Epoch {} Valid Accuracy {}  Best Accuracy {} Test Accuracy {}'.format(epoch, val_accuracy, best_val_accuracy, test_accuracy))
        # Note that this tutorial do not train the whole model to the end.
        # break

