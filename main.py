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

from cotraining import load_data, graphsage, init_dataloader, seed, save_exp
from cotraining.models import CroppedLlama2, NonParamPooler



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
    "lm_lr": 0,
    "lm_max_length": 512,
    "lm_weight_decay": 1e-4,
    "lm_padding": True,
    "lm_truncation": True,
    "lm_requires_grad": True,
    "pooler_hidden_size": 768, 
    "pooler_dropout": 0.5,
    "pooler_hidden_act": 'relu',

    "num_nodes": 2708,
    "num_node_features": 1024,
    "gnn_h_feats": 1024,
    "gnn_lr": 0.0005,
    "gnn_weight_decay": 0,
    "gnn_dropout": 0.5,
    "gnn_requires_grad": True,
    "gnn_num_layers":2,
    "gnn_use_residual": True,

    "once_batch_size": 1024,
    "once_shuffle": True,
    "once_drop_last": True,

    "train_batch_size": 128,
    "train_shuffle": True,
    "train_drop_last": True,

    "valid_batch_size": 128,
    "valid_shuffle": True,
    "valid_drop_last": True,

    "test_batch_size": 128,
    "test_shuffle": True,
    "test_drop_last": True,

    "use_param_free_pooler": True,
    "leading_alpha": 0.9,
    "use_node_cache": True,
    "node_cache": "../data/tmp/bypass_llama7b_cora/warm_emb.pth",

    "log_dir": "log/exp-name", 
    "save_interval": 5,
    "save_latest": True,
    "resume": False,
}
config = dict_to_namespace(config)

writer, saver, loader = save_exp(config)

seed(config.seed)

graph, num_classes, text = load_data('cora', use_dgl=True, use_text=True)
# graph.ndata['x'] = torch.load('arxiv_deberta.pt').squeeze()
graph = dgl.to_bidirected(graph, copy_ndata=True)
graph = dgl.remove_self_loop(graph)
graph = dgl.add_self_loop(graph)

lm = CroppedLlama2.from_pretrained('/home/ubuntu/data/models/Llama-2-7b-hf/').to(torch.half).cuda()
lm.post_init_crop(23)
lm_input = torch.load('../data/tmp/bypass_llama7b_cora/cache_layer_23.pth').to(torch.half)
model = graphsage(num_layers=config.gnn_num_layers, num_nodes=config.num_nodes, in_feats=config.num_node_features, h_feats=config.gnn_h_feats, num_classes=num_classes, dropout=config.gnn_dropout, alpha=config.leading_alpha, use_residual=config.gnn_use_residual).cuda()

if config.resume:
    t = loader(model, lm, 'latest')
    print('resume checkpoint models from latest')

for param in lm.parameters():
    param.requires_grad = config.lm_requires_grad
for param in model.parameters():
    param.requires_grad = config.gnn_requires_grad

if config.use_node_cache:
    node_cache = torch.load(config.node_cache, map_location='cpu')
    model.load_history(node_cache)

opt_group = []
if config.lm_lr > 0.:
    opt_group.append({'params': lm.parameters(), 'lr': config.lm_lr, "weight_decay": config.lm_weight_decay})
    LM_USE_NO_GRAD = False
else:
    LM_USE_NO_GRAD = True

if config.gnn_lr > 0:
    opt_group.append({'params': model.parameters(), 'lr': config.gnn_lr, "weight_decay": config.gnn_weight_decay})
    GNN_USE_NO_GRAD = False
else:
    GNN_USE_NO_GRAD = True

assert len(opt_group) > 0, f'no learnable param found'
opt = torch.optim.Adam(opt_group)

train_dataloader, valid_dataloader, test_dataloader = init_dataloader(graph, 'train', config), init_dataloader(graph, 'val', config), init_dataloader(graph, 'test', config)

best_val_accuracy = 0.
#best_model_path = 'best_deberta_pretrained_graphsage_model.pt'
#best_lm_path = 'best_deberta_pretrained_graphsage_lm.pt'

for epoch in range(100):
    model.train()

    with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
            # print(output_nodes)
            # inputs = [text[i] for i in output_nodes]
            labels = mfgs[-1].dstdata['y']
            # with torch.no_grad():
            if LM_USE_NO_GRAD:
                with torch.no_grad():
                    idx = torch.tensor([i for i in output_nodes])
                    lm_inputs = lm_input[idx].cuda()
                    inputs = lm(inputs_embeds=lm_inputs, use_cache=False)
            else:
                idx = torch.tensor([i for i in output_nodes])
                lm_inputs = lm_input[idx].cuda()
                inputs = lm(inputs_embeds=lm_inputs, use_cache=False)
            # inputs = mfgs[0].srcdata['x']
            if GNN_USE_NO_GRAD:
                with torch.no_grad():
                    predictions = model(mfgs=mfgs, x=inputs, batch_size=config.train_batch_size)
            else:
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
            writer.add_scalar('train/loss', loss.item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('train/accuracy', accuracy, epoch * len(train_dataloader) + step)

            del input_nodes, output_nodes, mfgs, inputs, labels, predictions, loss
            torch.cuda.empty_cache()
    if config.save_interval > 0 and epoch % config.save_interval == 0:
        saver(model, lm, f'epoch_{epoch}')
    if config.save_latest:
        saver(model, lm, 'latest')
    model.eval()

    predictions = []
    labels = []
    with torch.no_grad() and tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:

            # with torch.no_grad():
            
            idx = torch.tensor([i for i in output_nodes])
            lm_inputs = lm_input[idx].cuda()
            inputs = lm(inputs_embeds=lm_inputs, use_cache=False)
            # inputs = mfgs[0].srcdata['x']
            labels.append(mfgs[-1].dstdata['y'].cpu().numpy())
            # predictions.append(model(mfgs=mfgs, x=inputs, batch_size=config.valid_batch_size).argmax(1).cpu().numpy())
            predictions.append(model(mfgs=mfgs, x=inputs, batch_size=config.valid_batch_size).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        val_accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        writer.add_scalar('valid/accuracy', val_accuracy, epoch)
        if best_val_accuracy <= val_accuracy:
            best_val_accuracy = val_accuracy
            saver(model, lm, 'best')

    predictions = []
    labels = []
    with torch.no_grad() and tqdm.tqdm(test_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            idx = torch.tensor([i for i in output_nodes])
            lm_inputs = lm_input[idx].cuda()
            inputs = lm(inputs_embeds=lm_inputs, use_cache=False)
            # inputs = mfgs[0].srcdata['x']
            labels.append(mfgs[-1].dstdata['y'].cpu().numpy())
            # inputs = lm(inputs).to(device)
            predictions.append(model(mfgs=mfgs, x=inputs, batch_size=config.test_batch_size).argmax(1).cpu().numpy())
            # predictions.append(model(mfgs=mfgs, x=inputs, batch_size=config.test_batch_size).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        # print(predictions)
        labels = np.concatenate(labels)
        test_accuracy = sklearn.metrics.accuracy_score(labels, predictions)

        print('Epoch {} Valid Accuracy {}  Best Accuracy {} Test Accuracy {}'.format(epoch, val_accuracy, best_val_accuracy, test_accuracy))
        writer.add_scalar('test/accuracy', test_accuracy, epoch)

