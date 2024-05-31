#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

from transformers import BertTokenizer, BertModel, AutoTokenizer, DebertaModel, AutoModel, PreTrainedModel

def seed(seed_val):
    import random
    import numpy as np
    import torch
    # import tensorflow as tf
    # tf.random.set_seed(seed_value)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
seed(42)

class deberta:

    def __init__(self):
        self.__name__ = 'microsoft/deberta-base'
        self.__num_node_features__ = 768 
        self.device = 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
# Load model directly
        self.model = AutoModel.from_pretrained("microsoft/deberta-base")
        # self.model = DebertaModel.from_pretrained("microsoft/deberta-base")
        
        # self.__output_dim__ = self.__model__.
    # @property
    def parameters(self):
        return self.model.parameters()

    @property
    def num_node_features(self):
        return 768

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self

    def forward(self, text):

        def model_forward_input(input):
            input = self.tokenizer(input, return_tensors='pt').to(self.device)
            output = self.model(**input).last_hidden_state.mean(dim=1)
            # print(output.shape)
            # return self.model(**input).last_hidden_state.mean(dim=1)
            # print(output.shape)
            return torch.squeeze(output)

        return torch.stack(list(map(model_forward_input, text)))

    def __call__(self, data):
        if isinstance(data, str):
            return self.forward([data])
        if isinstance(data, list):
            return self.forward(data)


# In[2]:


# device = torch.device(0)
device = 'cpu'


# In[3]:


lm = deberta().to(device)


# In[4]:


import dgl
import torch
import numpy as np
# from ogb.nodeproppred import DglNodePropPredDataset

from data import load_data

# dataset = dgl.data.CoraGraphDataset()
# # device = 'cuda'      # change to 'cuda' for GPU
# graph = dataset[0]
graph, num_classes, text = load_data('ogbn-arxiv', use_dgl=True, use_text=True)
# print(graph.ndata)
# print(type(graph))
# graph = (dataset)


# get text feat
# dataset, num_classes, text = load_data('cora', use_dgl=True, use_text=True)
# features = []
# with torch.no_grad():
#     for i, t in enumerate(text):
#         if i % 1000 == 0:
#             print(i)
#         features.append(lm(t).squeeze().cpu())
# features = torch.stack(features)
# torch.save(features, 'arxiv.pt')


# In[5]:


import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

from data_utils import load_data

# dataset = dgl.data.CoraGraphDataset()
# # device = 'cuda'      # change to 'cuda' for GPU
# graph = dataset[0]
graph, num_classes, text = load_data('ogbn-arxiv', use_dgl=True, use_text=True)
# print(graph.ndata)
# print(type(graph))
# graph = (dataset)


# get text feat
# dataset, num_classes, text = load_data('cora', use_dgl=True, use_text=True)
# features = []
# with torch.no_grad():
#     for t in text:
#         features.append(lm(t).squeeze().cpu())
# features = torch.stack(features)
# torch.save(features, 'arxiv.pt')
# print("graph.ndata['feat'].shape", graph.ndata['feat'].shape)
# print("features.shape", features.shape)
features = torch.load('dataset/arxiv.pt')
graph.ndata['x'] = features

train_mask = graph.ndata['train_mask']
# train_mask = graph.train_mask
train_nids = torch.nonzero(train_mask, as_tuple=False).squeeze()
val_mask = graph.ndata['val_mask']
# val_mask = graph.val_mask
val_nids = torch.nonzero(val_mask, as_tuple=False).squeeze()
test_mask = graph.ndata['test_mask']
# test_mask = graph.test_mask
test_nids = torch.nonzero(test_mask, as_tuple=False).squeeze()

sampler = dgl.dataloading.NeighborSampler([4, 4])

train_dataloader = dgl.dataloading.DataLoader(
    # The following arguments are specific to DGL's DataLoader.
    graph,              # The graph
    train_nids,         # The node IDs to iterate over in minibatches
    sampler,            # The neighbor sampler
    device=device,      # Put the sampled MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=8,    # Batch size
    shuffle=True,       # Whether to shuffle the nodes for every epoch
    drop_last=True,    # Whether to drop the last incomplete batch
    num_workers=0       # Number of sampler processes
)


# In[6]:


import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        # Lines that are changed are marked with an arrow: "<---"
        
        h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        h = self.conv1(mfgs[0], (x, h_dst))  # <---
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]  # <---
        h = self.conv2(mfgs[1], (h, h_dst))  # <---
        return h

model = Model(in_feats=lm.__num_node_features__, h_feats=64, num_classes=num_classes).to(device)
# model = Model(in_feats=1433, h_feats=64, num_classes=7).to(device)


# In[7]:


# opt = torch.optim.Adam(list(model.parameters())+list(lm.parameters())) # 
opt = torch.optim.Adam([
    # {'params': lm.parameters(), 'lr': 1e-4},
    {'params': model.parameters(), 'lr': 0.01}])
valid_dataloader = dgl.dataloading.DataLoader(
    graph, val_nids, sampler,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
)


# In[8]:


import tqdm
import sklearn.metrics

best_accuracy = 0
best_model_path = 'model.pt'

dataset, num_classes, text = load_data('cora', use_dgl=True, use_text=True)

for epoch in range(100):
    model.train()

    with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
            # print(111111)
            # print(mfgs[0].srcdata)
            inputs = mfgs[0].srcdata['x']
            # print(inputs.shape, input_nodes.shape)
            # inputs = [text[i] for i in input_nodes]
            # inputs = lm(inputs).to(device)
            # mfgs = mfgs.to(device)
            labels = mfgs[-1].dstdata['y']
            # print(inputs.shape, labels.shape)
            # print(len(labels), inputs.shape, input_nodes.shape, output_nodes.shape)    
            # break        
            predictions = model(mfgs, inputs)
            labels = torch.flatten(labels)
            # print(predictions.shape, labels.shape)
            print(labels)
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

    model.eval()

    predictions = []
    labels = []
    with torch.no_grad() and tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            # inputs = [text[i] for i in input_nodes]
            # print(type(mfgs[0]))
            inputs = mfgs[0].srcdata['x']
            labels.append(mfgs[-1].dstdata['y'].cpu().numpy())
            # inputs = lm(inputs).to(device)
            predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        # print(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        # print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
        #     inputs = mfgs[0].srcdata['feat']
        #     labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
        #     predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
        # predictions = np.concatenate(predictions)
        # labels = np.concatenate(labels)
        # accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        print('Epoch {} Validation Accuracy {} Best Accuracy {}'.format(epoch, accuracy, best_accuracy))
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)

        # Note that this tutorial do not train the whole model to the end.
        # break

