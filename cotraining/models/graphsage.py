from typing import Iterator, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn import SAGEConv

from cotraining.utils import History


class graphsage(nn.Module):
    def __init__(self, num_layers, num_nodes, in_feats, h_feats, num_classes, dropout, alpha=0.9, device='cpu'):
        super(graphsage, self).__init__()
        self.history = History(num_nodes, embedding_dim=in_feats, device=device)
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_feats, h_feats, aggregator_type='mean'))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(h_feats))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(h_feats, h_feats, aggregator_type='mean'))
            self.bns.append(torch.nn.BatchNorm1d(h_feats))
        self.convs.append(SAGEConv(h_feats, num_classes, aggregator_type='mean'))

        self.dropout = dropout
        self.alpha = alpha

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, mfgs, x, batch_size):
        history = self.history
        node_emb = self.push_and_pull(history, x, batch_size, mfgs[0].srcdata['_ID'].cpu())
        x = node_emb
        for conv, bn, mfg in zip(self.convs[:-1], self.bns, mfgs[:-1] ):
            x = conv(mfg, x)
            x = bn(x)
            x = F.relu(x)
            node_emb = self.pull_only(x, history, mfg.dstdata['_ID'].cpu())
            x = (1 - self.alpha) * x + self.alpha * node_emb
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](mfgs[-1], x)
        return x.log_softmax(dim=-1)

    def push_and_pull(self, history: History, x: torch.Tensor,
                      batch_size: Optional[int] = None,
                      n_id: Optional[torch.Tensor] = None,
                      offset: Optional[torch.Tensor] = None,
                      count: Optional[torch.Tensor] = None) -> torch.Tensor:
        history.push(x[:batch_size], n_id[:batch_size], offset, count)
        h = history.pull(n_id[batch_size:]).to(x.device)
        return torch.cat([x[:batch_size], h], dim=0)
    
    def pull_only(self, x: torch.device, history: History, n_id: torch.Tensor):
        h = history.pull(n_id).to(x.device)
        return h
    
    def load_history(self, data):
        assert self.history.emb.shape == data.shape, f'need emb shape is {self.history.emb.shape} but given {data.shape}'
        self.history.emb.data.copy_(data)