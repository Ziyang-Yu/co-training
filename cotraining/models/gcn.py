from typing import Iterator, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn import GraphConv

from cotraining.utils import History


class GCN(nn.Module):
    def __init__(self, num_layers, num_nodes, in_feats, h_feats, num_classes, dropout, device='cpu'):
        super(GCN, self).__init__()
        self.histories = torch.nn.ModuleList([History(num_nodes, embedding_dim=in_feats, device=device)])
        for _ in range(num_layers - 1):
            self.histories.append(History(num_nodes, embedding_dim=h_feats, device=device))
        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(in_feats, h_feats))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(h_feats))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(h_feats, h_feats))
            self.bns.append(torch.nn.BatchNorm1d(h_feats))
        self.convs.append(GraphConv(h_feats, num_classes))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, mfgs, x, batch_size):
        x = self.push_and_pull(self.histories[0], x, batch_size, mfgs[0].srcdata['_ID'].cpu())
        for conv, bn, mfg, history in zip(self.convs[:-1], self.bns, mfgs[:-1], self.histories[1:]):

            x = conv(mfg, x)
            x = bn(x)
            x = F.relu(x)
            x = self.push_and_pull(history, x, batch_size, mfg.dstdata['_ID'].cpu())
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