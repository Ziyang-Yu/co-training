from typing import Iterator, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn import SAGEConv

from cotraining.utils import History


class graphsage(nn.Module):
    def __init__(self, num_nodes, in_feats, h_feats, num_classes, dropout, device='cpu'):
        super(graphsage, self).__init__()
        self.history = [History(num_nodes, embedding_dim=in_feats, device=device), History(num_nodes, embedding_dim=h_feats, device=device)]
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.h_feats = h_feats

    def parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters())


    def push_and_pull(self, history: History, x: torch.Tensor,
                      batch_size: Optional[int] = None,
                      n_id: Optional[torch.Tensor] = None,
                      offset: Optional[torch.Tensor] = None,
                      count: Optional[torch.Tensor] = None) -> torch.Tensor:
        # print("batch_size:", batch_size)
        history.push(x[:batch_size], n_id[:batch_size], offset, count)
        h = history.pull(n_id[batch_size:]).to(x.device)
        return torch.cat([x[:batch_size], h], dim=0)

    @torch.no_grad()
    def forward_once(self, mfgs) -> None:
        x = mfgs[0].srcdata['x']
        self.history[0].push(x, mfgs[0].srcdata['_ID'].cpu())
        h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        # print("h_dst.shape", h_dst.shape)
        h = self.conv1(mfgs[0], (x, h_dst))  # <---
        h = F.relu(h)
        self.history[1].push(h, mfgs[1].srcdata['_ID'].cpu())


    def forward(self, mfgs, x, batch_size) -> torch.Tensor:
        # Lines that are changed are marked with an arrow: "<---"
        # print("x.shape: ", x.shape)
        x = self.push_and_pull(self.history[0], x, batch_size, mfgs[0].srcdata['_ID'].cpu())
        # print("x.shape: ", x.shape)
        h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        # print("h_dst.shape", h_dst.shape)
        h = self.conv1(mfgs[0], (x, h_dst))  # <---
        h = F.relu(h)
        h = self.push_and_pull(self.history[1], h, batch_size, mfgs[1].srcdata['_ID'].cpu())
        h_dst = h[:mfgs[1].num_dst_nodes()]  # <---
        h = self.conv2(mfgs[1], (h, h_dst))  # <---
        return h