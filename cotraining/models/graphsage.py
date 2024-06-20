from typing import Iterator, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn import SAGEConv

from cotraining.utils import History


class graphsage(nn.Module):
    def __init__(self, num_layers, num_nodes, in_feats, h_feats, num_classes, dropout, device='cpu'):
        super(graphsage, self).__init__()
        # self.num_layers = num_layers
        # self.layers = [SAGEConv(in_feats, h_feats, aggregator_type='mean') for _ in range(num_layers)]
        # self.history = [History(num_nodes, embedding_dim=in_feats, device=device)]
        # self.history += [History(num_nodes, embedding_dim=h_feats, device=device) for _ in range(num_layers - 1)]
        # self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        # self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        # self.h_feats = h_feats
        self.histories = torch.nn.ModuleList([History(num_nodes, embedding_dim=in_feats, device=device)])
        for _ in range(num_layers - 1):
            self.histories.append(History(num_nodes, embedding_dim=h_feats, device=device))
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_feats, h_feats, aggregator_type='mean'))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(h_feats))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(h_feats, h_feats, aggregator_type='mean'))
            self.bns.append(torch.nn.BatchNorm1d(h_feats))
        self.convs.append(SAGEConv(h_feats, num_classes, aggregator_type='mean'))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, mfgs, x, batch_size):
        self.push_and_pull(self.histories[0], x, batch_size, mfgs[0].srcdata['_ID'].cpu())
        for conv, bn, mfg, history in zip(self.convs[:-1], self.bns, mfgs[:-1], self.histories[1:]):

            x = conv(mfg, x)
            x = bn(x)
            x = F.relu(x)
            x = self.push_and_pull(history, x, batch_size, mfg.dstdata['_ID'].cpu())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](mfgs[-1], x)
        return x.log_softmax(dim=-1)


    # def parameters(self):
    #     return list(self.conv1.parameters()) + list(self.conv2.parameters())


    def push_and_pull(self, history: History, x: torch.Tensor,
                      batch_size: Optional[int] = None,
                      n_id: Optional[torch.Tensor] = None,
                      offset: Optional[torch.Tensor] = None,
                      count: Optional[torch.Tensor] = None) -> torch.Tensor:
        history.push(x[:batch_size], n_id[:batch_size], offset, count)
        h = history.pull(n_id[batch_size:]).to(x.device)
        return torch.cat([x[:batch_size], h], dim=0)

    # @torch.no_grad()
    # def forward_once(self, mfgs) -> None:
    #     # x = mfgs[0].srcdata['x']
    #     # self.history[0].push(x, mfgs[0].srcdata['_ID'].cpu())
    #     # h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
    #     # # print("h_dst.shape", h_dst.shape)
    #     # h = self.conv1(mfgs[0], (x, h_dst))  # <---
    #     # h = F.relu(h)
    #     # self.history[1].push(h, mfgs[1].srcdata['_ID'].cpu())
    #     x = mfgs[0].srcdata['x']
    #     for i in range(self.num_layers-1):
    #         self.push_and_pull(self.history[i], x, mfgs[i].num_src_nodes(), mfgs[i].srcdata['_ID'].cpu())
    #         h_dst = x[:mfgs[i].num_dst_nodes()]
    #         h = self.layers[i](mfgs[i], (x, h_dst))
    #         h = F.relu(h)
    #         x = h
    #     self.push_and_pull(self.history[-1], h, mfgs[-1].num_src_nodes(), mfgs[-1].srcdata['_ID'].cpu())
    #     return h


    # def forward(self, mfgs, x, batch_size) -> torch.Tensor:
    #     # Lines that are changed are marked with an arrow: "<---"
    #     # print("x.shape: ", x.shape)
    #     x = self.push_and_pull(self.history[0], x, batch_size, mfgs[0].srcdata['_ID'].cpu())
    #     # print("x.shape: ", x.shape)
    #     h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
    #     # print("h_dst.shape", h_dst.shape)
    #     h = self.conv1(mfgs[0], (x, h_dst))  # <---
    #     h = F.relu(h)
    #     h = self.push_and_pull(self.history[1], h, batch_size, mfgs[1].srcdata['_ID'].cpu())
    #     h_dst = h[:mfgs[1].num_dst_nodes()]  # <---
    #     h = self.conv2(mfgs[1], (h, h_dst))  # <---
    #     return h