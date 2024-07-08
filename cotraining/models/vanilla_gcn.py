import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv

# Define a sample GCN network based on your provided code
class GCN(nn.Module):
    def __init__(self, num_layers, in_feats, h_feats, num_classes, dropout, use_residual=False, alpha=0.9, device='cpu'):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_feats, h_feats))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(h_feats))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(h_feats, h_feats))
            self.bns.append(nn.BatchNorm1d(h_feats))
        self.convs.append(GraphConv(h_feats, num_classes))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, mfgs, x):
        for conv, bn, mfg in zip(self.convs[:-1], self.bns, mfgs[:-1]):
            x = conv(mfg, x)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](mfgs[-1], x)
        return x.log_softmax(dim=-1)
