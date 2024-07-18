from typing import Optional

import torch
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv
from torch.nn import Module

from p2g.config import P2GConfig


class History(Module):
    def __init__(self, config: P2GConfig):
        super().__init__()

        self.num_embeddings = config.gnn_num_nodes
        self.embedding_dim = config.gnn_node_features
        assert self.num_embeddings > 0
        assert self.embedding_dim > 0
        dtype = config.gnn_dtype
        self.emb = torch.zeros(
            self.num_embeddings,
            self.embedding_dim,
            device="cpu",
            pin_memory=True,
            dtype=dtype,
            requires_grad=False,
        )
        self._device = "cpu"

    def reset_parameters(self):
        self.emb.fill_(0)

    @torch.no_grad()
    def pull(self, n_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.emb
        if n_id is not None:
            assert n_id.device == self.emb.device
            out = out.index_select(0, n_id)
        return out.to(device=self._device)

    @torch.no_grad()
    def push(
        self,
        x,
        n_id: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
    ):
        if n_id is None and x.size(0) != self.num_embeddings:
            raise ValueError
        elif n_id is None and x.size(0) == self.num_embeddings:
            self.emb.copy_(x)
        elif offset is None or count is None:
            self.emb[n_id] = x.to(self.emb.device).detach()
        else:  # Push in chunks:
            raise NotImplementedError

    def push_and_pull(
        self,
        x: torch.Tensor,
        batch_size: int,
        n_id: torch.Tensor,
    ) -> torch.Tensor:
        self.push(x[:batch_size], n_id[:batch_size])
        h = self.pull(n_id[batch_size:]).to(x.device)
        return torch.cat([x[:batch_size], h], dim=0)

    def pull_only(self, x: torch.Tensor, history, n_id: torch.Tensor) -> torch.Tensor:
        return history.pull(n_id)


class GCN(Module):
    def __init__(self, config: P2GConfig):
        super(GCN, self).__init__()
        in_feats = config.gnn_node_features
        h_feats = config.gnn_hidden_dim
        num_classes = config.gnn_num_nodes
        num_layers = config.gnn_layers
        dropout = config.gnn_dropout
        use_leading = config.gnn_use_leading
        alpha = config.gnn_leading_alpha
        self.history = History(config)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(in_feats, h_feats))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(h_feats))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(h_feats, h_feats))
            self.bns.append(torch.nn.BatchNorm1d(h_feats))
        self.convs.append(GraphConv(h_feats, num_classes))
        self.dropout = dropout
        self.use_leading = use_leading
        self.alpha = alpha

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, mfgs, x):
        batch_size = mfgs[-1].dstdata["_ID"].shape[0]
        x = self.history.push_and_pull(x, batch_size, mfgs[0].srcdata["_ID"].cpu())
        for conv, bn, mfg in zip(self.convs[:-1], self.bns, mfgs[:-1]):
            x = conv(mfg, x)
            x = bn(x)
            x = F.relu(x)
            if self.use_leading:
                node_emb = self.history.pull_only(x, self.history, mfg.dstdata["_ID"].cpu()).to(x.device)
                x = (1 - self.alpha) * x + self.alpha * node_emb
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](mfgs[-1], x)
        return x.log_softmax(dim=-1)


class GraphSage(Module):
    def __init__(self, config: P2GConfig):
        super(GraphSage, self).__init__()
        in_feats = config.gnn_node_features
        h_feats = config.gnn_hidden_dim
        num_classes = config.gnn_num_nodes
        num_layers = config.gnn_layers
        dropout = config.gnn_dropout
        use_leading = config.gnn_use_leading
        self.alpha = config.gnn_leading_alpha
        self.history = History(config)
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_feats, h_feats, aggregator_type="mean"))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(h_feats))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(h_feats, h_feats, aggregator_type="mean"))
            self.bns.append(torch.nn.BatchNorm1d(h_feats))
        self.convs.append(SAGEConv(h_feats, num_classes, aggregator_type="mean"))
        self.dropout = dropout
        self.use_leading = use_leading

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, mfgs, x):
        batch_size = mfgs[-1].dstdata["_ID"].shape[0]
        x = self.history.push_and_pull(x, batch_size, mfgs[0].srcdata["_ID"].cpu())
        for conv, bn, mfg in zip(self.convs[:-1], self.bns, mfgs[:-1]):
            x = conv(mfg, x)
            x = bn(x)
            x = F.relu(x)
            if self.use_leading:
                node_emb = self.history.pull_only(x, self.history, mfg.dstdata["_ID"].cpu())
                x = (1 - self.alpha) * x + self.alpha * node_emb
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](mfgs[-1], x)
        return x.log_softmax(dim=-1)


class NonParamPooler(Module):
    def __init__(self, config: P2GConfig):
        super().__init__()
        self.output_dim = config.gnn_node_features

    def forward(self, hidden_states):
        context_token = hidden_states[:, 0]
        if context_token.shape[1] == self.output_dim:
            return context_token
        elif context_token.shape[1] > self.output_dim:
            context_token = context_token[:, : self.output_dim]
        else:
            raise ValueError(f"the hidden states shape {context_token.shape} is less than gnn feature dim.")
        return context_token


class GNNLoss(Module):
    def __init__(self, config: P2GConfig):
        super().__init__()
        self.config = config
        self.pooler = NonParamPooler(config)
        if config.gnn_loss_type == "gcn":
            self.module = GCN(config)
        elif config.gnn_loss_type == "graphsage":
            self.module = GraphSage(config)
        else:
            raise NotImplementedError
        self._special_name = "gnn"
        self._special_emb = "warm_emb"

    def save_ckpt(self):
        name2params = dict(self.module.named_parameters())
        torch.save(
            name2params,
            self.config.gnn_ckpt_save_path + f"/{self._special_name}.pth",
        )

    def load_ckpt(self, path=None):
        name2params = torch.load(self.config.gnn_ckpt_save_path + f"/{self._special_name}.pth")
        self.module.load_state_dict(name2params, strict=False)

    def load_warm_emb(self):
        warmemb = torch.load(self.config.gnn_ckpt_path + f"/{self._special_emb}.pth")
        self.module.history.emb.copy_(warmemb)

    def forward(self, inputs, labels):
        logits = inputs
        input_nodes, output_nodes, mfgs = labels
        x = self.pooler(logits)
        pred = self.module(mfgs, x)
        labels = mfgs[-1].dstdata["y"]
        labels = torch.flatten(labels)
        labels = labels.to(pred.device)
        loss = F.cross_entropy(pred, labels)
        return loss

    def forward_eval(self, inputs, labels):
        logits = inputs
        input_nodes, output_nodes, mfgs = labels
        x = self.pooler(logits)
        pred = self.module(mfgs, x)
        pred_labels = torch.argmax(pred, dim=-1)
        labels = mfgs[-1].dstdata["y"]
        labels = torch.flatten(labels)
        labels = labels.to(pred.device)
        return pred_labels == labels
