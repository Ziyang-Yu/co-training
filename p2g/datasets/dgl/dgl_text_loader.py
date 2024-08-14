import os

import dgl
import torch
from deepspeed.accelerator.real_accelerator import get_accelerator
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.utils import logger
from dgl import set_dst_lazy_features, set_edge_lazy_features, set_src_lazy_features
from dgl.dataloading import NeighborSampler
from dgl.dataloading.dataloader import DataLoader as DGLDataLoader
from torch_geometric.utils import to_dgl
from transformers import AutoTokenizer

from p2g.config import P2GConfig
from p2g.ds.utils import get_local_rank


class TextDGLSampler(NeighborSampler):
    def __init__(self, text_list: list, config: P2GConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_list = text_list
        self.input_ids = []
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_cfg_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.padding_width = config.dataset_padding_width
        self._tokenize_all(config.dataset_tokenized_ids_cache)

    def _tokenize_all(self, cache_path):
        if os.path.exists(cache_path):
            self.input_ids = torch.load(cache_path)
            return
        else:
            input = self.tokenizer(
                self.text_list,
                padding=True,
                truncation=True,
                max_length=self.padding_width,
                return_tensors="pt",
            )
            self.input_ids = input["input_ids"]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.input_ids, cache_path)

    def sample_blocks(self, *args, **kwargs):
        input_nodes, output_nodes, blocks = super().sample_blocks(*args, **kwargs)
        input_ids = torch.cat([self.input_ids[i] for i in output_nodes], dim=0).reshape(-1, self.padding_width)
        return input_ids, (input_nodes, output_nodes, blocks)

    def assign_lazy_features(self, result):
        input_ids, (input_nodes, output_nodes, blocks) = result
        set_src_lazy_features(blocks[0], self.prefetch_node_feats)
        set_dst_lazy_features(blocks[-1], self.prefetch_labels)
        for block in blocks:
            set_edge_lazy_features(block, self.prefetch_edge_feats)
        return input_ids, (input_nodes, output_nodes, blocks)


class TextDGLDataloader(DGLDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def assign_lazy_features(self, result):
        input_ids, (input_nodes, output_nodes, blocks) = result
        set_src_lazy_features(blocks[0], self.prefetch_node_feats)
        set_dst_lazy_features(blocks[-1], self.prefetch_labels)
        for block in blocks:
            set_edge_lazy_features(block, self.prefetch_edge_feats)
        return input_ids, (input_nodes, output_nodes, blocks)


def load_data(config: P2GConfig, seed=0):
    dataset = config.dataset_name
    dataset_path = config.dataset_path
    if dataset == "cora":
        from .cora import get_raw_text_cora as get_raw_text

        num_classes = 7
    elif dataset == "ogbn-arxiv":
        from .arxiv import get_raw_text_arxiv as get_raw_text

        num_classes = 40
    else:
        raise ValueError(f"Error: Dataset {dataset} not supported")
    data, text = get_raw_text(dataset_path, use_text=True, seed=seed)
    data = to_dgl(data)
    return data, num_classes, text


def load_dgl_dataset(config: P2GConfig):
    graph, num_classes, text = load_data(config)
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    return graph, num_classes, text


def init_dataloader(graph, text, name, ds_config: dict, config: P2GConfig, device: torch.device):
    sampler = TextDGLSampler(text, config, [config.gnn_neighbor_num for _ in range(config.gnn_layers)])

    train_mask = graph.ndata["train_mask"]
    train_nids = torch.nonzero(train_mask, as_tuple=False).squeeze()
    val_mask = graph.ndata["val_mask"]
    val_nids = torch.nonzero(val_mask, as_tuple=False).squeeze()
    test_mask = graph.ndata["test_mask"]
    test_nids = torch.nonzero(test_mask, as_tuple=False).squeeze()
    if name == "train":
        train_loader = TextDGLDataloader(
            graph,
            train_nids,
            sampler,
            batch_size=ds_config["train_micro_batch_size_per_gpu"],
            device=device,
            shuffle=ds_config["train_shuffle"],
            drop_last=ds_config["train_drop_last"],
            num_workers=ds_config["num_workers"],
        )
        return train_loader
    elif name == "val":

        val_loader = dgl.dataloading.DataLoader(
            graph,
            val_nids,
            sampler,
            device=device,
            batch_size=ds_config["valid_batch_size"],
            shuffle=ds_config["valid_shuffle"],
            drop_last=ds_config["valid_drop_last"],
            num_workers=ds_config["num_workers"],
        )
        return val_loader
    elif name == "test":
        test_mask = graph.ndata["test_mask"]
        test_nids = torch.nonzero(test_mask, as_tuple=False).squeeze()
        test_dataloader = dgl.dataloading.DataLoader(
            graph,
            test_nids,
            sampler,
            device=device,
            batch_size=ds_config["test_batch_size"],
            shuffle=ds_config["test_shuffle"],
            drop_last=ds_config["test_drop_last"],
            num_workers=ds_config["num_workers"],
        )
        return test_dataloader
    else:
        raise NotImplementedError


def build_dataloader(ds_config: dict, config: P2GConfig):
    graph, num_classes, text = load_dgl_dataset(config)
    device_rank = get_local_rank()
    device = torch.device(device_rank)
    logger.info(f"local rank device = {device}")
    train_loader = init_dataloader(graph, text, "train", ds_config, config, device)
    val_loader = init_dataloader(graph, text, "val", ds_config, config, device)
    test_loader = init_dataloader(graph, text, "test", ds_config, config, device)
    logger.info(f"Train loader length: {len(train_loader.dataset)}")
    logger.info(f"Val loader length: {len(val_loader.dataset)}")
    logger.info(f"Test loader length: {len(test_loader.dataset)}")
    return train_loader, val_loader, test_loader
