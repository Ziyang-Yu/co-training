import os
from dataclasses import dataclass

import torch
import yaml
from peft.tuners.lora.config import LoraConfig
from transformers import AutoConfig


def _load_lora_config(config):
    # rank, alpha, dropout
    config = LoraConfig(
        r=config.llm_lora_rank,
        lora_alpha=config.llm_lora_alpha,
        lora_dropout=config.llm_lora_dropout,
    )
    return config


@dataclass
class P2GConfig:
    resume: bool

    llm_cfg_path: str
    llm_dtype_name: str
    llm_ckpt_path: str
    llm_model_type: str
    llm_lora_rank: int
    llm_lora_alpha: int
    llm_lora_dropout: float
    llm_lora_ckpt_save_path: str
    llm_use_checkpoint: bool

    ds_config_path: str

    gnn_loss_type: str
    gnn_num_nodes: int
    gnn_node_features: int
    gnn_layers: int
    gnn_use_leading: bool  # whether tie the history residual embedding to the graph
    gnn_leading_alpha: float
    gnn_dtype_name: str
    gnn_dropout: float
    gnn_neighbor_num: int
    gnn_ckpt_path: str
    gnn_ckpt_save_path: str

    dataset_name: str
    dataset_path: str
    dataset_tokenized_ids_cache: str
    dataset_padding_width: int
    dataloader_type: str

    exp_save_dir: str

    # def a post
    def __post_init__(self):
        self.llm_dtype = torch.float16 if self.llm_dtype_name == "float16" else torch.float32
        self.llm_lora_config = _load_lora_config(self)

        llm_config = AutoConfig.from_pretrained(self.llm_cfg_path)
        self.gnn_hidden_dim = llm_config.hidden_size
        if self.gnn_use_leading:
            assert self.gnn_node_features == 0, "when using leading, gnn_node_features should be 0 (unset)"
            self.gnn_node_features = llm_config.hidden_size
        self.gnn_dtype = torch.float16 if self.gnn_dtype_name == "float16" else torch.float32

        # check if the ckpt path exists
        if not os.path.exists(self.llm_lora_ckpt_save_path):
            os.makedirs(self.llm_lora_ckpt_save_path)
            self.llm_not_exist = True
        else:
            self.llm_not_exist = False
        if not os.path.exists(self.gnn_ckpt_save_path):
            os.makedirs(self.gnn_ckpt_save_path)
            self.gnn_not_exist = True
        else:
            self.gnn_not_exist = False


def build_p2g_config(config_path: str):
    with open(config_path, "r") as f:
        return P2GConfig(**yaml.safe_load(f))
