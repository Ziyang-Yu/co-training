from typing import List

import torch
from deepspeed.runtime.pipe.module import LayerSpec

from p2g.config import P2GConfig

from .base import P2GModel


def model_registry(config: P2GConfig):
    llm_model_type = config.llm_model_type
    if llm_model_type == "llama":
        from .llama.modeling import get_llama_model_specs

        return get_llama_model_specs(config)
    elif llm_model_type == "opt":
        from .opt.modeling import get_opt_specs

        return get_opt_specs(config)
    else:
        raise NotImplementedError(f"Model type {llm_model_type} not implemented")


def loss_registry(config: P2GConfig):
    from .gnn import GNNLoss

    return GNNLoss(config)


def specs_to_model(config: P2GConfig):
    model_specs: List[LayerSpec] = model_registry(config)
    model_layers = []
    for spec in model_specs:
        layer: P2GModel = spec.build()
        layer.load_ckpt()
        model_layers.append(layer)
    model = torch.nn.Sequential(*model_layers)
    loss_fn = loss_registry(config)
    loss_fn.load_warm_emb()
    return model, loss_fn
