import torch
from copy import deepcopy

from deepspeed.pipe import LayerSpec, PipelineModule, TiedLayerSpec
from deepspeed.utils import logger
from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.layer import Embedding as LoraEmbedding
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.layer import dispatch_default
from torch.nn import Embedding, Linear, Module

from p2g.models.base import P2GModel

LORA_LAYER_TYPES = (Embedding, Linear)
LORA_TARGET_TYPES = (LoraEmbedding, LoraLinear)


def recursive_setattr(model, module_name, module):
    split_list = module_name.split(".")
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)


def turn_to_lora(module: Module, lora_config: LoraConfig):
    logger.info(f"Prepare {module} to lora")
    if lora_config is None:
        return module
    replace_name2module = {}
    for name, mod in module.named_modules():
        if isinstance(mod, LORA_LAYER_TYPES):
            replace_name2module[name] = mod
    for key, val in replace_name2module.items():
        newval = dispatch_default(val, "lora", lora_config, **lora_config.to_dict())
        if key == "":
            module = newval
            break
        recursive_setattr(module, key, newval)
    for name, param in module.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    logger.info(f"Turned {module} to lora")
    return module


def _get_trainable_params(module: Module, is_last_stage: bool):
    name2params = {}
    param_cnt = 0
    llm_group = []
    gnn_group = []
    for name, param in module.named_parameters():
        if not is_last_stage:
            if "loss_fn" in name:
                continue
        if param.requires_grad:
            name2params[name] = param
            param_cnt += param.numel()
            if "loss_fn" in name:
                gnn_group.append(name)
            else:
                llm_group.append(name)
    logger.info(f"The module is {module}")
    logger.info(f"LLM Trainable params: {llm_group}")
    logger.info(f"GNN Trainable params: {gnn_group}")
    logger.info(f"Trainable param count: {param_cnt / 1e6} M")
    llm_group = [name2params[name] for name in llm_group]
    gnn_group = [name2params[name] for name in gnn_group]
    return llm_group, gnn_group


def _complete_optmizer_config(ds_config, train_steps):
    optimizer_config = ds_config["optimizer"]
    assert optimizer_config["type"] in ["SGD", "Adam"]
    optimizer_kwargs = optimizer_config.get("params", {})

    llm_group_config = ds_config["optimizer_groups"]["llm"]
    gnn_group_config = ds_config["optimizer_groups"]["gnn"]
    llm_kwargs = deepcopy(optimizer_kwargs)
    gnn_kwargs = deepcopy(optimizer_kwargs)
    for key, val in llm_group_config.items():
        llm_kwargs[key] = val
    for key, val in gnn_group_config.items():
        gnn_kwargs[key] = val

    scheduler_config = ds_config["scheduler"]
    assert scheduler_config["type"] in ["WarmupCosineLR"]
    if scheduler_config["type"] == "WarmupCosineLR":
        scheduler_kwargs = scheduler_config["params"]
        steps = scheduler_kwargs["total_num_steps"]
        if steps == "NEED_CALCULATION":
            scheduler_kwargs["total_num_steps"] = train_steps
        ds_config["scheduler"]["params"] = scheduler_kwargs

    return llm_kwargs, gnn_kwargs, ds_config


def _get_optimizer_group(llm_group, gnn_group, ds_config, train_steps):
    llm_kwargs, gnn_kwargs, scheduler_config = _complete_optmizer_config(ds_config, train_steps)
    llm_kwargs["params"] = llm_group
    gnn_kwargs["params"] = gnn_group
    return llm_kwargs, gnn_kwargs, scheduler_config


def build_lora(pipeline_model: PipelineModule, ds_config: dict, train_steps: int):
    for name, module in pipeline_model.named_modules():
        if isinstance(module, P2GModel):
            logger.info(f"Prepare lora for {name}: {type(module)}")
            module.prepare_peft(turn_to_lora)
    is_last_stage = pipeline_model.stage_id == pipeline_model.num_stages - 1
    # trainable_params = _get_trainable_params(pipeline_model, is_last_stage)
    llm_group, gnn_group = _get_trainable_params(pipeline_model, is_last_stage)
    llm_kwargs, gnn_kwargs, ds_config = _get_optimizer_group(llm_group, gnn_group, ds_config, train_steps)
    return pipeline_model, [llm_kwargs, gnn_kwargs], ds_config
