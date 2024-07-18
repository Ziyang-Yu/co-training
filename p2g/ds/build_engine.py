import deepspeed
import yaml
from deepspeed.pipe import LayerSpec, PipelineModule, TiedLayerSpec
from deepspeed.runtime.pipe.topology import ProcessTopology
from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.layer import dispatch_default

from p2g.config import P2GConfig
from p2g.models import P2GModel, loss_registry, model_registry
from p2g.peft import build_lora

from .build_ds_config import build_ds_config

from deepspeed.utils import logger


def build_engine(config: P2GConfig, train_steps: int):
    specs = model_registry(config)
    ds_config = build_ds_config(config.ds_config_path)

    loss_fn = loss_registry(config)
    topo = ProcessTopology(**ds_config["model_topo"]["process_topology"])
    pipeline_model = PipelineModule(
        layers=specs,
        topology=topo,
        activation_checkpoint_interval=0,  # do not use activation checkpointing of deepspeed, which is of some bugs
        partition_method=ds_config["model_topo"]["parts"],
        loss_fn=loss_fn,
        eval_fn=loss_fn.forward_eval,
    )
    for name, layer in pipeline_model.named_modules():
        if isinstance(layer, P2GModel):
            layer.load_ckpt()
            logger.info(f"load ckpt name={name} type={type(layer)}")
    pipeline_model, trainable_params, ds_config = build_lora(pipeline_model, ds_config, train_steps)
    model_engine, optimizer, data_loader, lr_schdlr = deepspeed.initialize(
        model=pipeline_model,
        config=ds_config,
        model_parameters=trainable_params,
    )
    # print the optimizer's group parameters size
    groups = []
    for group in optimizer.param_groups:
        groups.append(sum(p.numel() for p in group["params"]))
    logger.info(f"Optimizer groups: {groups}")
    return model_engine
