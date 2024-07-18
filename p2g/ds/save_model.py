from deepspeed.runtime.pipe.engine import PipelineEngine

from p2g.config import P2GConfig
from p2g.models.base import P2GModel
from p2g.models.gnn.modeling import GNNLoss


def save_peft_filter(name2params):
    return {k: v for k, v in name2params.items() if "lora" in k}


def save_model(engine: PipelineEngine, config: P2GConfig):
    for layer in engine.module.forward_funcs:
        if isinstance(layer, P2GModel):
            layer.save_peft(save_peft_filter)
    is_last_stage = engine.local_rank == engine.world_size - 1
    if is_last_stage:
        if isinstance(engine.module.loss_fn, GNNLoss):
            engine.module.loss_fn.save_ckpt()
