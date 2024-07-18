from deepspeed.runtime.pipe.engine import PipelineEngine
import torch.distributed as dist
from p2g.config import P2GConfig
import torch
from p2g.models.base import P2GModel
from p2g.models.gnn.modeling import GNNLoss

from deepspeed.utils import logger


def load_model(engine: PipelineEngine, config: P2GConfig):
    is_last_stage = engine.local_rank == engine.world_size - 1
    pass_load = False
    if config.llm_not_exist or config.gnn_not_exist or not config.resume:
        logger.info(
            "No pre-trained lora model to load, finetune from scratch, but still try to load warm embbeding for GNN"
        )
        pass_load = True
    flag = torch.tensor(int(pass_load), device=engine.device)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    if flag.item() > 0:
        if is_last_stage:
            if isinstance(engine.module.loss_fn, GNNLoss):
                engine.module.loss_fn.load_warm_emb()
                logger.info("Warm embedding loaded")
        return
    for layer in engine.module.forward_funcs:
        if isinstance(layer, P2GModel):
            layer.load_peft()
    if is_last_stage:
        if isinstance(engine.module.loss_fn, GNNLoss):
            engine.module.loss_fn.load_ckpt()
    logger.info("Pre-trained model loaded")
