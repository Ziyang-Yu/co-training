from argparse import Namespace
import os
import json
from typing import Callable
from typing_extensions import Tuple
import torch
import torch.utils.tensorboard.writer as tw

def save_exp(config: Namespace) -> Tuple[tw.SummaryWriter, Callable]:
    # turn config to a dict 
    config_dict = {k: v for k, v in config._get_kwargs()}
    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    json_path = os.path.join(log_dir, 'config.json')
    with open(json_path, 'w') as f:
        json.dump(config_dict, f)
    writer_dir = os.path.join(log_dir, 'writer')
    writer = tw.SummaryWriter(log_dir=writer_dir)
    ckpt_dir = os.path.join(log_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    def saver(model, lm, tag, LM_USE_NO_GRAD):
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{tag}_model.pt'))
        if not LM_USE_NO_GRAD:
            torch.save(lm.state_dict(), os.path.join(ckpt_dir, f'{tag}_lm.pt'))
    def loader(model, lm, tag):
        if not os.path.exists(os.path.join(ckpt_dir, f'{tag}_model.pt')):
            return
        if not os.path.exists(os.path.join(ckpt_dir, f'{tag}_lm.pt')):
            return
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, f'{tag}_model.pt')))
        lm.load_state_dict(torch.load(os.path.join(ckpt_dir, f'{tag}_lm.pt')))
    return writer, saver, loader

