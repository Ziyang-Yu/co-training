from argparse import Namespace
import os
from typing import Callable
from typing_extensions import Tuple
import torch
import torch.utils.tensorboard.writer as tw

def save_exp(config: Namespace) -> Tuple[tw.SummaryWriter, Callable]:
    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    writer_dir = os.path.join(log_dir, 'writer')
    writer = tw.SummaryWriter(log_dir=writer_dir)
    ckpt_dir = os.path.join(log_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    def saver(model, lm, tag):
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{tag}_model.pt'))
        torch.save(lm.state_dict(), os.path.join(ckpt_dir, f'{tag}_lm.pt'))
    def loader(model, lm, tag):
        if not os.path.exists(os.path.join(ckpt_dir, f'{tag}_model.pt')):
            return
        if not os.path.exists(os.path.join(ckpt_dir, f'{tag}_lm.pt')):
            return
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, f'{tag}_model.pt')))
        lm.load_state_dict(torch.load(os.path.join(ckpt_dir, f'{tag}_lm.pt')))
    return writer, saver, loader

