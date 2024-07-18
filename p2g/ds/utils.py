import torch.distributed as dist


def get_local_rank():
    # from torch
    if not dist.is_available():
        return -1
    if not dist.is_initialized():
        return -1
    return dist.get_rank() % dist.get_world_size()
