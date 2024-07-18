import deepspeed


def build_dist_env():
    deepspeed.init_distributed(dist_backend="nccl")
