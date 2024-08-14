import torch

try:
    import flash_attn
except ImportError:
    flash_attn = None


def make_mask(batch_size: int, seq_length: int, device: torch.device, dtype: torch.dtype):
    if flash_attn:
        return torch.ones(batch_size, seq_length, device=device, dtype=dtype)
    else:
        return torch.ones(batch_size, 1, seq_length, seq_length, device=device, dtype=dtype)
