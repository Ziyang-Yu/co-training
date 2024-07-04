
import torch
from torch.autograd.graph import saved_tensors_hooks


class save_on_cpu_async(saved_tensors_hooks):
    def __init__(self, pin_memory=False, device_type="cuda", device=None):
        if device == 'cpu':
            pack_to_cpu = lambda x: x 
            unpack_from_cpu = lambda x: x
            super().__init__(pack_to_cpu, unpack_from_cpu)
            return
        device_module = getattr(torch, device_type, torch.cuda)
        new_stream = torch.cuda.Stream(device=device)

        def pack_to_cpu(tensor):
            if not pin_memory:
                return (tensor.device, tensor.cpu(), None)
            # print(f'pack tensor {tensor.shape} {tensor.layout} ', flush=True)
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(device_module.is_available() and not tensor.is_sparse),
            )
            end_event = torch.cuda.Event()
            with torch.cuda.stream(new_stream):
                packed.copy_(tensor, non_blocking=True)
                end_event.record(new_stream)
            return (tensor.device, packed, end_event)

        def unpack_from_cpu(packed):
            device, tensor, end_event = packed
            # print(f'unpack tensor {tensor.shape} {tensor.layout} ', flush=True)
            if end_event: 
                end_event.synchronize()
            return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)


class save_on_cpu(saved_tensors_hooks):
    def __init__(self, pin_memory=False, device_type="cuda"):
        device_module = getattr(torch, device_type, torch.cuda)

        def pack_to_cpu(tensor):
            if not pin_memory:
                return (tensor.device, tensor.cpu())
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(device_module.is_available() and not tensor.is_sparse),
            )
            packed.copy_(tensor)
            return (tensor.device, packed)

        def unpack_from_cpu(packed):
            device, tensor = packed
            return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)


import logging

import torch


class TagInfo:
    """
    定义 tag 的相关信息

    Args:
        tag (str): Tag 名称
    """

    def __init__(self, tag: str):
        self.tag = tag
        # 该 Tag 是否为第一次 iteration 计算
        self.is_first_iter = True
        # 该 Tag 中中间变量的显存总量
        self.total_activations_memory = 0.0
        # 该 Tag 下已经 offload 的显存量
        self.current_offload_memory = 0.0
        self.stream = torch.cuda.Stream()

    def reset_current_offload_memory(self):
        """
        重置相关变量
        """
        self.current_offload_memory = 0.0
        if self.is_first_iter:
            logging.info(
                f"[{self.tag}] total_activations_memory: {self.total_activations_memory}"
            )
        self.is_first_iter = False


class CPUOffload():
    _tag_infos = {}

    def __init__(self, offload_ratio: float, tag: str):
        super().__init__()
        self.offload_ratio = offload_ratio
        self.tag = tag
        if getattr(torch.autograd, 'graph', None) is None:
            self.hook = None
            logging.warning(
                f'Current torch version is {torch.__version__}, not support torch.autograd.graph.saved_tensors_hooks. '
                f'Please upgrade your torch version.')
        else:
            if tag not in self._tag_infos:
                logging.info(f"Regist {tag}, offload ratio is {offload_ratio}")
                self._tag_infos[tag] = TagInfo(tag)
            self.hook = torch.autograd.graph.saved_tensors_hooks(
                self.offload_hook, self.load_hook
            )
            self.stream = self._tag_infos[tag].stream

    def __enter__(self):
        if self.hook:
            self.hook.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hook:
            self._tag_infos[self.tag].reset_current_offload_memory()
            self.hook.__exit__(exc_type, exc_value, traceback)

    @classmethod
    def get_tensor_memory(cls, x: torch.Tensor):
        return x.element_size() * x.nelement()

    def offload_hook(self, x: torch.Tensor):
        tag_info = self._tag_infos[self.tag]
        tensor_memory = self.get_tensor_memory(x)
        if tag_info.is_first_iter:
            tag_info.total_activations_memory += tensor_memory
            packed_tuple = (x.device, x.to("cpu", non_blocking=True))
        elif (
                (tag_info.current_offload_memory + tensor_memory) / tag_info.total_activations_memory
        ) <= self.offload_ratio:
            tag_info.current_offload_memory += tensor_memory
            with torch.cuda.stream(self.stream):
                packed_tuple = (x.device, x.to("cpu", non_blocking=True))
        else:
            packed_tuple = (None, x)

        return packed_tuple

    def load_hook(self, packed_tuple: tuple):
        device, x = packed_tuple
        if device is not None and x.device != device:
            event = torch.cuda.Event()
            with torch.cuda.stream(self.stream):
                x = x.to(device, non_blocking=True)
                event.record()
            event.wait()
        return x
