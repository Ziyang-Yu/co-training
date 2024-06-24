from typing import List

import torch

from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, PreTrainedModel
from transformers.models.deberta.modeling_deberta import ContextPooler
from .auto_checkpoint_deberta import DebertaModel

from torch.autograd.graph import saved_tensors_hooks
class save_on_cpu_async(saved_tensors_hooks):
    def __init__(self, pin_memory=False, device_type="cuda", device=None):
        device_module = getattr(torch, device_type, torch.cuda)
        new_stream = torch.cuda.Stream(device=device)

        def pack_to_cpu(tensor):
            if not pin_memory:
                return (tensor.device, tensor.cpu(), None)
            print(f'pack tensor {tensor.shape} {tensor.layout} ', flush=True)
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
            print(f'unpack tensor {tensor.shape} {tensor.layout} ', flush=True)
            if end_event: 
                end_event.synchronize()
            print(f'done tensor {tensor.shape} {tensor.layout} ', flush=True)
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


class deberta(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.__name__ = 'microsoft/deberta-base'
        self.__num_node_features__ = 768 
        self.device = 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
# Load model directly
        self.model = DebertaModel.from_pretrained("microsoft/deberta-base")
        self.pooler = ContextPooler(config) 
        self.config = config
        # self.model = DebertaModel.from_pretrained("microsoft/deberta-base")
        
        # self.__output_dim__ = self.__model__.
    # @property
    def parameters(self):
        return self.model.parameters()

    @property
    def num_node_features(self) -> int:
        return self.__num_node_features__

    def to(self, device) -> None:
        self.model = self.model.to(device)
        self.pooler = self.pooler.to(device)
        self.device = device
        return self

    # @torch.no_grad()
    # def forward_once(self, texts) -> torch.Tensor:

    #     # input = self.tokenizer(texts, padding=True, truncation=True, max_length=self.config.model.deberta.max_length, return_tensors='pt').to(self.device)
    #     inputs = self.tokenizer(texts, padding=self.config.lm_padding, truncation=self.config.lm_truncation, max_length=self.config.lm_max_length, return_tensors='pt').to(self.device)
    #     output = self.model(**inputs)
    #     print(output)
    #     raise KeyError
    #     return output


    # def forward(self, texts) -> torch.Tensor:
    #     # def model_forward_input(input):
    #     input = self.tokenizer(texts, padding=self.config.lm_padding, truncation=self.config.lm_truncation, max_length=self.config.lm_max_length, return_tensors='pt').to(self.device)
    #     output = self.model(**input).last_hidden_state.mean(dim=1)
    #     return output

    def forward(self, texts: List[str]) -> torch.Tensor:
        input = self.tokenizer(texts, padding=self.config.lm_padding, truncation=self.config.lm_truncation, max_length=self.config.lm_max_length, return_tensors='pt').to(self.device)
        with save_on_cpu(pin_memory=True, device=self.device):
            encoder_layer = self.model(**input)[0]
            pooled_output = self.pooler(encoder_layer)
            return pooled_output

    def __call__(self, data) -> torch.Tensor:
        if isinstance(data, str):
            return self.forward([data])
        if isinstance(data, list):
            return self.forward(data)