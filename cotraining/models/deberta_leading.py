from typing import List

import torch
from transformers import (AutoModel, AutoTokenizer, BertModel, BertTokenizer,
                          PreTrainedModel)
from transformers.models.deberta.modeling_deberta import ContextPooler

from ..utils.auto_ckpt import CPUOffload
from .auto_checkpoint_deberta import DebertaModel


class DEBERTALeading(torch.nn.Module):

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

    def forward(self, neighbor_texts: List[str], target_texts: List[str]) -> torch.Tensor:
        neighbor_texts = self.tokenizer(neighbor_texts, padding=self.config.lm_padding, truncation=self.config.lm_truncation, max_length=self.config.lm_max_length, return_tensors='pt').to(self.device)
        target_texts = self.tokenizer(target_texts, padding=self.config.lm_padding, truncation=self.config.lm_truncation, max_length=self.config.lm_max_length, return_tensors='pt').to(self.device)
        
        encoder_layer = self.model(**input)[0]
        pooled_output = encoder_layer[:,0,:]
        return pooled_output

    def __call__(self, data) -> torch.Tensor:
        if isinstance(data, str):
            return self.forward([data])
        if isinstance(data, list):
            return self.forward(data)