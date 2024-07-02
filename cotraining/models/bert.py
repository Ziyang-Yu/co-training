from typing import List

import torch

from transformers import BertTokenizer, BertModel, AutoTokenizer, DebertaModel, AutoModel, PreTrainedModel
from transformers.models.deberta.modeling_deberta import ContextPooler

class bert(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.__name__ = 'google-bert/bert-base-uncased'
        self.__num_node_features__ = 768 
        self.device = 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(self.__name__)
# Load model directly
        self.model = AutoModel.from_pretrained(self.__name__)
        self.pooler = ContextPooler(config) 
        self.config = config
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

    def forward(self, texts: List[str]) -> torch.Tensor:
        input = self.tokenizer(texts, padding=self.config.lm_padding, truncation=self.config.lm_truncation, max_length=self.config.lm_max_length, return_tensors='pt').to(self.device)
        encoder_layer = self.model(**input)[0]
        pooled_output = self.pooler(encoder_layer)
        return pooled_output

    def __call__(self, data) -> torch.Tensor:
        if isinstance(data, str):
            return self.forward([data])
        if isinstance(data, list):
            return self.forward(data)