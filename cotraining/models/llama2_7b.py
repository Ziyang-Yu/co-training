from typing import List

import torch

from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, PreTrainedModel
from transformers.models.deberta.modeling_deberta import ContextPooler
from .auto_checkpoint_deberta import DebertaModel
#from transformers import LlamaTokenizer, LlamaForCausalLM

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id="meta-llama/Llama-2-7b-hf"

#tokenizer = AutoTokenizer.from_pretrained(model_id)
#import torch
#model =AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)

class NonParamPooler(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        context_token = hidden_states[:, 0]
        return context_token

    @property
    def output_dim(self):
        return self.config.hidden_size

class llama2_7b(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.__name__ = 'meta-llama/Llama-2-7b'
        self.__num_node_features__ = 768 
        self.device = 'cpu'
        #self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
# Load model directly
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model =AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
        #self.model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
        if config.use_param_free_pooler:
            self.pooler = NonParamPooler(config)
        else:
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
