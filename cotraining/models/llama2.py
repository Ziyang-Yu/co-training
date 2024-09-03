from typing import List, Optional, Tuple, Union
import torch
from transformers import LlamaModel
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast, Cache, logger


class NonParamPooler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        context_token = hidden_states[:, -1, :1024]
        return context_token

class CroppedLlama2(LlamaModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooler = NonParamPooler()

    def post_init_crop(self, crop_layer_idx):
        self.embed_tokens = None
        self.layers = self.layers[crop_layer_idx + 1: ]

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)[0]
        return self.pooler(output)


if __name__ == '__main__':
    model = CroppedLlama2.from_pretrained('/home/ubuntu/data/models/Llama-2-7b-hf/')
    model.post_init_crop(23)
    model(inputs_embeds=torch.rand(16, 512, 4096), use_cache=False)