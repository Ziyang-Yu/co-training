from typing import Callable, List

import torch
from deepspeed.runtime.pipe.module import LayerSpec
from torch.nn import Embedding, Module
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

from p2g.config import P2GConfig
from p2g.models.base import P2GModel
from p2g.models.utils import make_mask


class PipeLlamaEmb(Module, P2GModel):
    def __init__(self, config: LlamaConfig, pipe_config: P2GConfig):
        super().__init__()
        self.config = pipe_config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self._special_name = "embed_tokens"

    def load_ckpt(self, path=None):
        ckpt = self.config.llm_ckpt_path or path
        if ckpt:
            self.embed_tokens.load_state_dict(torch.load(ckpt + f"/{self._special_name}.pth"))

    def prepare_peft(self, prepare_fn: Callable):
        lora_config = self.config.llm_lora_config
        self.embed_tokens = prepare_fn(self.embed_tokens, lora_config)

    def save_peft(self, save_filter: Callable):
        name2params = dict(self.embed_tokens.named_parameters())
        name2params = save_filter(name2params)
        torch.save(
            name2params,
            self.config.llm_lora_ckpt_save_path + f"/{self._special_name}.pth",
        )

    def load_peft(self):
        name2params = torch.load(self.config.llm_lora_ckpt_save_path + f"/{self._special_name}.pth")
        self.embed_tokens.load_state_dict(name2params, strict=False)

    def forward(self, input_args: torch.Tensor):
        input_ids = input_args
        inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds


class PipeLlamaDecoder(Module, P2GModel):
    def __init__(self, config: LlamaConfig, layer_idx: int, pipe_config: P2GConfig):
        super().__init__()
        self.config = pipe_config
        self.decoder = LlamaDecoderLayer(config, layer_idx)
        self.layer_ = layer_idx
        self.mask_cache = {}
        self._special_name = f"decoder.{layer_idx}"

    def load_ckpt(self, path=None):
        ckpt = self.config.llm_ckpt_path or path
        if ckpt:
            self.decoder.load_state_dict(torch.load(ckpt + f"/{self._special_name}.pth"))

    def prepare_peft(self, prepare_fn: Callable):
        lora_config = self.config.llm_lora_config
        self.decoder = prepare_fn(self.decoder, lora_config)

    def save_peft(self, save_filter: Callable):
        name2params = dict(self.decoder.named_parameters())
        name2params = save_filter(name2params)
        torch.save(
            name2params,
            self.config.llm_lora_ckpt_save_path + f"/{self._special_name}.pth",
        )

    def load_peft(self):
        name2params = torch.load(self.config.llm_lora_ckpt_save_path + f"/{self._special_name}.pth")
        self.decoder.load_state_dict(name2params, strict=False)

    def forward(self, input):
        hidden_states = input
        position_ids = (
            torch.arange(hidden_states.size(1), device=hidden_states.device)
            .unsqueeze(0)
            .expand(hidden_states.size(0), -1)
        )
        if hidden_states.shape in self.mask_cache:
            attn_mask = self.mask_cache[hidden_states.shape]
        else:
            attn_mask = make_mask(
                hidden_states.size(0), hidden_states.size(1), hidden_states.device, hidden_states.dtype
            )
            self.mask_cache[hidden_states.shape] = attn_mask
        (hidden_states,) = self.decoder(
            hidden_states=hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            output_attentions=False,
            use_cache=False,
        )
        return hidden_states


class PipeLlamaHead(Module, P2GModel):
    def __init__(self, config: LlamaConfig, pipe_config: P2GConfig):
        super().__init__()
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def load_ckpt(self, path=None):
        pass

    def prepare_peft(self, prepare_fn: Callable):
        pass

    def save_peft(self, save_filter: Callable):
        pass

    def load_peft(self):
        pass

    def forward(self, inputs):
        hidden_states = inputs
        hidden_states = self.norm(hidden_states)
        return hidden_states


def get_llama_model_specs(pipe_config: P2GConfig) -> List[LayerSpec]:
    model_path = pipe_config.llm_cfg_path
    llm_dtype = pipe_config.llm_dtype
    config = LlamaConfig.from_pretrained(model_path, dtype=llm_dtype)

    specs = []
    emb_spec = LayerSpec(PipeLlamaEmb, config, pipe_config)
    specs.append(emb_spec)
    for i in range(config.num_hidden_layers):
        layer_spec = LayerSpec(PipeLlamaDecoder, config, i, pipe_config)
        specs.append(layer_spec)
    head_spec = LayerSpec(PipeLlamaHead, config, pipe_config)
    specs.append(head_spec)
    return specs
