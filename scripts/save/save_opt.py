from transformers import AutoModelForCausalLM, AutoTokenizer, OPTModel
import torch
from p2g.models.opt.modeling import OPTHead, OPTEmb

model_name = "facebook/opt-1.3b"

model: OPTModel = OPTModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

head = OPTHead(model.config)
head.project_out = model.decoder.project_out
head.final_layer_norm = model.decoder.final_layer_norm
torch.save(head.state_dict(), f"opt-1.3b/head.pth")

emb = OPTEmb(model.config)
emb.embed_tokens = model.decoder.embed_tokens
emb.embed_positions = model.decoder.embed_positions
emb.project_in = model.decoder.project_in
torch.save(emb.state_dict(), f"opt-1.3b/embed_tokens.pth")

for i, layer in enumerate(model.decoder.layers):
    torch.save(layer.state_dict(), f"opt-1.3b/decoder.{i}.pth")

model.save_pretrained("/home/jupyter/models/opt-1.3b")
tokenizer.save_pretrained("/home/jupyter/models/opt-1.3b")
