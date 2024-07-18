from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./llama2-7b")
tokenizer.save_pretrained("./llama2-7b")
import torch

torch.save(model.lm_head.state_dict(), f"llama2-pipe/lm_head.pth")
torch.save(model.model.embed_tokens.state_dict(), f"llama2-pipe/emd.pth")
for i, layer in enumerate(model.model.layers):
    torch.save(layer.state_dict(), f"llama2-pipe/layer_{i}.pth")
