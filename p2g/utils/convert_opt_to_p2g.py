import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTModel

from p2g.models.opt.modeling import OPTEmb, OPTHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    model: OPTModel = OPTModel.from_pretrained(args.model_name)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    head = OPTHead(model.config)
    head.project_out = model.decoder.project_out
    head.final_layer_norm = model.decoder.final_layer_norm
    torch.save(head.state_dict(), f"{args.output_dir}/head.pth")

    emb = OPTEmb(model.config)
    emb.embed_tokens = model.decoder.embed_tokens
    emb.embed_positions = model.decoder.embed_positions
    emb.project_in = model.decoder.project_in

    torch.save(emb.state_dict(), f"{args.output_dir}/embed_tokens.pth")

    for i, layer in enumerate(model.decoder.layers):
        torch.save(layer.state_dict(), f"{args.output_dir}/decoder.{i}.pth")


main()
