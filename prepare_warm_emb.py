import argparse
#import dgl
#import torch
#from tqdm import tqdm
#from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel, OPTModel
#from torch_geometric.utils import to_dgl
from p2g.config import build_p2g_config
#from p2g.datasets.dgl.dgl_text_loader import load_dgl_dataset
#from p2g.models.gnn.modeling import NonParamPooler
#from p2g.models.opt.modeling import OPTEmb, OPTHead
#from cotraining.models import llama2_7b
from vllm import LLM

#torch.multiprocessing.set_start_method('spawn')
#import time

def load_data(config, seed=0):
    dataset = config.dataset_name
    dataset_path = config.dataset_path
    if dataset == "cora":
        from p2g.datasets.dgl.cora import get_raw_text_cora as get_raw_text

        num_classes = 7
    elif dataset == "ogbn-arxiv":
        from p2g.datasets.dgl.arxiv import get_raw_text_arxiv as get_raw_text

        num_classes = 40
    else:
        raise ValueError(f"Error: Dataset {dataset} not supported")
    data, text = get_raw_text(dataset_path, use_text=True, seed=seed)
    data = to_dgl(data)
    return data, num_classes, text


def load_dgl_dataset(config):
    graph, num_classes, text = load_data(config)
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    return graph, num_classes, text


def get_emb(model, text_list, padding_length, batch_size=1):
    # encoder_layer = self.model(**input)[0]
    # pooled_output = self.pooler(encoder_layer)
    # return pooled_output
    outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size)):
            text_list_batch = text_list[i : i + batch_size]
            #input = tokenizer(
            #    text_list_batch,
            #    padding=True,
            #    truncation=True,
            #    max_length=padding_length,
            #    return_tensors="pt",
            #)
            # move input to cuda
            #for k, v in input.items():
            #    input[k] = v if isinstance(v, torch.Tensor) else v
            #encoder_layer = model(**input)[0]
            #pooled_output = pooler(encoder_layer).cpu()
            #print(pooled_output.shape)
            #outputs.append(pooled_output)
#            print(text_list_batch)
            outputs.append(model(text_list_batch)[0].detail.hidden_states)
#            time.sleep(1)
    full_emb = torch.cat(outputs, dim=0)
#    return full_emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lm_type", type=str, required=True, choices=["llama", "opt"])
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    p2gconfig = build_p2g_config(args.config_path)
    padding_length = p2gconfig.dataset_padding_width
    #model = llama2_7b()
    #model = (
    #    OPTModel.from_pretrained(args.model_name)
    #    if args.lm_type == "opt"
    #    else LlamaModel.from_pretrained(args.model_name)
    #)
    #pooler = NonParamPooler(p2gconfig)
    #tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    #tokenizer.pad_token = tokenizer.eos_token if args.lm_type == "llama" else tokenizer.pad_token
    #graph, num_classes, text = load_dgl_dataset(p2gconfig)
    #text_list = text
    #print(111111111111111111111111111111111111)
    if args.lm_type == "llama":
        model = LLM("meta-llama/Llama-2-7b-hf", tensor_parallel_size=4)
    #print(22222222222222222222222222222222)
    emb = get_emb(model, ['set joasjfof'], padding_length)
    torch.save(emb, args.output_dir + "/warm_emb.pth")


main()
