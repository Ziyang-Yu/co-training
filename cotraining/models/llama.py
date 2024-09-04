import torch
import tqdm
from transformers import AutoTokenizer, AutoConfig, LlamaTokenizer, LlamaForCausalLM
from torch_geometric.nn.models import MLP
from fairscale.nn.model_parallel.layers import (
    #ColumnParallelLinear,
    ParallelEmbedding,
    #RowParallelLinear,
)
from transformers.models.llama import LlamaConfig
from transformers.models.llama import LlamaModel


class llama(torch.nn.Module):
    
    def __init__(self, num_hidden_layers):
        super(llama, self).__init__()
        self.config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.config.num_hidden_layers = num_hidden_layers
        self.config.output_hidden_states = True
        self.config.hidden_size = 1024
        self.model = LlamaModel(self.config)
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(self.config)
        
    def __call__(self, text):

        #return_tensor = []
        #print('Start forwarding')
        #for t in tqdm.tqdm(text):
        inputs = self.tokenizer(text, max_length=4096, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.cuda() for key, value in inputs.items()}
        #print(list(self.model(**inputs).keys()))
        return_tensor = self.model(**inputs).last_hidden_state
        #print(len(text), return_tensor.shape)
        #return_tensor = torch.stack(return_tensor)
        return return_tensor[:, -1, :]
        
        


if __name__ == "__main__":

    model = MLP4text(16,16,16,4)
    print(model(['I love you', 'I like you']))

        
