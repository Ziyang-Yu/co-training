import torch

from transformers import AutoTokenizer, AutoConfig
from torch_geometric.nn.models import MLP
from fairscale.nn.model_parallel.layers import (
    #ColumnParallelLinear,
    ParallelEmbedding,
    #RowParallelLinear,
)
class mlp(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, model_id="meta-llama/Llama-2-7b-chat-hf"):
        super(mlp, self).__init__()
        self.model = MLP(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.in_channels = in_channels

    def __call__(self, text):

        
        token_ids = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.in_channels).input_ids
        token_ids = torch.tensor(token_ids, dtype=torch.float).cuda()
        #print(token_ids)
        return self.model(token_ids)
        
        


if __name__ == "__main__":

    model = MLP4text(16,16,16,4)
    print(model(['I love you', 'I like you']))

        
