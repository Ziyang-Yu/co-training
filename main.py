import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset

from data_utils import load_data
from transformers import BertTokenizer, BertModel, AutoTokenizer, DebertaModel


class deberta:

    def __init__(self):
        self.__name__ = 'deberta-base'
        # self.__num_node_features__ = 
        self.device = 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        self.model = DebertaModel.from_pretrained("microsoft/deberta-base")
        
        # self.__output_dim__ = self.__model__.

    @property
    def num_node_features(self):
        return 768

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self

    def forward(self, text):

        def model_forward_input(input):
            input = self.tokenizer(input, return_tensors='pt').to(self.device)
            output = self.model(**input).last_hidden_state.mean(dim=1)
            # print(output.shape)
            # return self.model(**input).last_hidden_state.mean(dim=1)
            # print(output.shape)
            return torch.squeeze(output)

        return torch.stack(list(map(model_forward_input, text)))

    def __call__(self, data):
        x = self.forward(data.text)
        data.x = x
        return data
    



class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
class dataset:

    def __init__(self, name) -> None:

        if name == 'cora':

            # self.data = Planetoid(root='/tmp/Cora', name='Cora', split='random')
            data, num_classes, text = load_data('cora', use_dgl=True, use_text=True)
            self.num_classes = num_classes
            self.text = text
            self.y = data.y
            self.train_mask = data.train_mask
            self.val_mask = data.val_mask
            self.test_mask = data.test_mask
            self.edge_index = data.edge_index
        else:
            raise NotImplementedError
        
    def to(self, device):
        self.y = self.y.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        self.edge_index = self.edge_index.to(device)
        return self

        
    # def update(self, x):
    #     self.data.x = x

    # @property
    # def num_classes(self):
    #     return self.num_classes
    

if __name__ == '__main__':

    # data, num_classes, text = load_data('cora', use_dgl=True, use_text=True)
    data = dataset('cora').to('cuda')


    lm = deberta().to('cuda')
    gcn = GCN(num_node_features=lm.num_node_features, num_classes=data.num_classes).to('cuda')
    with torch.no_grad():
        data = lm(data)
        data = gcn(data)
    # print(lm(dataset).x.shape)
    # gcn = GCN(num_node_features=lm.num_node_features, num_classes=dataset.num_classes)




    # print(len(dataset.data.train_mask), len(text))

    
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    # model = DebertaModel.from_pretrained("microsoft/deberta-base").cuda()
    # # text = text[0]
    # # print(max(len(text[i].split(' ')) for i in range(len(text))))
    # encoded_input = [tokenizer(text[i], return_tensors='pt') for i in range(len(text))]
    # output = []
    # with torch.no_grad():
    #     for tokens in encoded_input:
    #         tokens = tokens.to('cuda')
    #         output.append(model(**tokens).last_hidden_state.cpu())
    # print(output[0])
    
    # output = torch.stack([model(**encoded_input[i]).last_hidden_state for i in range(len(text))])
    # print(output[0])
