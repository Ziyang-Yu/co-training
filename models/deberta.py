import torch

from transformers import BertTokenizer, BertModel, AutoTokenizer, DebertaModel, AutoModel, PreTrainedModel

class deberta:

    def __init__(self):
        self.__name__ = 'microsoft/deberta-base'
        self.__num_node_features__ = 768 
        self.device = 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
# Load model directly
        self.model = AutoModel.from_pretrained("microsoft/deberta-base")
        # self.model = DebertaModel.from_pretrained("microsoft/deberta-base")
        
        # self.__output_dim__ = self.__model__.
    # @property
    def parameters(self):
        return self.model.parameters()

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
        if isinstance(data, str):
            return self.forward([data])
        if isinstance(data, list):
            return self.forward(data)