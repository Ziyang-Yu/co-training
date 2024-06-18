import torch

from transformers import BertTokenizer, BertModel, AutoTokenizer, DebertaModel, AutoModel, PreTrainedModel

class deberta(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.__name__ = 'microsoft/deberta-base'
        self.__num_node_features__ = 768 
        self.device = 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
# Load model directly
        self.model = AutoModel.from_pretrained("microsoft/deberta-base")
        self.config = config
        # self.model = DebertaModel.from_pretrained("microsoft/deberta-base")
        
        # self.__output_dim__ = self.__model__.
    # @property
    def parameters(self):
        return self.model.parameters()

    @property
    def num_node_features(self) -> int:
        return self.__num_node_features__

    def to(self, device) -> None:
        self.model = self.model.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def forward_once(self, texts) -> torch.Tensor:

        # input = self.tokenizer(texts, padding=True, truncation=True, max_length=self.config.model.deberta.max_length, return_tensors='pt').to(self.device)
        inputs = self.tokenizer(texts, padding=self.config.lm_padding, truncation=self.config.lm_truncation, max_length=self.config.lm_max_length, return_tensors='pt').to(self.device)
        output = self.model(**inputs).last_hidden_state.mean(dim=1)
        return output


    def forward(self, texts) -> torch.Tensor:
        # def model_forward_input(input):
        input = self.tokenizer(texts, padding=self.config.lm_padding, truncation=self.config.lm_truncation, max_length=self.config.lm_max_length, return_tensors='pt').to(self.device)
        output = self.model(**input).last_hidden_state.mean(dim=1)
        return output

    def __call__(self, data) -> torch.Tensor:
        if isinstance(data, str):
            return self.forward([data])
        if isinstance(data, list):
            return self.forward(data)