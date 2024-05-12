import torch

from data_utils import load_data
from transformers import BertTokenizer, BertModel, AutoTokenizer, DebertaModel


if __name__ == '__main__':

    data, num_classes, text = load_data('cora', use_dgl=True, use_text=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    model = DebertaModel.from_pretrained("microsoft/deberta-base").cuda()
    # text = text[0]
    # print(max(len(text[i].split(' ')) for i in range(len(text))))
    encoded_input = [tokenizer(text[i], return_tensors='pt') for i in range(len(text))]
    output = []
    with torch.no_grad():
        for tokens in encoded_input:
            tokens = tokens.to('cuda')
            output.append(model(**tokens).last_hidden_state.cpu())
    print(output[0])
        
    # output = torch.stack([model(**encoded_input[i]).last_hidden_state for i in range(len(text))])
    # print(output[0])
