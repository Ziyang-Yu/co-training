from data_utils import load_data
from transformers import BertTokenizer, BertModel



if __name__ == '__main__':

    data, num_classes, text = load_data('cora', use_dgl=True, use_text=True)
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-large-cased')
    model = BertModel.from_pretrained("google-bert/bert-large-cased")
    # text = text[0]
    print(max(len(text[i].split(' ')) for i in range(len(text))))
    encoded_input = [tokenizer(text[i], return_tensors='pt') for i in range(len(text))]
    output = [model(**encoded_input[i]) for i in range(len(text))]
    print(output)