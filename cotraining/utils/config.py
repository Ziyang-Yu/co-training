import argparse

def load_config() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
                        prog='Add Config',
                        description='Set all the hyperparameters for the project')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model')
    parser.add_argument('--epoch', type=int, default=2000, help='Number of epochs to train the model')
    parser.add_argument('--lm_type', type=str, default='deberta-base', help='Type of language model to use')
    parser.add_argument('--lm_lr', type=float, default=0.01, help='Learning rate for the language model')
    parser.add_argument('--lm_max_length', type=int, default=512, help='Maximum length of the input sequence for the language model')
    parser.add_argument('--lm_weight_decay', type=float, default=1e-4, help='Weight decay for the language model')
    parser.add_argument('--lm_padding', type=bool, default=True, help='Whether to pad the input sequence for the language model')
    parser.add_argument('--lm_truncation', type=bool, default=True, help='Whether to truncate the input sequence for the language model')
    parser.add_argument('--lm_requires_grad', type=bool, default=True, help='Whether to require gradient for the language model')
    parser.add_argument('--pooler_hidden_size', type=int, default=768, help='Hidden size of the pooler layer')
    parser.add_argument('--pooler_dropout', type=float, default=0.5, help='Dropout for the pooler layer')
    parser.add_argument('--pooler_hidden_act', type=str, default='relu', help='Activation function for the pooler layer')
    parser.add_argument('--num_nodes', type=int, default=169343, help='Number of nodes in the graph')
    parser.add_argument('--num_node_features', type=int, default=768, help='Number of features for each node in the graph')
    parser.add_argument('--gnn_h_feats', type=int, default=256, help='Hidden features for the GNN')
    parser.add_argument('--gnn_lr', type=float, default=0.0005, help='Learning rate for the GNN')
    parser.add_argument('--gnn_weight_decay', type=float, default=0, help='Weight decay for the GNN')
    parser.add_argument('--gnn_dropout', type=float, default=0.5, help='Dropout for the GNN')
    parser.add_argument('--gnn_requires_grad', type=bool, default=True, help='Whether to require gradient for the GNN')
    parser.add_argument('--gnn_num_layers', type=int, default=7, help='Number of layers for the GNN')
    parser.add_argument('--gnn_clip', type=float, default=1.0, help='Gradient clipping for the GNN')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Batch size for the training')
    parser.add_argument('--train_shuffle', type=bool, default=True, help='Whether to shuffle the data for the training')
    parser.add_argument('--train_drop_last', type=bool, default=True, help='Whether to drop the last batch for the training')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='Batch size for the validation')
    parser.add_argument('--valid_shuffle', type=bool, default=True, help='Whether to shuffle the data for the validation')
    parser.add_argument('--valid_drop_last', type=bool, default=True, help='Whether to drop the last batch for the validation')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Batch size for the testing')
    parser.add_argument('--test_shuffle', type=bool, default=True, help='Whether to shuffle the data for the testing')
    parser.add_argument('--test_drop_last', type=bool, default=True, help='Whether to drop the last batch for the testing')

    args = parser.parse_args()
    return args