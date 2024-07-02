import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import tqdm
import sklearn
import numpy as np

from cotraining import *


torch.use_deterministic_algorithms(True) 
graph, num_classes, text = load_data('cora', use_dgl=True, use_text=True)
# print(graph.ndata['x'].shape)
config = load_config()
seed(config.seed)

gcn = GCN(num_layers=config.gnn_num_layers, num_nodes=config.num_nodes, in_feats=config.num_node_features, h_feats=config.gnn_h_feats, num_classes=num_classes, dropout=config.gnn_dropout).to(config.device)
gcn.reset_parameters()

optimizer = torch.optim.Adam(gcn.parameters(), lr=config.gnn_lr, weight_decay=config.gnn_weight_decay)
train_dataloader, valid_dataloader, test_dataloader = init_dataloader(graph, 'train', config), init_dataloader(graph, 'val', config), init_dataloader(graph, 'test', config)


best_val_accuracy = 0.
best_model_path = 'hist_gcn_model.pt'
# best_lm_path = 'deberta_pretrained_graphsage_lm.pt'

for epoch in range(config.epoch):
    gcn.train()

    with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
            # print(output_nodes)
            # inputs = [text[i] for i in output_nodes]
            labels = mfgs[-1].dstdata['y']
            # with torch.no_grad():
            inputs = mfgs[0].srcdata['x']
            predictions = gcn(mfgs=mfgs, x=inputs, batch_size=config.train_batch_size)
            labels = torch.flatten(labels)
            # print(predictions.device, labels.device)
            loss = F.cross_entropy(predictions, labels)
            # loss = torch.tensor(0.)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gcn.parameters(), config.gnn_clip)
            optimizer.step()

            accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())

            tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)

            del input_nodes, output_nodes, mfgs, inputs, labels, predictions, loss
            torch.cuda.empty_cache()
            # print(torch.cuda.mem_get_info())
    gcn.eval()

    predictions = []
    labels = []
    with torch.no_grad() and tqdm.tqdm(valid_dataloader) as tq:
        for input_nodes, output_nodes, mfgs in tq:

            # with torch.no_grad():
            inputs = mfgs[0].srcdata['x']
            labels.append(mfgs[-1].dstdata['y'].cpu().numpy())
            predictions.append(gcn(mfgs=mfgs, x=inputs, batch_size=config.valid_batch_size).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        val_accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        if best_val_accuracy <= val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(gcn, best_model_path)
    # try:
    #     if best_model:
    #         for p1, p2 in zip(best_model.parameters(), torch.load(best_model_path).parameters()):
    #             if p1.data.ne(p2.data).sum() > 0:
    #                 print(False)
    #         print(True)
    # except:
    #     pass
    best_model = torch.load(best_model_path)
    predictions = []
    labels = []
    with torch.no_grad() and tqdm.tqdm(test_dataloader) as tq:
        for input_nodes, output_nodes, mfgs in tq:

            inputs = mfgs[0].srcdata['x']
            labels.append(mfgs[-1].dstdata['y'].cpu().numpy())
            predictions.append(best_model(mfgs=mfgs, x=inputs, batch_size=config.test_batch_size).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        # print(predictions)
        labels = np.concatenate(labels)
        test_accuracy = sklearn.metrics.accuracy_score(labels, predictions)

        print('Epoch {} Valid Accuracy {}  Best Accuracy {} Test Accuracy {}'.format(epoch, val_accuracy, best_val_accuracy, test_accuracy))