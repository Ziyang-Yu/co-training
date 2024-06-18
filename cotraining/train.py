import tqdm
import torch

def forward_once(dataloader, model):
    with tqdm.tqdm(dataloader) as tq:
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
            # inputs = [text[i] for i in input_nodes]
            with torch.no_grad():
                # x = lm(inputs)
                # x = mfgs[0].srcdata['x']
                # print(len(input_nodes), len(output_nodes))
                # x = [text[i] for i in input_nodes]
                # x = lm.forward_once(x)
                model.forward_once(mfgs)