import torch
import dgl

def init_dataloader(graph, name, config):

    sampler = dgl.dataloading.NeighborSampler([10, 10])

    train_mask = graph.ndata['train_mask']
    train_nids = torch.nonzero(train_mask, as_tuple=False).squeeze()
    val_mask = graph.ndata['val_mask']
    val_nids = torch.nonzero(val_mask, as_tuple=False).squeeze()
    test_mask = graph.ndata['test_mask']
    test_nids = torch.nonzero(test_mask, as_tuple=False).squeeze()

    if name == 'once':

        once_train_loader = dgl.dataloading.DataLoader(
            graph, 
            train_nids, 
            sampler,
            device=config.device,
            batch_size=config.once_batch_size,
            shuffle=config.once_shuffle,
            drop_last=config.once_drop_last,
            num_workers=0,
        )
        once_val_loader = dgl.dataloading.DataLoader(
            graph, 
            val_nids, 
            sampler,
            device=config.device,
            batch_size=config.once_batch_size,
            shuffle=config.once_shuffle,
            drop_last=config.once_drop_last,
            num_workers=0,
        )
        once_test_loader = dgl.dataloading.DataLoader(
            graph, 
            test_nids, 
            sampler,
            device=config.device,
            batch_size=config.once_batch_size,
            shuffle=config.once_shuffle,
            drop_last=config.once_drop_last,
            num_workers=0,
        )

        return once_train_loader, once_val_loader, once_test_loader

    if name == 'train':
        train_loader = dgl.dataloading.DataLoader(
            # The following arguments are specific to DGL's DataLoader.
            graph,              # The graph
            train_nids,         # The node IDs to iterate over in minibatches
            sampler,            # The neighbor sampler
            device=config.device,      # Put the sampled MFGs on CPU or GPU
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=config.train_batch_size,    # Batch size
            shuffle=config.train_shuffle,       # Whether to shuffle the nodes for every epoch
            drop_last=config.train_drop_last,    # Whether to drop the last incomplete batch
            num_workers=0       # Number of sampler processes
        )
        return train_loader
    elif name == 'val':

        val_loader = dgl.dataloading.DataLoader(
            graph, 
            val_nids, 
            sampler,
            device=config.device,
            batch_size=config.valid_batch_size,
            shuffle=config.valid_shuffle,
            drop_last=config.valid_drop_last,
            num_workers=0,
        )
        return val_loader
    elif name == 'test':
        test_mask = graph.ndata['test_mask']
        # test_mask = graph.test_mask
        test_nids = torch.nonzero(test_mask, as_tuple=False).squeeze()
        test_dataloader = dgl.dataloading.DataLoader(
            graph, 
            test_nids, 
            sampler,
            batch_size=config.test_batch_size,
            shuffle=config.test_shuffle,
            drop_last=config.test_drop_last,
            num_workers=0,
            device=config.device
        )
        return test_dataloader
    else:
        raise NotImplementedError