from deepspeed.runtime.pipe.engine import PipelineEngine

from p2g.config import P2GConfig

from .dgl import build_dgl_dataloader


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def build_dataloader(ds_config: dict, config: P2GConfig):
    dataloader_type = config.dataloader_type
    if dataloader_type == "dgl":
        train_dl, val_dl, test_dl = build_dgl_dataloader(ds_config, config)
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {dataloader_type}")
    train_length, val_length, test_length = (
        len(train_dl.dataset) * ds_config["n_epoches"],
        len(val_dl.dataset),
        len(test_dl.dataset),
    )
    train_dl = iter(cyclic_iter(train_dl))
    val_dl = iter(cyclic_iter(val_dl))
    test_dl = iter(cyclic_iter(test_dl))
    return train_dl, val_dl, test_dl, train_length, val_length, test_length
