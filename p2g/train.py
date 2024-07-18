import torch
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.utils import logger

from p2g.config import P2GConfig, build_p2g_config
from p2g.datasets import build_dataloader
from p2g.ds import build_dist_env, build_ds_config, build_engine, load_model, save_model
from p2g.models.base import P2GModel
from p2g.peft import build_writer


class TrainRunner:
    def __init__(self, config_path: str):
        build_dist_env()
        self.config: P2GConfig = build_p2g_config(config_path)
        self.ds_config: dict = build_ds_config(self.config.ds_config_path)
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.train_length,
            self.val_length,
            self.test_length,
        ) = build_dataloader(self.ds_config, self.config)
        self.engine: PipelineEngine = build_engine(self.config, self.train_length)
        self._train_hooks = [build_writer(self.config)]

    def train_step(self, loader, step):
        loss = self.engine.train_batch(loader)
        for hook in self._train_hooks:
            hook(loss, step)
        return loss

    def test_step(self, loader, is_last):
        preds, loss = self.engine.eval_batch(
            data_iter=loader, do_eval=True, return_logits=True, compute_loss=True, bcast_loss=False, last_batch=is_last
        )
        return preds, loss

    def validate_epoch(self, epoch, epochs):
        testloader = iter(self.val_loader)
        total_step = self.val_length // self.engine.train_batch_size()
        predictions = []
        is_last_rank = self.engine.local_rank == self.engine.world_size - 1
        for step in range(total_step):
            loss, pred = self.test_step(testloader)
            if is_last_rank:
                predictions.append(pred)
        if is_last_rank:
            predictions = torch.cat(predictions, dim=0)  # boolean tensor
            accuracy = torch.sum(predictions).item() / len(predictions)
            logger.info(f"Epoch {epoch + 1} / {epochs}, Validation accuracy: {accuracy}")

    def test_epoch(self, epoch, epochs):
        testloader = iter(self.test_loader)
        total_step = self.test_length
        predictions = []
        losses = []

        is_last_rank = self.engine.local_rank == self.engine.world_size - 1
        for step in range(total_step):
            if is_last_rank:
                logger.info(f"Testing step rank={self.engine.local_rank} {step} / {total_step}")
            loss, pred = self.test_step(testloader, step == total_step - 1)
            predictions.append(pred)
            losses.append(loss)
        import torch.distributed as dist

        dist.barrier()
        if is_last_rank:
            predictions = torch.cat(predictions[:total_step], dim=0)
            accuracy = torch.sum(predictions).item() / len(predictions)
            loss = torch.mean(torch.stack(losses))
            logger.info(f"Epoch {epoch + 1} / {epochs}, Test accuracy: {accuracy}, Test loss: {loss}")

    def save_checkpoint(self):
        # 1. save the lora model, 2. save the gnn model
        save_model(self.engine, self.config)
        logger.info("Checkpoint saved")

    def load_checkpoint(self):
        # 1. load the lora model, 2. load the gnn model
        load_model(self.engine, self.config)
        logger.info("Checkpoint loaded")

    def train(self):
        steps = self.ds_config["train_steps"]
        save_interval = self.ds_config.get("save_interval", max(steps // 10, 10))
        logger.info(f"Start training for {steps} steps, save checkpoint every {save_interval} steps")
        self.load_checkpoint()
        if self.engine.local_rank == 0:
            for name, mod in self.engine.module.named_modules():
                if isinstance(mod, P2GModel):
                    if "Emb" in name:
                        from p2g.models.opt.modeling import PipeOPTEmb

                        mod: PipeOPTEmb
                        mod.embed_tokens.embed_tokens.weight.zero_()
        self.save_checkpoint()
        return
        for step in range(steps):
            self.train_step(self.train_loader, step)
            if step % save_interval == 0 and step != 0:
                self.save_checkpoint()
                logger.info(f"Step {step} / {steps}, Checkpoint saved")
        self.save_checkpoint()

    def eval(self):
        self.load_checkpoint()
        self.validate_epoch(0, 1)

    def test(self):
        self.load_checkpoint()
        self.test_epoch(0, 1)

    def __call__(self, mode="train"):
        if mode == "train":
            self.train()
        elif mode == "eval":
            self.eval()
        elif mode == "test":
            self.test()
        else:
            raise ValueError(f"Unknown mode {mode}")
