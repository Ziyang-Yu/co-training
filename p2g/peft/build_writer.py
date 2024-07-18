from p2g.config import P2GConfig
from tensorboardX import SummaryWriter


def build_writer(config: P2GConfig):
    writer_path = config.exp_save_dir + "/log"
    writer = SummaryWriter(writer_path)

    def add_scalar(scalar_value, step):
        writer.add_scalar("train_loss", scalar_value, step)

    return add_scalar
