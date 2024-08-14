import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="P2G")
    parser.add_argument("--config", type=str, default="config.yaml", help="config file")
    parser.add_argument("--mode", type=str, default="train", help="train or eval or test")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    from .train import TrainRunner

    runner = TrainRunner(args.config)
    runner(args.mode)


if __name__ == "__main__":
    main()
