import os
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="path to wandb logs")
    parser.add_argument("--path", required=True, type=str, help="prefix for int id")
    args = parser.parse_args()
    for dir in os.listdir(args.path):
        id = dir.split('-')[-1]
        try:
            id = int(id)
            print(id)
            os.system(f"wandb sync --id={args.prefix}_{id} {os.path.join(args.path, dir)}")
        except:
            pass