
from argparse import ArgumentParser
from optuna_dashboard import run_server
from optuna.storages import JournalStorage, JournalFileStorage

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="path to Journal Storage")
    args = parser.parse_args()
    storage = JournalStorage(JournalFileStorage(args.path))
    run_server(storage, host="127.0.0.1", port=8080)
