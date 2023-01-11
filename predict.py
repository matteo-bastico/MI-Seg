import torch

from argparse import ArgumentParser, Namespace
from networks.utils import model_from_argparse_args


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    hyper_parameters = checkpoint["hyper_parameters"]
    model = model_from_argparse_args(Namespace(**hyper_parameters))
    model_weights = checkpoint["state_dict"]
    # update keys by dropping `auto_encoder.`
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    model.eval()
    x = torch.rand(
        1,
        hyper_parameters['in_channels'],
        hyper_parameters['roi_x'],
        hyper_parameters['roi_y'],
        hyper_parameters['roi_z'],
    )
    with torch.no_grad():
        pred = model(x, modalities=1)
    print(pred)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default="Test_unet/3hs4d1wo/checkpoints/5-6.ckpt", type=str, help="Checkpoint")
    args = parser.parse_args()
    print(args)
    main(args)