from typing import Union
from argparse import ArgumentParser


def add_model_argparse_args(parser: ArgumentParser):
    group = parser.add_argument_group("monai.net")
    # Common args
    group.add_argument("--pretrained", type=str, help="path to pre-trained model checkpoint")
    group.add_argument("--ckpt_path", type=str, help="path to the pytorch lightning training checkpoint")
    group.add_argument("--model_name", default="unetr", type=str, help="model name (unetr, swin_unetr)")
    group.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    group.add_argument("--out_channels", default=14, type=int, help="number of output channels")
    group.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    group.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    group.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    group.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
    group.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
    group.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
    group.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
    group.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
    group.add_argument("--no_conv_block", action="store_true", help="convolutional block is not used in Unet blocks")
    group.add_argument("--no_res_block", action="store_true", help="residual block is not used in Unet blocks")
    group.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    group.add_argument("--spatial_dims", default=3, type=int, help="number of spatial dims of UNTER input")
    group.add_argument("--qkv_bias", action="store_true",
                        help="bias term for the qkv linear layer in self attention block")
    group.add_argument("--vit_norm_name", type=str, default="layer", help="Normalization type in ViT blocks")
    group.add_argument("--vit_norm_no_affine", action="store_true", help="Not affine parameters in ViT norm")
    group.add_argument("--encoder_norm_name", type=str, default="instance_cond",
                        help="Normalization type in encoder blocks")
    group.add_argument("--encoder_norm_no_affine", action="store_true", help="Not affine parameters in ViT norm")
    group.add_argument("--decoder_norm_name", type=str, default="group",
                        help="Normalization type in decoder blocks")
    group.add_argument("--decoder_norm_no_affine", action="store_true", help="Not affine parameters in ViT norm")
    group.add_argument("--num_groups", type=int, default=4, help="For group norm")
    group.add_argument("--num_styles", type=int, default=2, help="For instance_cond norm")
    '''
    # Swin-UNETR exclusive args
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--attn_drop_rate", default=0.0, type=float, help="attn drop rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    '''
    # Loss,
    group = parser.add_argument_group("loss")
    group.add_argument("--squared_dice", action="store_true", help="use squared Dice")
    group.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
    group.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    # Optimizer
    group = parser.add_argument_group("optimizer")
    group.add_argument("--lr", default=1e-4, type=float, help="optimization learning rate")
    group.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
    group.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    group.add_argument("--momentum", default=0.99, type=float, help="momentum only for SGD")
    # Scheduler
    group.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
    # Inference
    group = parser.add_argument_group("inference")
    group.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    group.add_argument("--sw_batch_size", default=1, type=int, help="sliding window batch size for inference")
    group.add_argument("--infer_cpu", action="store_true", help="Save in CPU the stitched output prediction")
    # Early stop
    group = parser.add_argument_group("early_stop")
    group.add_argument("--patience", default=3, type=int, help="patience for early stop")
    group.add_argument("--min_delta", default=0.00, type=float,
                       help="minimum change the monitored in accuracy to qualify as an improvement")
    # Checkpointing
    group = parser.add_argument_group("checkpointing")
    group.add_argument("--save_top_k", default=3, type=int, help="number of checkpoints to save with best accuracy")
    # Logger
    group = parser.add_argument_group("wandb_logger")
    group.add_argument("--experiment_name", type=str, help="wandb name")
    group.add_argument("--group", type=str, help="wandb group")
    group.add_argument("--project", type=str, help="wandb project")
    group.add_argument("--entity", type=str, help="wandb entity")
    return parser


def add_data_argparse_args(parser: ArgumentParser):
    group = parser.add_argument_group("dataset(s)")
    group.add_argument("--data_dir", default="dataset/MultiModalPelvic", type=str, help="dataset directory")
    group.add_argument('--json_lists',default=['CT.json', 'MR.json'], nargs='+',
                        help='Json list(s) of input dataset(s)', type=str)
    group.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
    group.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
    group.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
    group.add_argument("--patches_training_sample", default=1, type=int, help="number of patches per training sample")
    group.add_argument("--randFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
    group.add_argument("--randRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
    group.add_argument("--randScaleIntensityd_prob", default=0.1, type=float,
                        help="randScaleIntensityd aug probability")
    group.add_argument("--randShiftIntensityd_prob", default=0.1, type=float,
                        help="randShiftIntensityd aug probability")
    group.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
    group.add_argument("--cache_num", default=24, type=int, help="samples to cache on dataloader")
    group.add_argument("--loader_workers", default=8, type=int, help="number of workers to load dataset in cache")
    group.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    group.add_argument("--num_workers", default=8, type=int, help="number of workers for the dataloaders")
    return parser