import argparse
import os
import random

import gcsfs
import ml_collections
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from configs.config import get_config
from networks.discriminator import McImageDis as discriminator
from networks.mmt import MMT as generator
from trainer_brats import trainer_brats
from trainer_ixi import trainer_ixi
from trainer_spine import trainer_spine

os.environ['NCCL_DEBUG'] = 'INFO'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

# Root path
# TODO: update root path default value
parser.add_argument('--root_path', type=str,
                    default='/subtlemedical-ml-pipeline',
                    help='data root dir')
parser.add_argument('--dataset', type=str,
                    default='IXI', help='dataset name')
parser.add_argument('--n_contrast', type=int, default=3, help='total number of contrast in the dataset')

# set up model
parser.add_argument('--cfg', type=str, default='configs/mmt_ixi.yml', help='model configuration')
parser.add_argument('--exp', type=str, default='IXI_Single', help='name of experiment')
parser.add_argument('--k', type=int,
                    default=2, help='number of inputs')
parser.add_argument('--zero_gad', action='store_true', help='synthesis T1Gd only', default=False)
parser.add_argument('--mra_synth', action='store_true', help='MRA Synthesis from T1+T2 using IXI', default=False)
# set up loss function
parser.add_argument('--lambda_self', type=float, default=5, help='weight of self-recon loss')
parser.add_argument('--lambda_cross', type=float, default=20, help='weight of self-cross loss')
parser.add_argument('--lambda_triplet', type=float, default=0, help='weight of triplet loss')
parser.add_argument('--margin', type=float, default=0.1, help='margin for triplet loss')
parser.add_argument('--lambda_GAN', type=float, default=0.1, help='weight of GAN loss')
parser.add_argument('--lambda_seg', type=float, default=0, help='weight of segmentation loss')
parser.add_argument('--lambda_perceptual', type=float, default=0,
                    help='weight of vgg19 perceptual loss')
parser.add_argument('--lambda_ssim', type=float, default=0, help='MSF_SSIM loss')
parser.add_argument('--enh_weight', action='store_true',
                    help='Weight the cross reconstruction loss with enhancement of post-pre', default=False)
parser.add_argument('--seg_channel', type=int, default=3, help='number of segmentation channels')
parser.add_argument('--label_smoothing', action='store_true', help='use label smoothing for training discriminator')
parser.add_argument('--no_cross_contrast_attn', action='store_true',
                    help='If True, attention is computed across multiple input contrasts, else the contrast dimension is collapsed',
                    default=False)

# set up optimizer
parser.add_argument('--optimizer', type=str, default='adamw')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')
parser.add_argument('--base_lr_g', type=float, default=5e-4,
                    help='network learning rate')
parser.add_argument('--base_lr_d', type=float, default=1e-4,
                    help='network learning rate')
# set up training
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--warmup_epoch', type=int, default=3, help='epochs for lr warmup')
parser.add_argument('--max_epochs', type=int,
                    default=2, help='maximum epoch number to train')
parser.add_argument('--ckpt', type=str, default=None, help='load ckpt and resume training')
parser.add_argument('--continue_training', action='store_true', help='Continue training from previous checkpoint',
                    default=False)
parser.add_argument('--val_freq', type=int, default=1, help='validation frequency')
parser.add_argument('--vis_freq', type=int, default=50, help='frequency of save images to tensorboard')
parser.add_argument('--deterministic', type=int, default=0,
                    help='whether use deterministic training')
# TODO: update project id and data path
parser.add_argument(
    '--project', default="bitstrapped-gpu-testing",
    type=str, help="Project name"
)
parser.add_argument(
    '--data_path', default="bitstrapped-gpu-testing/pre-processed",
    type=str, help="Pre-processed data path"
)
parser.add_argument('--model_dir', dest='model_dir',
                    default=os.getenv('AIP_MODEL_DIR'), type=str, help='Model directory')
args = parser.parse_args()

fs = gcsfs.GCSFileSystem(project=args.project)


def init():
    """
    Initialize the 'train' folder.

    This function creates a 'train' folder in the specified root path and ensures it exists.

    Returns:
        None
    """
    folder = 'train'
    folder = f"{args.root_path}/{folder}"
    os.makedirs(folder, exist_ok=True)


# Call the 'init' function to create the 'train' folder
init()


def download_folder(raw_data, folder):
    """
    Download data from Google Cloud Storage to a local folder.

    Args:
        raw_data: The Google Cloud Storage path to the raw data.
        folder: The local folder where the data should be downloaded.

    Returns:
        None
    """
    download_command = f"gsutil -m cp -r gs://{raw_data}/{folder}/* {args.root_path}/{folder}"
    os.system(download_command)


download_folder(raw_data=args.data_path, folder='train')

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset

    snapshot_path = f"model/{args.exp}_epo{args.max_epochs}_bs{args.batch_size * args.n_gpu}_lrg{args.base_lr_g}_{args.lambda_self}_{args.lambda_cross}_{args.lambda_triplet}_{args.lambda_GAN}_vgg{args.lambda_perceptual}"
    os.makedirs(snapshot_path, exist_ok=True)

    config = get_config(args)
    G = generator(img_size=config.DATA.IMG_SIZE,
                  patch_size=config.MODEL.SWIN.PATCH_SIZE,
                  in_chans=config.MODEL.SWIN.IN_CHANS,
                  out_chans=config.MODEL.SWIN.OUT_CHANS,
                  embed_dim=config.MODEL.SWIN.EMBED_DIM,
                  depths=config.MODEL.SWIN.DEPTHS,
                  num_heads=config.MODEL.SWIN.NUM_HEADS,
                  window_size=config.MODEL.SWIN.WINDOW_SIZE,
                  mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                  qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                  qk_scale=config.MODEL.SWIN.QK_SCALE,
                  drop_rate=config.MODEL.DROP_RATE,
                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                  ape=config.MODEL.SWIN.APE,
                  patch_norm=config.MODEL.SWIN.PATCH_NORM,
                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                  seg=args.lambda_seg > 0,
                  seg_channel=args.seg_channel,
                  num_contrast=args.n_contrast,
                  cross_contrast_attn=not args.no_cross_contrast_attn).cuda()

    D = None
    if args.lambda_GAN > 0:
        # discriminator configurations
        config_d = ml_collections.ConfigDict()
        config_d.n_layer = 4
        config_d.gan_type = 'lsgan'
        config_d.dim = 64
        config_d.norm = 'bn'
        config_d.activ = 'lrelu'
        config_d.num_scales = 3
        config_d.pad_type = 'zero'
        config_d.input_dim = 1
        D = discriminator(config_d, n_contrast=4, label_smoothing=args.label_smoothing).cuda()

    trainer = {'BRATS': trainer_brats, 'IXI': trainer_ixi, 'Spine': trainer_spine}
    trainer[dataset_name](args, G, D, snapshot_path, args.model_dir)
