import os

import torch

from configs.config import get_config
from model_evaluation import evaluator_ixi
from networks.mmt import MMT as generator


def load_model(args, device):
    # Create the model
    config = get_config(args)
    G = generator(
        img_size=config.DATA.IMG_SIZE,
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
        seg=args.seg,
        num_contrast=args.n_contrast,
        cross_contrast_attn=not args.no_cross_contrast_attn
    ).to(device)

    # Load weights
    state_dict = torch.load(os.path.join(args.model_path, args.ckpt), map_location='cpu')
    G.load_state_dict(state_dict['G'])

    # Set the model to evaluation mode (not training mode)
    G.eval()
    print("Model loaded")
    return G


def predict_with_model(model, args, input_combination, targets, folder_name):
    # Get predictions
    master_list = evaluator_ixi(args, model, input_combination, targets, folder_name)
    return master_list
