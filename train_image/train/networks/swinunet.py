# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from torch.utils import checkpoint

from .transformer import WindowAttention, Mlp, window_partition, cyclic_shift, window_reverse, PatchMerging, PatchExpand,PatchEmbed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
logger = logging.getLogger(__name__)


class SwinUNetBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (tuple): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        cross_contrast_attn (bool, optional): If True, attention is computed across multiple input contrasts, else the contrast dimension is collapsed
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=(5, 6), shift_size=(0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, cross_contrast_attn=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if self.input_resolution[0] <= self.window_size[0] or self.input_resolution[1] <= self.window_size[1]:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = (0, 0)
            self.window_size = self.input_resolution
        assert 0 <= self.shift_size[0] < self.window_size[0] and 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.cross_contrast_attn = cross_contrast_attn

        if min(self.shift_size) > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask.unsqueeze(1), self.window_size)  # nW, 1, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, window_size**2, window_size**2
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def with_pos_embed(self, tensor, pos=None):
        if pos is not None:
            n_contrast = pos.shape[0]
            return tensor + pos.view(1, n_contrast, 1, 1, -1)
        return tensor

    def forward(self, x, contrast_embed=None):
        # x: input (query); x_kv: decoder outputs
        # H, W = self.input_resolution
        B, n_contrast, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)  # B, n_contrast, H, W, C

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = cyclic_shift(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, n_contrast, window_size, window_size, C

        x_windows_embedded = self.with_pos_embed(x_windows, contrast_embed)
        # W-MSA/SW-MSA

        if self.cross_contrast_attn:
            attn_windows, _ = self.attn(x_q=x_windows_embedded, x_k=x_windows_embedded, x_v=x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size*n_contrast, C
        else:
            x_windows = x_windows.view(
                x_windows.shape[0] * n_contrast, 1, self.window_size[0], self.window_size[1], C
            )
            x_windows_embedded = x_windows_embedded.view(
                x_windows_embedded.shape[0] * n_contrast, 1, self.window_size[0],
                self.window_size[1], C
            )
            attn_windows, _ = self.attn(x_q=x_windows_embedded, x_k=x_windows_embedded, x_v=x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, n_contrast, self.window_size[0], self.window_size[1], C) # nW*B, n_contrast, window_size, window_size, C

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B n_contrast H' W'  C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = cyclic_shift(shifted_x, shifts=self.shift_size, dims=(2, 3))
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))   # B, n_contrast, H, W, C

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class SwinUNetEncoderLayer(nn.Module):
    """ A basic Swin UNet Encoder layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at beginning of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        cross_contrast_attn (bool): If True, attention is computed across multiple input contrasts, else the contrast dimension is collapsed
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 cross_contrast_attn=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.cross_contrast_attn = cross_contrast_attn

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample((input_resolution[0]*2, input_resolution[1]*2), dim=dim//2, norm_layer=norm_layer)
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            SwinUNetBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0,0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2),
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, cross_contrast_attn=cross_contrast_attn)
            for i in range(depth)])



    def forward(self, x, contrast_embed=None):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, contrast_embed)
            else:
                x = blk(x, contrast_embed)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class SwinUNetDecoderLayer(nn.Module):
    """ A basic Swin UNet Decoder layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        cross_contrast_attn (bool): If True, attention is computed across multiple input contrasts, else the contrast dimension is collapsed
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False, concat=True, cross_contrast_attn=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.cross_contrast_attn = cross_contrast_attn

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample((input_resolution[0]//2, input_resolution[1]//2), dim=dim*2, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

        if concat:
            self.concat_linear = nn.Linear(2 * dim, dim)
        else:
            self.concat_linear = None
        # build blocks
        self.blocks = nn.ModuleList([
            SwinUNetBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0,0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2),
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, cross_contrast_attn=cross_contrast_attn)
            for i in range(depth)])

    def forward(self, x, x_enc=None, contrast_embed=None):
        if self.upsample is not None:
            x = self.upsample(x)
        if self.concat_linear and x_enc is not None:
            x = torch.cat([x, x_enc], -1)
            x = self.concat_linear(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, contrast_embed)
            else:
                x = blk(x, contrast_embed)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class SwinUNet(nn.Module):
    r""" SwinUNet
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        cross_contrast_attn (bool): If True, attention is computed across multiple input contrasts, else the contrast dimension is collapsed
    """

    def __init__(self, img_size=(160, 192), patch_size=4, out_chans=1,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=(5, 6), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, ce=True,
                 use_checkpoint=False, final_upsample="expand_first", num_contrast=4,
                 cross_contrast_attn=True, **kwargs):
        super().__init__()

        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.ce = ce  # add contrast embeddings
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.n_contrast = num_contrast
        self.cross_contrast_attn = cross_contrast_attn

        patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1]
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers_enc = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinUNetEncoderLayer(dim=int(embed_dim * 2 ** i_layer),
                                 input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                   patches_resolution[1] // (2 ** i_layer)),
                                 depth=depths[i_layer],
                                 num_heads=num_heads[i_layer],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                 norm_layer=norm_layer,
                                 downsample=PatchMerging if (i_layer > 0) else None,
                                 use_checkpoint=use_checkpoint,
                                 cross_contrast_attn=cross_contrast_attn)
            self.layers_enc.append(layer)

        # build decoder layers
        self.layers_dec = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinUNetDecoderLayer(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                 input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                   patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                 depth=depths[(self.num_layers - 1 - i_layer)],
                                 num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                     depths[:(self.num_layers - 1 - i_layer) + 1])],
                                 norm_layer=norm_layer,
                                 upsample=PatchExpand if (i_layer > 0) else None,
                                 concat = i_layer > 0,
                                 use_checkpoint=use_checkpoint,
                                 cross_contrast_attn=cross_contrast_attn)
            self.layers_dec.append(layer)

        # self.norm = norm_layer(self.num_features)
        # self.norm_up= norm_layer(self.embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder
    def forward_encoder(self, x, contrast_embeds=None):
        # transformer encoder processing
        enc_out = []

        for contrast_embed, layer in zip(contrast_embeds, self.layers_enc):
            x = layer(x, contrast_embed=contrast_embed)
            enc_out.append(x)

        return enc_out

    # Decoder
    def forward_decoder(self, enc_out, contrast_embeds=None):
        dec_out = []
        x = enc_out[-1]
        for i, layer_dec in enumerate(self.layers_dec):
            x = layer_dec(x, enc_out[self.num_layers - 1 - i], contrast_embed=contrast_embeds[self.num_layers - 1 - i])
            dec_out.append(x)
        return dec_out

    def forward(self, x, contrast_embeds):
        enc_out = self.forward_encoder(x, contrast_embeds=contrast_embeds) # list of skip connections
        dec_out = self.forward_decoder(enc_out, contrast_embeds=contrast_embeds) # multi-scale outputs
        return dec_out

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
