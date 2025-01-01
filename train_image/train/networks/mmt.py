import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .swinunet import SwinUNet
from .transformer import WindowAttention, Mlp, window_partition, cyclic_shift, window_reverse, FinalPatchExpand_X4, \
    PatchExpand


class ResBlock(nn.Module):
    def __init__(
            self, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Head, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7,
                                             padding=3),
                                   ResBlock(out_channels, 5, act=nn.ReLU(True)),
                                   ResBlock(out_channels, 5, act=nn.ReLU(True)))

    def forward(self, x):
        return self.model(x)


class Tail(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True):
        super(Tail, self).__init__()
        self.model = nn.Sequential(ResBlock(in_channels, 5, act=nn.ReLU(True)),
                                   ResBlock(in_channels, 5, act=nn.ReLU(True)),
                                   nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7,
                                             padding=3),
                                   nn.ReLU() if relu else nn.Identity())

    def forward(self, x):
        return self.model(x)


class MMTDecoderBlock(nn.Module):
    r""" MMT Transformer Decoder Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        cross_contrast_attn (bool): If True, attention is computed across multiple input contrasts, else the contrast dimension is collapsed
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, cross_contrast_attn=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.cross_contrast_attn = cross_contrast_attn
        if self.input_resolution[0] <= self.window_size[0] or self.input_resolution[1] <= self.window_size[1]:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = (0, 0)
            self.window_size = self.input_resolution
        assert 0 <= self.shift_size[0] < self.window_size[0] and 0 <= self.shift_size[1] < self.window_size[
            1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.self_attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

            mask_windows = window_partition(img_mask.unsqueeze(1),
                                            self.window_size)  # nW, 1, window_size, window_size, 1
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

    def forward(self, x, x_kv, contrast_embed=None, contrast_token=None, return_attention=False):
        # x: input (query); x_kv: decoder outputs
        # H, W = self.input_resolution

        B, n_contrast, H, W, C = x.shape

        ###### self attn
        shortcut = x
        x = self.norm1(x)  # B, n_contrast, H, W, C

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = cyclic_shift(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, n_contrast, window_size, window_size, C

        # W-MSA/SW-MSA
        x_windows_embedded = self.with_pos_embed(x_windows, contrast_token)

        if self.cross_contrast_attn:
            attn_windows, _ = self.self_attn(x_q=x_windows_embedded, x_k=x_windows_embedded, x_v=x_windows,
                                             mask=self.attn_mask)  # nW*B, window_size*window_size*n_contrast, C
        else:
            x_windows = x_windows.view(
                x_windows.shape[0] * n_contrast, 1, self.window_size[0], self.window_size[1], C
            )
            x_windows_embedded = x_windows_embedded.view(
                x_windows_embedded.shape[0] * n_contrast, 1, self.window_size[0],
                self.window_size[1], C
            )
            attn_windows, _ = self.self_attn(x_q=x_windows_embedded, x_k=x_windows_embedded, x_v=x_windows,
                                             mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, n_contrast, self.window_size[0], self.window_size[1],
                                         C)  # nW*B, n_contrast, window_size, window_size, C

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B n_contrast H' W'  C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = cyclic_shift(shifted_x, shifts=self.shift_size, dims=(2, 3))
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)

        ##### cross attn
        shortcut = x
        x = self.norm2(x)  # B, n_contrast, H, W, C

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = cyclic_shift(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            shifted_x_kv = cyclic_shift(x_kv, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
        else:
            shifted_x = x
            shifted_x_kv = x_kv

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, n_contrast, window_size, window_size, C
        x_kv_windows = window_partition(shifted_x_kv, self.window_size)  # nW*B, n_contrast, window_size, window_size, C
        # W-MSA/SW-MSA

        attn_windows, attn = self.cross_attn(x_q=self.with_pos_embed(x_windows, contrast_token),
                                             x_k=self.with_pos_embed(x_kv_windows, contrast_embed),
                                             x_v=x_kv_windows, mask=self.attn_mask,
                                             return_attention=return_attention)  # nW*B, window_size*window_size*n_contrast, C

        # merge windows
        attn_windows = attn_windows.view(-1, n_contrast, self.window_size[0], self.window_size[1],
                                         C)  # nW*B, n_contrast, window_size, window_size, C

        # attn:  (nW*B, nH, Wh*Ww*n_contrast_q, Wh*Ww**n_contrast_kv)
        if return_attention:
            # sum over multi-heads
            attn = torch.sum(attn, dim=1)  # (nW*B, Wh*Ww*n_contrast_q, Wh*Ww**n_contrast_kv)
            # sum over local window (assume n_contrast_q=1)
            attn = torch.sum(attn, dim=1)  # (nW*B, Wh*Ww**n_contrast_kv)
            n_contrast_kv = x_kv.shape[1]
            attn = attn.view(-1, n_contrast_kv, self.window_size[0], self.window_size[1],
                             1)  # (nW*B, n_contrast_kv, window_size, window_size, 1

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B n_contrast H' W'  C
        if return_attention:
            shifted_attn = window_reverse(attn, self.window_size, H, W)  # B n_contrast_kv H' W' 1

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = cyclic_shift(shifted_x, shifts=self.shift_size, dims=(2, 3))
            attn = cyclic_shift(shifted_attn, shifts=self.shift_size, dims=(2, 3)) if return_attention else None
        else:
            x = shifted_x
            attn = shifted_attn if return_attention else None

        x = shortcut + self.drop_path(x)

        ##### FFN
        x = x + self.drop_path(self.mlp(self.norm3(x)))  # B, n_contrast, H, W, C

        return x, attn

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


class MMTDecoderLayer(nn.Module):
    """ A basic MMT Decoder layer for one stage.

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
        upsample (nn.Module | None, optional): Downsample layer at beginning of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        cross_contrast_attn (bool): If True, attention is computed across multiple input contrasts, else the contrast dimension is collapsed
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 cross_contrast_attn=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.cross_contrast_attn = cross_contrast_attn

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample((input_resolution[0] // 2, input_resolution[1] // 2), dim=dim * 2, dim_scale=2,
                                     norm_layer=norm_layer)
        else:
            self.upsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            MMTDecoderBlock(dim=dim, input_resolution=input_resolution,
                            num_heads=num_heads, window_size=window_size,
                            shift_size=(0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2),
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop, attn_drop=attn_drop,
                            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                            norm_layer=norm_layer, cross_contrast_attn=cross_contrast_attn)
            for i in range(depth)])

    def forward(self, x, x_kv, contrast_embed=None, contrast_token=None, return_attention=False):
        if self.upsample is not None:
            x = self.upsample(x)
        att_maps = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x, att_map = checkpoint.checkpoint(blk, x, x_kv, contrast_embed, contrast_token, return_attention)
                att_maps.append(att_map)
            else:
                x, att_map = blk(x, x_kv, contrast_embed, contrast_token, return_attention)
                att_maps.append(att_map)
        return x, att_maps


class MMT(nn.Module):
    r""" MMT Transformer

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

    def __init__(self, img_size=224, patch_size=4, in_chans=3, out_chans=1,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, ce=True,
                 use_checkpoint=False, final_upsample="expand_first", num_contrast=4, seg=False, seg_channel=3,
                 cross_contrast_attn=True, return_indiv_dec_outs=False, **kwargs):
        super().__init__()

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{}".format(depths,
                                                                                                          depths_decoder,
                                                                                                          drop_path_rate))

        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.ce = ce  # add contrast embeddings
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.n_contrast = num_contrast
        self.cross_contrast_attn = cross_contrast_attn
        self.return_indiv_dec_outs = return_indiv_dec_outs
        self.img_size = img_size

        print('CROSS CONTRAST ATTN=', cross_contrast_attn)

        self.heads = nn.ModuleList()
        conv_ch_num = int(embed_dim / patch_size ** 2)
        if conv_ch_num == 1:
            conv_ch_num = 3

        for _ in range(self.n_contrast):
            head = Head(in_channels=1, out_channels=conv_ch_num)
            self.heads.append(head)

        self.tails = nn.ModuleList()
        for _ in range(self.n_contrast):
            tail = Tail(in_channels=conv_ch_num, out_channels=1)
            self.tails.append(tail)
        if seg:
            self.tails.append(Tail(in_channels=conv_ch_num, out_channels=seg_channel, relu=False))

        patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1]
        self.patches_resolution = patches_resolution
        self.patch_size = patch_size

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.contrast_embeds = nn.ModuleList([nn.Embedding(num_contrast, int(embed_dim * 2 ** i))
                                              for i in range(self.num_layers)])
        self.contrast_tokens = nn.ModuleList([nn.Embedding(num_contrast + 1 if seg else num_contrast,
                                                           int(embed_dim * 2 ** i)) for i in range(self.num_layers)])

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.encoder = SwinUNet(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                out_chans=out_chans, embed_dim=embed_dim, depths=depths,
                                num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate, ape=ape, patch_norm=patch_norm,
                                use_checkpoint=use_checkpoint,
                                cross_contrast_attn=cross_contrast_attn)

        # build transformer decoder layers
        self.layers_dec = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MMTDecoderLayer(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
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
                                    use_checkpoint=use_checkpoint,
                                    cross_contrast_attn=cross_contrast_attn)
            self.layers_dec.append(layer)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size[0] // patch_size, img_size[1] // patch_size),
                                          dim_scale=patch_size,
                                          dim=embed_dim // patch_size ** 2)

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

    def encode_imgs(self, img_list, contrasts):
        img_codes = []
        for contrast, img in zip(contrasts, img_list):
            x = self.heads[contrast](img)
            img_codes.append(x)
        return img_codes

    def merge_codes(self, img_codes):
        embedding_outputs = []
        for x in img_codes:
            x = rearrange(x, 'b c (h p1) (w p2) -> b 1 h w (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
            embedding_outputs.append(x)
        embeddings = torch.cat(embedding_outputs, 1)  # (B, n_contrast, h, w, c)
        return embeddings

    # Encoder
    def forward_encoder(self, img_list, inputs, contrast_embeds=None):
        img_codes = self.encode_imgs(img_list, inputs)  # list of B, 16, H, W
        x = self.merge_codes(img_codes)  # B, n_contrast, h, w, c
        enc_out = self.encoder(x, contrast_embeds)
        return enc_out

    # Decoder
    def forward_decoder(self, x, enc_out, contrast_embeds=None, contrast_tokens=None, return_attention=False):
        att_maps = []
        dec_outs = []
        for i, layer_dec in enumerate(self.layers_dec):
            x, att_map = layer_dec(x, enc_out[i], contrast_embed=contrast_embeds[self.num_layers - 1 - i],
                                   contrast_token=contrast_tokens[self.num_layers - 1 - i],
                                   return_attention=return_attention)
            dec_outs += x
            att_maps += att_map
        if self.return_indiv_dec_outs:
            return x, att_maps, dec_outs
        return x, att_maps

    def up_x4(self, x):
        B, n_contrast, H, W, C = x.shape
        x = self.up(x)
        output = []
        for i in range(n_contrast):
            if self.final_upsample == "expand_first":
                x_i = x[:, i, :, :, :]
                output.append(x_i.permute(0, 3, 1, 2))
        return output

    def forward(self, x, inputs, outputs, return_attention=False):
        contrast_embeds = [self.contrast_embeds[i].weight[inputs] for i in range(self.num_layers)]
        enc_out = self.forward_encoder(x, inputs, contrast_embeds=contrast_embeds)
        img_outputs = []
        att_maps = []
        indiv_dec_outs = []
        for output in outputs:
            contrast_tokens = [self.contrast_tokens[i].weight[[output]] for i in range(self.num_layers)]
            B, _, H, W, C = enc_out[0].shape
            tgt = torch.zeros(B, 1, H, W, C).cuda()
            if self.return_indiv_dec_outs:
                x, att_map, dec_outs = self.forward_decoder(tgt, enc_out, contrast_embeds=contrast_embeds,
                                                            contrast_tokens=contrast_tokens,
                                                            return_attention=return_attention)
                indiv_dec_outs += dec_outs
            else:
                x, att_map = self.forward_decoder(tgt, enc_out, contrast_embeds=contrast_embeds,
                                                  contrast_tokens=contrast_tokens,
                                                  return_attention=return_attention)
            att_maps += att_map
            x = self.up_x4(x)
            img_outputs.append(self.tails[output](x[0]))
        if self.return_indiv_dec_outs:
            return img_outputs, enc_out, att_maps, indiv_dec_outs
        return img_outputs, enc_out, att_maps

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
