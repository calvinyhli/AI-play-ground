"""
@author:liyinghao
@homepage:https://github.com/calvinyhli
This code borrows heavily on the source code repo:https://github.com/huggingface/pytorch-image-models.
Specific algorithms and implementations were adapted and modified for this project.
"""
import math
from typing import Tuple, Optional, Union, Callable
import time
import torch
from torch import nn
from einops import rearrange
from itertools import repeat
from timm.layers.grid import ndgrid 
from timm.layers.weight_init import trunc_normal_
from timm.layers import Mlp, PatchEmbed, ClassifierHead, resize_rel_pos_bias_table

def window_partition(
        x: torch.Tensor,
        window_size: Tuple[int, int]
) -> torch.Tensor:
    B, H, W, C = x.shape

    x = x.view(B, H // window_size[0], window_size[1], W // window_size[1], window_size[1], C)
    x_windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    # x_win = rearrange(x, "b (h ws1) (w ws2) c -> (b h w) ws1 ws2 c", ws1=window_size[0], ws2=window_size[1])
    return x_windows

def window_reverse(
        windows: torch.Tensor,
        window_size: Tuple[int, int],
        H: int,
        W: int
) -> torch.Tensor:
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x

def get_relative_position_index(win_h: int, win_w: int):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(ndgrid(torch.arange(win_h), torch.arange(win_w)))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

class WindowAttention(nn.Module):
    """ 
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports shifted and non-shifted windows.
    """
    def __init__(
            self,
            dim: int,
            n_heads: int,
            head_dim: Optional[int]=None,
            window_size: Tuple[int, int]=(7,7),
            qkv_bias: bool=True,
            attn_drop: float=0.,
            proj_drop: float=0.
            ):
        super().__init__()
        """
        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            head_dim: Number of channels per head (dim // num_heads if not set)
            window_size: The height and width of the window.
            qkv_bias:  learnable bias to query, key, value if True
            attn_drop: Dropout ratio of attention weight.
            proj_drop: Dropout ratio of output.
        """
        self.dim = dim
        self.window_size = window_size
        win_h, win_w = self.window_size
        self.win_area = win_h * win_w
        self.n_heads = n_heads
        head_dim = head_dim or dim // n_heads
        attn_dim = head_dim * n_heads
        self.scale = head_dim ** -0.5
        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_pos_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), n_heads))
        self.register_buffer("relative_pos_index", get_relative_position_index(win_h, win_w), persistent=False)
        self.qkv = nn.Linear(dim, attn_dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_pos_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def set_window_size(self, window_size: Tuple[int, int]) -> None:
        """Update window size & interpolate position embeddings
        Args:
            window_size (int): New window size
        """
        if window_size == self.window_size:
            return
        self.window_size = window_size
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        with torch.no_grad():
            new_bias_shape = (2 * win_h - 1) * (2 * win_w - 1), self.n_heads
            self.relative_position_bias_table = nn.Parameter(
                resize_rel_pos_bias_table(
                    self.relative_position_bias_table,
                    new_window_size=self.window_size,
                    new_bias_shape=new_bias_shape,
            ))
            self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w), persistent=False)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        rel_pos_bias = self.relative_pos_bias_table[
            self.relative_pos_index.view(-1)].view(self.win_area, self.win_area, -1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
        return rel_pos_bias.unsqueeze(0)
        
    def forward(self, x, mask:Optional[torch.Tensor]=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        Bi, N, C = x.shape
        qkv = self.qkv(x).reshape(Bi, N, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        score = (q @ k.transpose(-2, -1)) / self.scale
        attn = score + self._get_rel_pos_bias()
        if mask is not None:
            attn = attn.view(-1, mask.shape[0], self.n_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        
        x = x.transpose(1, 2).reshape(Bi, N, -1)
        x = self.proj(x)
        X = self.proj_drop(x)

        return x
    
class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    """
    def __init__(self, 
            dim: int,
            input_resolution: Tuple[int, int],
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size: Tuple[int, int] = (7,7),
            shift_size: int = 0,
            always_partition: bool = False,
            dynamic_mask: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_prob: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            window_size: Window size.
            num_heads: Number of attention heads.
            head_dim: Enforce the number of channels per head
            shift_size: Shift size for SW-MSA.
            always_partition: Always partition into full windows and shift
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.target_shift_size = (shift_size, shift_size)  # store for later resize
        self.always_partition = always_partition
        self.dynamic_mask = dynamic_mask
        self.window_size, self.shift_size = self._calc_window_shift(window_size, shift_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.dropout1 = nn.Dropout(drop_prob)
        self.attn = WindowAttention(
            dim,
            n_heads=num_heads,
            head_dim=head_dim,
            window_size=self.window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.dropout2 = nn.Dropout(drop_prob)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )

    def get_attn_mask(self, x:Optional[torch.Tensor]=None) -> Optional[torch.Tensor]:
        if self.shift_size:
            H, W = self.input_resolution
            H = math.ceil(H / self.window_size[0]) * self.window_size[0]
            W = math.ceil(W / self.window_size[1]) * self.window_size[1]
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    (0, -self.window_size[0]),
                    (-self.window_size[0], -self.shift_size[0]),
                    (-self.shift_size[0], None),
            ):
                for w in (
                        (0, -self.window_size[1]),
                        (-self.window_size[1], -self.shift_size[1]),
                        (-self.shift_size[1], None),
                ):
                    img_mask[:, h[0]:h[1], w[0]:w[1], :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask
    
    def _calc_window_shift(
            self,
            target_window_size: Union[int, Tuple[int, int]],
            target_shift_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if target_shift_size is None:
            # if passed value is None, recalculate from default window_size // 2 if it was previously non-zero
            target_shift_size = self.target_shift_size
            if any(target_shift_size):
                target_shift_size = (target_window_size[0] // 2, target_window_size[1] // 2)
        else:
            target_shift_size = self.target_shift_size

        if self.always_partition:
            return target_window_size, target_shift_size

        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def set_input_size(
            self,
            feat_size: Tuple[int, int],
            window_size: Tuple[int, int],
            always_partition: Optional[bool] = None,
    ):
        """
        Args:
            feat_size: New input resolution
            window_size: New window size
            always_partition: Change always_partition attribute if not None
        """
        self.input_resolution = feat_size
        if always_partition is not None:
            self.always_partition = always_partition
        self.window_size, self.shift_size = self._calc_window_shift(window_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.attn.set_window_size(self.window_size)
        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )

    def _attn(self,x):
        B, H, W, C = x.shape
        if any(self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x
        # pad for resolution not divisible by window size
        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = shifted_x.shape
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        # W-MSA/SW-MSA
        if getattr(self, 'dynamic_mask', False):
            attn_mask = self.get_attn_mask(shifted_x)
        else:
            attn_mask = self.attn_mask
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
        shifted_x = shifted_x[:, :H, :W, :].contiguous()
        # reverse cyclic shift
        if any(self.shift_size):
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x
        return x
    
    def forward(self, x):
        B, H, W, C = x.shape
        x += self.dropout1(self._attn(self.norm1(x)))
        x = x.reshape(B, -1, C)
        x += self.dropout1(self.mlp(self.norm2(x)))
        x = x.reshape(B, H, W, C)
        return x

class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    """
    def __init__(
        self, 
        dim: int,
        o_dim: Optional[int] = None,
        norm_layer: Callable = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.o_dim = o_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.o_dim, bias=False)
    
    def forward(self, x):
        B, H, W, C = x.shape
        x = nn.functional.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        _, H, W, _ = x.shape

        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H // 2, W // 2, -1)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class SwinTransformerStage(nn.Module):
    """ 
    A basic Swin Transformer layer for one stage.
    """

    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: Tuple[int, int],
            window_size: Tuple[int, int],
            depth: int,
            downsample: bool = True,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            always_partition: bool = False,
            dynamic_mask: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_prob: float = 0.,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            downsample: Downsample layer at the end of the layer.
            num_heads: Number of attention heads.
            head_dim: Channels per head (dim // num_heads if not set)
            window_size: Local window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging layer
        if downsample:
            self.downsample = PatchMerging(
                dim=dim,
                o_dim=out_dim,
                norm_layer=norm_layer,
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # build blocks
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size[0],
                always_partition=always_partition,
                dynamic_mask=dynamic_mask,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_prob=drop_prob[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)])

    def set_input_size(
            self,
            feat_size: Tuple[int, int],
            window_size: int,
            always_partition: Optional[bool] = None,
    ):
        """ Updates the resolution, window size and so the pair-wise relative positions.

        Args:
            feat_size: New input (feature) resolution
            window_size: New window size
            always_partition: Always partition / shift the window
        """
        self.input_resolution = feat_size
        if isinstance(self.downsample, nn.Identity):
            self.output_resolution = feat_size
        else:
            self.output_resolution = tuple(i // 2 for i in feat_size)
        for block in self.blocks:
            block.set_input_size(
                feat_size=self.output_resolution,
                window_size=window_size,
                always_partition=always_partition,
            )

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x

class SwinTransformer(nn.Module):
    """ Swin Transformer
    A PyTorch implementation of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    """
    def __init__(
        self, 
        img_size: Tuple[int, int] = (224,224),
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = 'avg',
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        head_dim: Optional[int] = None,
        window_size: Tuple[int, int] = (7, 7),
        always_partition: bool = False,
        strict_img_size: bool = True,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        embed_layer: Callable = PatchEmbed,
        norm_layer: Union[str, Callable] = nn.LayerNorm,
        weight_init: str = '',
        **kwargs,
    ):
        super().__init__()
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.output_fmt = 'NHWC'

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = self.head_hidden_size = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_info = []

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
            strict_img_size=strict_img_size,
            output_fmt='NHWC',
        )
        patch_grid = self.patch_embed.grid_size

        # build layers
        # head_dim = to_ntuple(self.num_layers)(head_dim)
        head_dim = tuple(repeat(head_dim, self.num_layers))
        if not isinstance(window_size, (list, tuple)):
            window_size = tuple(repeat(self.num_layers)(window_size))
        elif len(window_size) == 2:
            window_size = (window_size,) * self.num_layers
        assert len(window_size) == self.num_layers
        mlp_ratio = tuple(repeat(mlp_ratio, self.num_layers))
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        layers = []
        in_dim = embed_dim[0]
        scale = 1
        for i in range(self.num_layers):
            out_dim = embed_dim[i]
            layers += [SwinTransformerStage(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(
                    patch_grid[0] // scale,
                    patch_grid[1] // scale
                ),
                depth=depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                head_dim=head_dim[i],
                window_size=window_size[i],
                always_partition=always_partition,
                dynamic_mask=not strict_img_size,
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_prob=dpr[i],
                norm_layer=norm_layer,
            )]
            in_dim = out_dim
            if i > 0:
                scale *= 2
            self.feature_info += [dict(num_chs=out_dim, reduction=patch_size * scale, module=f'layers.{i}')]
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            input_fmt=self.output_fmt,
        )

    def set_input_size(
            self,
            img_size: Optional[Tuple[int, int]] = None,
            patch_size: Optional[Tuple[int, int]] = None,
            window_size: Optional[Tuple[int, int]] = None,
            window_ratio: int = 8,
            always_partition: Optional[bool] = None,
    ) -> None:
        """ Updates the image resolution and window size.

        Args:
            img_size: New input resolution, if None current resolution is used
            patch_size (Optional[Tuple[int, int]): New patch size, if None use current patch size
            window_size: New window size, if None based on new_img_size // window_div
            window_ratio: divisor for calculating window size from grid size
            always_partition: always partition into windows and shift (even if window size < feat size)
        """
        if img_size is not None or patch_size is not None:
            self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)
            patch_grid = self.patch_embed.grid_size

        if window_size is None:
            window_size = tuple([pg // window_ratio for pg in patch_grid])

        for index, stage in enumerate(self.layers):
            stage_scale = 2 ** max(index - 1, 0)
            stage.set_input_size(
                feat_size=(patch_grid[0] // stage_scale, patch_grid[1] // stage_scale),
                window_size=window_size,
                always_partition=always_partition,
            )

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layers(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

if __name__ == "__main__":
    x = torch.randn((4, 3, 224, 224))
    net = SwinTransformer()
    output = net(x)
    print(output)
    print(output.shape)