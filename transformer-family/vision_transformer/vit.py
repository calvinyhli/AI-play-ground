import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Attention(nn.Module):
    """
    input: [B, S, D]
    """
    def __init__(self, dim, n_heads, head_dim, dropout=0.) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.dim = dim
        self.n_heads = n_heads
        self.proj_dim = n_heads * head_dim
        self.to_qkv = nn.Linear(self.dim, self.proj_dim*3)
        self.proj_layer = nn.Linear(self.proj_dim, self.dim) if (self.proj_dim != self.dim) else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        # split for multi-head
        q, k, v = map(lambda t: rearrange(t, "b s (n d) -> b n s d", n = self.n_heads), qkv)
        # compute self-attention
        score = F.softmax((q @ k.transpose(-1, -2)) / math.sqrt(self.proj_dim), dim=-1) @ v

        out = rearrange(score, "b n s d -> b s (n d)")
        out = self.proj_layer(out)

        return out

class MLP(nn.Module):
    def __init__(self, dim, dim_h, dropout=0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_h, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.mlp(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, n_heads, head_dim, dim_h, dropout=0.1) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Attention(dim, n_heads, head_dim, dropout=0.), 
                                      MLP(dim, dim_h, dropout=0.1)]))
        
    def forward(self, x):    
        for attn, mlp in self.layers:
            x = attn(x) + x
            x = mlp(x) + x
        return self.layernorm(x)

class ViT(nn.Module):
    """
    input: [B, H, W, C]
    """
    def __init__(self, img_size, patch_size, num_classes, dim, depth, n_heads, head_dim, dim_h, channel=3, pool="cls", dropout=0.) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "Images size must be divisible by patch size"
        num_patchs = (img_size // patch_size) * (img_size // patch_size)
        patch_dim = patch_size * patch_size * channel
        assert pool in {"cls", "mean"}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patchs+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.transformerencoder = TransformerEncoder(
            depth = depth,
            dim = dim,
            n_heads = n_heads,
            head_dim = head_dim,
            dim_h = dim_h,
            dropout = dropout
        )
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head  = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_token = repeat(self.cls_token, "1 1 d -> b 1 d", b = x.shape[0])
        x = torch.concat((cls_token, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformerencoder(x)
        x = x[:, 0] if self.pool == "cls" else x.mean(dim=1)
        x = self.to_latent(x)
        out = self.mlp_head(x)
        return out
    
if __name__ == "__main__":
    transformerencoder = TransformerEncoder(depth=6, 
                                            dim=128, 
                                            n_heads=4, 
                                            head_dim=32, 
                                            dim_h=256, 
                                            dropout=0.1)
    
    print(transformerencoder)

    net = ViT(img_size=32, 
              patch_size=4, 
              num_classes=10, 
              dim=128, 
              depth=6, 
              n_heads=4, 
              head_dim=32, 
              dim_h=256)
    imgs = torch.randn((4, 3, 32, 32))
    out  = net(imgs)
    print(out)
    print(out.shape)