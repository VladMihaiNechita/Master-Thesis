
import torch
import torch.nn as nn


# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed_sincos.py
def get_2d_sincos_pos_embed(grid_size, hidden_size):
    # pos_dim -> Number of frequency values used for one coordinate direction (x or y)
    # omega -> List of frequencies
    
    # assert hidden_size % 4 == 0
    pos_dim = hidden_size // 4  
    
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (10000 ** omega)
    # This creates frequencies smoothly spaced on a logarithmic scale between 1 and roughly 1/10000

    y, x = torch.meshgrid(torch.arange(grid_size, dtype=torch.float32), 
                          torch.arange(grid_size, dtype=torch.float32), 
                          indexing="ij")

    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    y = y.reshape(-1, 1) * omega.reshape(1, -1)

    return torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)


@torch.compiler.disable
def random_masking(x, mask_ratio):
    # Compiling argsort creates an extremely slow Triton kernel; keep masking eager.
    # x: [B, num_patches, hidden_size]

    B, num_patches, hidden_size = x.shape
    num_keep = int(num_patches * (1 - mask_ratio))

    noise = torch.rand(B, num_patches, device=x.device)

    ids_shuffle = torch.argsort(noise, dim=1)
    shuffle_positions = torch.arange(num_patches, device=x.device).expand(B, -1)
    ids_restore = torch.empty_like(ids_shuffle).scatter_(1, ids_shuffle, shuffle_positions)
    ids_keep = ids_shuffle[:, :num_keep]

    x = torch.gather(input=x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, hidden_size))
    # x: [B, num_keep, hidden_size]

    mask = torch.ones(B, num_patches, device=x.device)
    mask[:, :num_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    # mask: [B, num_patches], 0 is keep, 1 is remove

    return x, mask, ids_restore


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = probability


    def forward(self, x):
        if self.probability == 0.0 or not self.training:
            return x

        keep_probability = 1 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_probability)
        return x * mask / keep_probability


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, mlp_ratio, num_heads, drop_path_rate=0.0):
        super().__init__()
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.drop_path = DropPath(drop_path_rate)

        # LayerNorm 1
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        nn.init.constant_(self.norm1.bias, 0)
        nn.init.constant_(self.norm1.weight, 1.0)

        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, bias=True, batch_first=True)
        nn.init.xavier_uniform_(self.attention.in_proj_weight)  # Shape: (3 * hidden_size, hidden_size)
        nn.init.constant_(self.attention.in_proj_bias, 0)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)  # Shape: (hidden_size, hidden_size)
        nn.init.constant_(self.attention.out_proj.bias, 0)

        # LayerNorm 2
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        nn.init.constant_(self.norm2.bias, 0)
        nn.init.constant_(self.norm2.weight, 1.0)

        # MLP block
        self.mlp = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=mlp_hidden_size, bias=True), 
                                 nn.GELU(), 
                                 nn.Linear(in_features=mlp_hidden_size, out_features=hidden_size, bias=True))
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[0].bias, 0)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[2].bias, 0)


    def forward(self, x):
        y = self.norm1(x)  # LayerNorm
        y, _ = self.attention(y, y, y, need_weights=False)  # Multi-Head Self-Attention
        x = x + self.drop_path(y)  # Residual connection

        y = self.norm2(x)  # LayerNorm
        y = self.mlp(y)  # MLP block
        x = x + self.drop_path(y)  # Residual connection

        return x
