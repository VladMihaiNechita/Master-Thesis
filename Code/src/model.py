import torch
import torch.nn as nn

from .model_util import TransformerBlock, get_2d_sincos_pos_embed, random_masking


# B - Batch size
# C - Number of channels
# H - Height of the image
# P - Patch size
# W - Width of the image


class VisionTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_size = config["patch_size"]
        encoder_hidden_size = config["encoder_hidden_size"]
        grid_size = config["image_size"] // patch_size
        num_patches = grid_size ** 2

        # Patching the image
        self.projection = nn.Conv2d(in_channels=config["image_channels"], out_channels=encoder_hidden_size, 
                                    kernel_size=patch_size, stride=patch_size)
        nn.init.xavier_uniform_(self.projection.weight.view(self.projection.weight.shape[0], -1))
        
        # Class token
        self.cls_token = nn.Parameter(data=torch.zeros(1, 1, encoder_hidden_size), requires_grad=True)
        nn.init.normal_(self.cls_token, std=0.02)

        # Positional embedding
        self.pos_embed = nn.Parameter(data=torch.zeros(1, num_patches + 1, encoder_hidden_size), 
                                      requires_grad=False)

        patch_pos_embed = get_2d_sincos_pos_embed(grid_size, encoder_hidden_size)
        with torch.no_grad():
            self.pos_embed[:, 1:, :].copy_(patch_pos_embed)

        # Transformer Blocks
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size=encoder_hidden_size, mlp_ratio=config["encoder_mlp_ratio"], 
                                                      num_heads=config["encoder_num_heads"]) 
                                     for _ in range(config["encoder_num_layers"])])
        self.norm = nn.LayerNorm(encoder_hidden_size, eps=1e-6)
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)
        

    def forward(self, x, mask_ratio):
        # x: (B, C, H, W)
        
        x = self.projection(x)
        # x: (B, encoder_hidden_size, H/P, W/P)

        x = x.flatten(2)
        # [B, encoder_hidden_size, num_patches]

        x = x.transpose(1, 2)
        # [B, num_patches, encoder_hidden_size]

        x = x + self.pos_embed[:, 1:, :]
        # [B, num_patches, encoder_hidden_size]

        x, mask, ids_restore = random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        # [B, num_keep + 1, encoder_hidden_size]
    
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x, mask, ids_restore
    

    def forward_no_mask(self, x):
        # x: (B, C, H, W)

        x = self.projection(x)
        # x: (B, encoder_hidden_size, H/P, W/P)

        x = x.flatten(2)
        # [B, encoder_hidden_size, num_patches]

        x = x.transpose(1, 2)
        # [B, num_patches, encoder_hidden_size]

        x = x + self.pos_embed[:, 1:, :]
        # [B, num_patches, encoder_hidden_size]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        # [B, num_keep + 1, encoder_hidden_size]
    
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x
    

class VisionTransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_size = config["patch_size"]
        encoder_hidden_size = config["encoder_hidden_size"]
        decoder_hidden_size = config["decoder_hidden_size"]
        grid_size = config["image_size"] // patch_size
        num_patches = grid_size ** 2
        patch_dim = (patch_size ** 2) * config["image_channels"]

        # Projection
        self.decoder_embed = nn.Linear(in_features=encoder_hidden_size, out_features=decoder_hidden_size, bias=True)
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        nn.init.constant_(self.decoder_embed.bias, 0)
        
        # Mask token
        self.mask_token = nn.Parameter(data=torch.zeros(1, 1, decoder_hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional embedding
        self.decoder_pos_embed = nn.Parameter(data=torch.zeros(1, num_patches + 1, decoder_hidden_size), 
                                              requires_grad=False)
        
        patch_pos_embed = get_2d_sincos_pos_embed(grid_size, decoder_hidden_size)
        with torch.no_grad():
            self.decoder_pos_embed[:, 1:, :].copy_(patch_pos_embed)

        # Transformer Blocks
        self.decoder_blocks = nn.ModuleList([TransformerBlock(hidden_size=decoder_hidden_size, mlp_ratio=config["decoder_mlp_ratio"], 
                                                              num_heads=config["decoder_num_heads"]) 
                                            for _ in range(config["decoder_num_layers"])])

        self.decoder_norm = nn.LayerNorm(decoder_hidden_size, eps=1e-6)
        nn.init.constant_(self.decoder_norm.bias, 0)
        nn.init.constant_(self.decoder_norm.weight, 1.0)

        # Reconstruction prediction head
        self.decoder_pred = nn.Linear(in_features=decoder_hidden_size, out_features=patch_dim, bias=True)
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        nn.init.constant_(self.decoder_pred.bias, 0)


    def forward(self, latent, ids_restore):
        x = self.decoder_embed(latent)

        mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1)

        x_without_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_without_cls = torch.gather(x_without_cls, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        x = torch.cat([x[:, :1, :], x_without_cls], dim=1)
        x = x + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)
        # [B, num_patches + 1, hidden_size]

        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # remove CLS token

        return x
    

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config["patch_size"]
        self.image_channels = config["image_channels"]
        self.norm_pix_loss = config["norm_pix_loss"]

        self.encoder = VisionTransformerEncoder(config)
        self.decoder = VisionTransformerDecoder(config)
        

    def patchify(self, imgs):
        # imgs: [B, C, H, W]
        B, C, H, W = imgs.shape
        p = self.patch_size

        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(B, h * w, p ** 2 * C)
        # x: [B, num_patches, patch_size * patch_size * C]

        return x


    def forward_loss(self, imgs, pred, mask):
        # imgs: [B, C, H, W]
        # pred: [B, num_patches, patch_size * patch_size * C]
        # mask: [B, num_patches], 0 is keep, 1 is remove

        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        mse_loss = ((pred - target) ** 2).mean(dim=-1)
        mse_loss = (mse_loss * mask).sum() / mask.sum()

        return mse_loss


    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.encoder(x=imgs, mask_ratio=mask_ratio)
        pred = self.decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


class VisionTransformerClassifier(nn.Module):  # Unused
    def __init__(self, config, num_classes):
        super().__init__()

        self.encoder = VisionTransformerEncoder(config)

        hidden_size = config["encoder_hidden_size"]
        self.head = nn.Sequential(nn.BatchNorm1d(hidden_size, affine=False, eps=1e-6), nn.Linear(hidden_size, num_classes))
        nn.init.normal_(self.head[1].weight, std=0.01)
        nn.init.constant_(self.head[1].bias, 0)


    def forward(self, x):
        x = self.encoder.forward_no_mask(x)
        #x = x[:, 0]  # CLS token
        x = x[:, 1:].mean(dim=1)  # Mean of all tokens

        x = self.head(x)
        return x


class VisionTransformerClassifierHead(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        hidden_size = config["encoder_hidden_size"]
        self.head = nn.Sequential(nn.BatchNorm1d(hidden_size, affine=False, eps=1e-6), nn.Linear(hidden_size, num_classes))
        nn.init.normal_(self.head[1].weight, std=0.01)
        nn.init.constant_(self.head[1].bias, 0)


    def forward(self, x):
        x = self.head(x)
        return x


class FineTuningModel(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.encoder = VisionTransformerEncoder(config)
        del self.encoder.norm

        hidden_size = config["encoder_hidden_size"]
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=2e-5)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, images):
        # [B, C, H, W]
        
        x = self.encoder.projection(images).flatten(2).transpose(1, 2)
        # [B, num_patches, encoder_hidden_size]
        
        x = x + self.encoder.pos_embed[:, 1:, :]  # Adds a positional embedding
        # [B, num_patches, encoder_hidden_size]

        # Constructs the CLS token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]

        # Add the CLS token to the beginning of the sequence
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        # [B, num_patches + 1, encoder_hidden_size]

        for block in self.encoder.blocks:
            x = block(x)

        #x = x[:, 0]  # CLS token
        x = x[:, 1:].mean(dim=1)  # Mean of all tokens
        return self.head(self.norm(x))
