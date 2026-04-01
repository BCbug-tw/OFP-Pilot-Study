import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class MaskedAutoencoderViT(nn.Module):
    """ Wrapped MAE Architecture taking torchvision ImageNet weights natively """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16):
        super().__init__()
        
        self.patch_size = patch_size
        
        # Load ImageNet pretrained ViT
        self.vit = vit_b_16(weights='IMAGENET1K_V1')
        self.cls_token = self.vit.class_token
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size)**2 + 1, decoder_embed_dim))
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, nhead=decoder_num_heads, 
            dim_feedforward=decoder_embed_dim*4, activation="gelu", 
            batch_first=True, norm_first=True
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth, enable_nested_tensor=False)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * 3, bias=True)
        
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # We process it through the torchvision vit patch embedding
        x = self.vit.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # add positional embedding
        x = x + self.vit.encoder.pos_embedding[:, 1:, :] 
        
        # masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # append cls token
        cls_token = self.vit.class_token + self.vit.encoder.pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # run through encoder blocks
        for layer in self.vit.encoder.layers:
            x = layer(x)
            
        x = self.vit.encoder.ln(x)
        return x, mask, ids_restore
        
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss
        
    def patchify(self, imgs):
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.decoder_pred(self.decoder_norm(self.decoder_blocks(self.decoder_embed(latent))))
        
        # Real forward_decoder applies positional embeddings properly. Let's use the method above.
        pred_full = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred_full, mask)
        
        return loss, pred_full, mask

def mae_vit_base_patch16(**kwargs):
    return MaskedAutoencoderViT(**kwargs)
