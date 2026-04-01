import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_b, Swin_B_Weights

class SimMIMSwinB(nn.Module):
    def __init__(self, encoder_stride=32):
        super().__init__()
        self.encoder_stride = encoder_stride
        self.in_chans = 3
        
        # Load ImageNet pretrained Swin-B natively
        self.encoder = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        
        # Swin-B outputs [B, 7, 7, 1024] (for 224x224 input)
        self.encoder_dim = self.encoder.norm.normalized_shape[0] # 1024
        
        # SimMIM lightweight decoder (1x1 conv mapping to pixels)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.encoder_dim, self.encoder_stride ** 2 * self.in_chans, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride)
        )

    def forward(self, x, mask):
        # We process the unmasked image purely natively through the ImageNet Swin-B
        z = self.encoder.features(x)
        z = self.encoder.norm(z)  # [B, 7, 7, 768]
        
        z = z.permute(0, 3, 1, 2).contiguous() # [B, 768, 7, 7]
        x_rec = self.decoder(z) # [B, 3, 224, 224]
        
        # Mask is [B, 56, 56] based on patch_size=4
        mask_expanded = mask.float().unsqueeze(1)
        mask_expanded = mask_expanded.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
        
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask_expanded).sum() / (mask_expanded.sum() + 1e-5)
        
        return loss, x_rec

def simmim_swin_base(**kwargs):
    return SimMIMSwinB(**kwargs)
