import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

class PseudoSparseResNet50(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Load standard ResNet-50 architecture with ImageNet weights
        self.encoder = resnet50(weights='IMAGENET1K_V1')
        
        # Remove the classification head and average pooling layer
        self.encoder.fc = nn.Identity()
        self.encoder.avgpool = nn.Identity()
        
        self.encoder_stride = 32
        self.in_chans = 3
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2),  
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),   
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),   
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),    
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.in_chans, kernel_size=2, stride=2)    
        )

    def apply_pseudo_sparse_mask(self, x, mask):
        B, C, H, W = x.shape
        mask_scaled = mask.float().unsqueeze(1) 
        mask_scaled = F.interpolate(mask_scaled, size=(H, W), mode='nearest')
        visible_mask = (1.0 - mask_scaled).to(x.device)
        return x * visible_mask

    def forward_features(self, x, mask):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.apply_pseudo_sparse_mask(x, mask)
        
        x = self.encoder.layer1(x)
        x = self.apply_pseudo_sparse_mask(x, mask)
        
        x = self.encoder.layer2(x)
        x = self.apply_pseudo_sparse_mask(x, mask)
        
        x = self.encoder.layer3(x)
        x = self.apply_pseudo_sparse_mask(x, mask)
        
        x = self.encoder.layer4(x)
        return x

    def forward(self, x, mask):
        z = self.forward_features(x, mask) 
        x_rec = self.decoder(z) 
        loss = self.forward_loss(x, x_rec, mask)
        return loss, x_rec

    def forward_loss(self, x, x_rec, mask):
        mask_expanded = mask.float().unsqueeze(1) 
        mask_expanded = F.interpolate(mask_expanded, size=(x.shape[2], x.shape[3]), mode='nearest') 
        mask_expanded = mask_expanded.expand(-1, 3, -1, -1) 
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask_expanded).sum() / (mask_expanded.sum() + 1e-5) 
        return loss

def spark_resnet50(**kwargs):
    model = PseudoSparseResNet50(**kwargs)
    return model
