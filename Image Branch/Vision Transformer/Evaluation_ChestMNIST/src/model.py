import os
import sys
import torch
import torch.nn as nn

# Inherit MAE definitions from Stage 2A DAPT 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DAPT/src')))
from mae import mae_vit_base_patch16

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=14, pretrained_path=None):
        super().__init__()
        # Load the base MAE model structure
        self.mae = mae_vit_base_patch16()
        
        # Load weights based on strategy
        if pretrained_path:
            if os.path.exists(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                msg = self.mae.load_state_dict(checkpoint['model'], strict=False)
                print(f"Loaded Pretrained DAPT Weights from: {pretrained_path}")
            else:
                print(f"Warning: Expected DAPT weights at {pretrained_path} but not found. Using ImageNet baseline.")
        else:
            print("Baseline Strategy: Using default ImageNet pretrained weights.")
            
        embed_dim = self.mae.cls_token.shape[2]
        
        # Append Supervised Classification Head
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)
        
    def forward(self, x):
        # forward_encoder returns (latent, mask, ids_restore)
        # Set mask_ratio=0.0 to process the entire image during classification
        latent, _, _ = self.mae.forward_encoder(x, mask_ratio=0.0)
        
        # latent is [B, L, D]. The 0th token is the appended cls_token
        cls_feature = latent[:, 0, :]
        return self.head(cls_feature)

def get_vit_classifier(num_classes=14, pretrained_path=None):
    return ViTClassifier(num_classes, pretrained_path)
