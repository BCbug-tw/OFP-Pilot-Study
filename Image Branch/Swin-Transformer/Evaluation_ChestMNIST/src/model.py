import os
import sys
import torch
import torch.nn as nn

# Import SimMIM Architecture from Stage 2A DAPT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DAPT/src')))
from simmim import simmim_swin_base

class SwinClassifier(nn.Module):
    def __init__(self, num_classes=14, pretrained_path=None):
        super().__init__()
        # Load the base SimMIM model structure
        self.simmim = simmim_swin_base()
        
        # Load weights based on strategy
        if pretrained_path:
            if os.path.exists(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                msg = self.simmim.load_state_dict(checkpoint['model'], strict=False)
                print(f"Loaded Pretrained DAPT Weights from: {pretrained_path}")
                if msg.missing_keys:
                    print(f"Missing keys during load: {msg.missing_keys}")
                if msg.unexpected_keys:
                    print(f"Unexpected keys during load: {msg.unexpected_keys}")
            else:
                print(f"Warning: Expected DAPT weights at {pretrained_path} but not found. Using ImageNet baseline.")
        else:
            print("Baseline Strategy: Using default ImageNet pretrained weights.")
            
        # Extract purely the Swin-B encoder and drop the decoder
        self.encoder = self.simmim.encoder
        
        # The final norm outputs features of size (B, 7, 7, 1024)
        norm_shape = self.encoder.norm.normalized_shape[0]
        
        # Classification Head (Global Average Pooling -> Linear)
        self.head = nn.Linear(norm_shape, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)
        
    def forward(self, x):
        # We don't apply any masks during Fine-Tuning classification
        # We call the encoder directly bypassing the `forward_features` masking logic in SimMIM
        x = self.encoder.features(x)
        x = self.encoder.norm(x)
        
        # x is (B, 7, 7, 1024). Global average pooling over spatial dims (1, 2)
        x = x.mean(dim=(1, 2))
        return self.head(x)

def get_swin_classifier(num_classes=14, pretrained_path=None):
    return SwinClassifier(num_classes, pretrained_path)
