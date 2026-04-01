import os
import sys
import torch
import torch.nn as nn

# Import SparK Architecture from Stage 2A DAPT 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DAPT/src')))
from spark import spark_resnet50

class CnnClassifier(nn.Module):
    def __init__(self, num_classes=14, pretrained_path=None):
        super().__init__()
        # Load the base SparK model structure
        self.spark = spark_resnet50()
        
        # Load weights based on strategy
        if pretrained_path:
            if os.path.exists(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                msg = self.spark.load_state_dict(checkpoint['model'], strict=False)
                print(f"Loaded Pretrained DAPT Weights from: {pretrained_path}")
            else:
                print(f"Warning: Expected DAPT weights at {pretrained_path} but not found. Using ImageNet baseline.")
        else:
            print("Baseline Strategy: Using default ImageNet pretrained weights.")
            
        # Extract purely the ResNet-50 encoder
        self.encoder = self.spark.encoder
        
        # In resnet50, layer4 outputs 2048 channels
        embed_dim = 2048 
        
        # Global Average Pooling and Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)
        
    def forward(self, x):
        # We don't apply any masks during Fine-Tuning classification
        # Recreate the unmasked standard ResNet forward pass manually
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        
        # Pooling and Classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

def get_resnet_classifier(num_classes=14, pretrained_path=None):
    return CnnClassifier(num_classes, pretrained_path)
