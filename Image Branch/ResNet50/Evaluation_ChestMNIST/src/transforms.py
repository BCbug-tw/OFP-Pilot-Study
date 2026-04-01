import random
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

def get_transforms(stage='train', img_size=224):
    t_list = []
    
    # Resize to 224x224 using BICUBIC interpolation
    t_list.append(transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC))
    
    # Augmentations
    if stage == 'train':
        t_list.append(transforms.RandomHorizontalFlip(p=0.5))
        t_list.append(transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)))
    
    t_list.append(transforms.ToTensor())
    
    # Standard normalization
    t_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(t_list)
