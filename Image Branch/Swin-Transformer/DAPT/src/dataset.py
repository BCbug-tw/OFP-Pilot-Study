import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import medmnist

# SimMIM Mask Generator
class MaskGenerator:
    """
    SimMIM Mask Generator for Swin-Transformer.
    Generates a boolean mask by upsampling a coarse grid.
    """
    def __init__(self, input_size=224, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        # 1 means MASKED, 0 means VISIBLE
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        # Upsample the coarse grid to model patch grid
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return torch.tensor(mask, dtype=torch.bool)

class ChestMNISTUnlabeledSimMIMDataset(Dataset):
    def __init__(self, split='train', img_size=224, mask_generator=None):
        self.split = split
        info = medmnist.INFO['chestmnist']
        DataClass = getattr(medmnist, info['python_class'])
        
        self.dataset = DataClass(split=split, download=True, size=img_size, as_rgb=True)
        self.mask_generator = mask_generator
        
        # SimMIM standard DAPT augmentations
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.67, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Discard classification label
        img, _ = self.dataset[idx] 
        img = self.transform(img)
        
        mask = self.mask_generator() if self.mask_generator else torch.zeros((1,1))
        return img, mask
