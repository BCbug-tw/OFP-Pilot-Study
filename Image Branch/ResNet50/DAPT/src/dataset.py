import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import medmnist

class SparkMaskGenerator:
    """
    SparK Mask Generator for CNNs.
    Generates a boolean mask indicating which spatial patches are REMOVED.
    """
    def __init__(self, input_size=224, mask_patch_size=32, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        
        self.grid_size = self.input_size // self.mask_patch_size
        self.token_count = self.grid_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        # 1 means MASKED, 0 means VISIBLE
        import numpy as np
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.grid_size, self.grid_size))
        return torch.tensor(mask, dtype=torch.bool)

class ChestMNISTUnlabeledSparKDataset(Dataset):
    def __init__(self, split='train', img_size=224, mask_generator=None):
        self.split = split
        info = medmnist.INFO['chestmnist']
        DataClass = getattr(medmnist, info['python_class'])
        
        self.dataset = DataClass(split=split, download=True, size=img_size, as_rgb=True)
        self.mask_generator = mask_generator
        
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
