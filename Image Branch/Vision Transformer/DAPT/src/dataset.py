import torch
from torch.utils.data import Dataset
from torchvision import transforms
import medmnist

class ChestMNISTUnlabeledDataset(Dataset):
    def __init__(self, split='train', img_size=224):
        self.split = split
        info = medmnist.INFO['chestmnist']
        DataClass = getattr(medmnist, info['python_class'])
        
        # We download and extract the dataset automatically if it doesn't exist
        self.dataset = DataClass(split=split, download=True, size=img_size, as_rgb=True)
        
        # Self-supervised structural augmentations
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # We discard the label entirely for DAPT logic
        img, _ = self.dataset[idx] 
        return self.transform(img)
