import os
import numpy as np
import torch
from torch.utils.data import Dataset
import medmnist

class ChestMNISTDataset(Dataset):
    def __init__(self, split='train', stage='train', transform=None, img_size=224, max_samples=None):
        self.stage = stage
        self.transform = transform
        self.img_size = img_size
        
        info = medmnist.INFO['chestmnist']
        DataClass = getattr(medmnist, info['python_class'])
        
        self.dataset = DataClass(split=split, transform=None, download=True, size=img_size, as_rgb=True)
        
        from config import Config
        self.use_subsample = hasattr(Config, 'USE_SUBSAMPLE') and Config.USE_SUBSAMPLE
        
        if self.use_subsample:
            labels = self.dataset.labels
            pos_indices = np.where(labels[:, 3] == 1)[0]
            neg_indices = np.where(labels.sum(axis=1) == 0)[0]
            
            if split == 'train':
                total_samples = Config.SUBSAMPLE_TRAIN
            elif split == 'val':
                total_samples = Config.SUBSAMPLE_VAL
            elif split == 'test':
                total_samples = Config.SUBSAMPLE_TEST
            else:
                total_samples = len(self.dataset)
            
            num_pos = int(total_samples * Config.SUBSAMPLE_POS_RATIO)
            num_neg = total_samples - num_pos
            
            np.random.seed(Config.SEED)
            sampled_pos = np.random.choice(pos_indices, num_pos, replace=False)
            sampled_neg = np.random.choice(neg_indices, num_neg, replace=False)
            
            self.indices = np.concatenate([sampled_pos, sampled_neg])
            np.random.shuffle(self.indices)
        elif max_samples is not None and max_samples < len(self.dataset):
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
            all_labels = self.dataset.labels
            try:
                msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=max_samples, random_state=42)
                indices = np.arange(len(self.dataset))
                _, subset_idx = next(msss.split(indices, all_labels))
                self.indices = subset_idx
            except ImportError:
                print("Warning: iterstrat not installed. Using numpy random choice.")
                np.random.seed(42)
                self.indices = np.random.choice(len(self.dataset), max_samples, replace=False)
        else:
            self.indices = np.arange(len(self.dataset))
            
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, target = self.dataset[real_idx]
        
        if self.use_subsample:
            target = np.array([target[3]])
            
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(target, dtype=torch.float32)
