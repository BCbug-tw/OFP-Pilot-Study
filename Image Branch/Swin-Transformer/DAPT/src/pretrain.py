import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChestMNISTUnlabeledSimMIMDataset
from config import Config

# Import local replacement SimMIM Architecture 
from simmim import simmim_swin_base
from dataset import MaskGenerator

def pretrain():
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    print("Loading ChestMNIST DAPT dataset (unlabeled)...")
    mask_generator = MaskGenerator(
        input_size=Config.IMAGE_SIZE,
        mask_patch_size=Config.MASK_PATCH_SIZE,
        model_patch_size=Config.PATCH_SIZE,
        mask_ratio=Config.MASK_RATIO
    )
    
    train_dataset = ChestMNISTUnlabeledSimMIMDataset(
        split='train', 
        img_size=Config.IMAGE_SIZE, 
        mask_generator=mask_generator
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        drop_last=True
    )
    
    print("Initialize SimMIM Masked Autoencoder (Swin-B) with ImageNet weights...")
    model = simmim_swin_base()
    # Weights pulled from torchvision inside simmim_swin_base()
        
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY, betas=(0.9, 0.999))
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    best_loss = float('inf')
    
    print("Starting DAPT Loop...")
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [DAPT]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            loss, _ = model(images, masks)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch}/{Config.EPOCHS}] Loss: {epoch_loss:.4f}")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': epoch_loss,
            }, Config.CHECKPOINT_PATH)
            print(f"Saved new best DAPT model with Loss: {epoch_loss:.4f}")

if __name__ == '__main__':
    pretrain()
