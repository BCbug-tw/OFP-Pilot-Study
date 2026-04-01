import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mae import mae_vit_base_patch16
from dataset import ChestMNISTUnlabeledDataset
from config import Config

def pretrain():
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    print("Loading ChestMNIST DAPT dataset (unlabeled)...")
    train_dataset = ChestMNISTUnlabeledDataset(split='train', img_size=Config.IMAGE_SIZE)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        drop_last=True
    )
    
    print("Initialize Masked Autoencoder (MAE-ViT) with ImageNet weights...")
    model = mae_vit_base_patch16(img_size=Config.IMAGE_SIZE)
    # Weights automatically downloaded within the mae constructor
        
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY, betas=(0.9, 0.95))
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    best_loss = float('inf')
    
    print("Starting DAPT Loop...")
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [DAPT]")
        for images in pbar:
            images = images.to(device)
            
            optimizer.zero_grad()
            loss, _, _ = model(images, mask_ratio=Config.MASK_RATIO)
            
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
