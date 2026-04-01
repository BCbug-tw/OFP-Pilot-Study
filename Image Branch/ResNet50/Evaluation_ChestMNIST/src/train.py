import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ChestMNISTDataset
from transforms import get_transforms
from model import get_resnet_classifier
from config import Config
from utils import set_seed, MetricLogger, save_checkpoint
from medmnist import Evaluator
from plot_loss import plot_loss

def train():
    set_seed(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    logger = MetricLogger(Config.OUTPUT_DIR)
    
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")
    
    # Data Setup
    logger.info("Loading datasets...")
    
    if getattr(Config, 'USE_SUBSAMPLE', False):
        logger.info(f"Using Subsampled Dataset (Binary Infiltration) - Total: {Config.SUBSAMPLE_TRAIN + Config.SUBSAMPLE_VAL + Config.SUBSAMPLE_TEST}")
        logger.info(f"-> Train: {Config.SUBSAMPLE_TRAIN}, Val: {Config.SUBSAMPLE_VAL}, Test: {Config.SUBSAMPLE_TEST} (Pos Ratio: {Config.SUBSAMPLE_POS_RATIO})")
    else:
        logger.info("Using original ChestMNIST dataset (14 classes)")
    train_transform = get_transforms(stage='train', img_size=Config.IMAGE_SIZE)
    val_transform = get_transforms(stage='val', img_size=Config.IMAGE_SIZE)
    
    train_dataset = ChestMNISTDataset(split='train', stage='train', transform=train_transform, img_size=Config.IMAGE_SIZE)
    val_dataset = ChestMNISTDataset(split='val', stage='val', transform=val_transform, img_size=Config.IMAGE_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # Model Setup
    model_type = "DAPT" if Config.USE_DAPT else "Baseline (ImageNet)"
    logger.info(f"Initializing Fine-Tuning {model_type} CNN Classifier...")
    model = get_resnet_classifier(num_classes=Config.NUM_CLASSES, pretrained_path=Config.PRETRAIN_CHECKPOINT)
    model.to(device)
    
    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, Config.EPOCHS + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Train]")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Val]"):
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch [{epoch}/{Config.EPOCHS}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, Config.CHECKPOINT_PATH)
            logger.info(f"Saved new best model with Val Loss: {val_loss:.4f}")

    # Plot loss after training finishes
    plot_loss(show_plot=False)

if __name__ == '__main__':
    train()
