import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChestMNISTDataset
from transforms import get_transforms
from model import get_vit_classifier
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
    
    model_type = "DAPT" if Config.USE_DAPT else "Baseline (ImageNet)"
    logger.info(f"Initializing Fine-Tuning {model_type} ViT Classifier...")
    model = get_vit_classifier(num_classes=Config.NUM_CLASSES, pretrained_path=Config.PRETRAIN_CHECKPOINT)
    
    if hasattr(Config, 'USE_LORA') and Config.USE_LORA:
        from peft import LoraConfig, get_peft_model
        logger.info("Initializing PEFT LoRA strategy...")
        lora_config = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            target_modules=["out_proj", "mlp.0", "mlp.3"],
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            modules_to_save=["head"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    from utils import get_llrd_param_groups
    if hasattr(Config, 'USE_LORA') and Config.USE_LORA:
        logger.info(f"Using standard optimizer for LoRA parameters with LR: {Config.LORA_LR}")
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=Config.LORA_LR, weight_decay=Config.WEIGHT_DECAY)
    elif hasattr(Config, 'USE_LLRD') and Config.USE_LLRD:
        logger.info(f"Using LLRD strategy with decay rate: {Config.LLRD_DECAY}")
        optimizer_groups = get_llrd_param_groups(
            model, 
            weight_decay=Config.WEIGHT_DECAY, 
            lr=Config.LR, 
            layer_decay=Config.LLRD_DECAY
        )
        optimizer = torch.optim.Adam(optimizer_groups)
    else:
        logger.info("Using standard learning rate for all layers.")
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    best_val_loss = float('inf')
    scaler = torch.amp.GradScaler('cuda', enabled=getattr(Config, 'USE_AMP', False))
    
    for epoch in range(1, Config.EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Train]")
        optimizer.zero_grad()
        for i, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)
            
            with torch.amp.autocast('cuda', enabled=getattr(Config, 'USE_AMP', False)):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            accum_steps = getattr(Config, 'GRAD_ACCUM_STEPS', 1)
            loss = loss / accum_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accum_steps
            pbar.set_postfix({'loss': f"{loss.item() * accum_steps:.4f}"})
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Val]"):
                images = images.to(device)
                targets = targets.to(device)
                
                with torch.amp.autocast('cuda', enabled=getattr(Config, 'USE_AMP', False)):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch [{epoch}/{Config.EPOCHS}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, Config.CHECKPOINT_PATH)
            logger.info(f"Saved new best model with Val Loss: {val_loss:.4f}")

    # Plot loss after training finishes
    plot_loss(show_plot=False)

if __name__ == '__main__':
    train()
