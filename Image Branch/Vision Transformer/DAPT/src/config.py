import os
import torch

class Config:
    # Model/Data
    IMAGE_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 5e-5  # Lower learning rate for Phase 2 DAPT
    WEIGHT_DECAY = 0.05
    MASK_RATIO = 0.75
    
    # Checkpoints
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "mae_vit_dapt_best.pth")

    # Misc
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
