import os
import torch

class Config:
    # Model
    IMAGE_SIZE = 224
    PATCH_SIZE = 32
    MASK_RATIO = 0.6
    
    # Training - Stage 2A DAPT 
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 5e-5  
    WEIGHT_DECAY = 0.05
    
    # Checkpoints
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "spark_resnet_dapt_best.pth")


    # Misc
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
