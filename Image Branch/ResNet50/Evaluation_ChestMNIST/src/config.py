import os
import torch

class Config:
    # Model
    IMAGE_SIZE = 224
    NUM_CLASSES = 1
    
    # Subsampling
    USE_SUBSAMPLE = True
    SUBSAMPLE_TRAIN = 8000
    SUBSAMPLE_VAL = 1000
    SUBSAMPLE_TEST = 1000
    SUBSAMPLE_POS_RATIO = 0.4
    
    # Training - Stage 2B FT 
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 5e-5
    WEIGHT_DECAY = 1e-4
    
    # Experiment Toggle
    USE_DAPT = True # Set to True for DAPT experiment, False for ImageNet baseline
    
    # Checkpoints
    DAPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "DAPT", "output")
    PRETRAIN_CHECKPOINT = os.path.join(DAPT_DIR, "spark_resnet_dapt_best.pth") if USE_DAPT else None
    
    # Localized Output Directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    exp_name = "dapt" if USE_DAPT else "baseline"
    if 'USE_SUBSAMPLE': # Actually we defined it above in the same class but hasattr check is safer globally, wait, let's just use boolean directly
        exp_name += "_10k_binary"
        
    OUTPUT_DIR = os.path.join(BASE_DIR, f"output_{exp_name}")
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, f"resnet_finetune_best_{exp_name}.pth")

    # Misc
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
