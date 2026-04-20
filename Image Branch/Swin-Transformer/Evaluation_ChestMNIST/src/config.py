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
    # Experiment Toggle
    USE_DAPT = True # Set to True for DAPT experiment, False for ImageNet baseline
    USE_LLRD = False  # Set to True to enable Layer-Wise Learning Rate Decay (LLRD) for Swin-T
    USE_LORA = False # Set to True to enable LoRA for Swin-T
    LLRD_DECAY = 0.85 # Decay rate for LLRD
    
    # Training - Stage 2B FT 
    BATCH_SIZE = 16 # Reduced from 32 to avoid GPU OOM
    GRAD_ACCUM_STEPS = 2 # Effective batch size = 32
    EPOCHS = 50
    LR = 1e-4 if USE_DAPT else 5e-5
    WARMUP_EPOCHS = 5
    HEAD_LR_MULTIPLIER = 10
    WEIGHT_DECAY = 1e-4
    
    # LoRA Settings
    LORA_R = 32
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_LR = 5e-4
    
    # Checkpoints
    DAPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "DAPT", "output")
    PRETRAIN_CHECKPOINT = os.path.join(DAPT_DIR, "simmim_swin_dapt_best.pth") if USE_DAPT else None
    
    # Localized Output Directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    exp_name = "dapt" if USE_DAPT else "baseline"
    if USE_LORA:
        exp_name += "_lora"
    elif USE_LLRD:
        exp_name += "_llrd"
        
    if USE_SUBSAMPLE:
        exp_name += "_10k_binary"
        
    OUTPUT_DIR = os.path.join(BASE_DIR, f"output_{exp_name}")
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, f"swin_finetune_best_{exp_name}.pth")

    # Misc
    SEED = 42
    NUM_WORKERS = 0  # Changed to 0 to prevent System RAM OOM on Windows when using spawn
    USE_AMP = True   # Enable Mixed Precision to save VRAM
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
