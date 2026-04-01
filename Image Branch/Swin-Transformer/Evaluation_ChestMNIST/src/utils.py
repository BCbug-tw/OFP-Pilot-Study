import random
import os
import numpy as np
import torch
import logging

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MetricLogger:
    def __init__(self, log_dir):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
    def info(self, msg):
        self.logger.info(msg)

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)

def get_swin_layer_id(name, num_layers):
    """
    Assign a conceptual layer id for varying depth parameter names.
    Swin-T and Swin-B usually have 4 stages. 
    Patch embedding: layer 0
    Stages 1-4: layers 1-4
    Head/Norm: layer 5
    """
    if "head" in name or "norm5" in name or name.startswith("norm") or name.startswith("classifier") or "encoder.norm" in name:
        return num_layers
    elif "layers.0" in name or "features.1" in name: return 1
    elif "layers.1" in name or "features.3" in name: return 2
    elif "layers.2" in name or "features.5" in name: return 3
    elif "layers.3" in name or "features.7" in name: return 4
    elif "patch_embed" in name or "features.0" in name: return 0
    else: return 0

def get_llrd_param_groups(model, weight_decay, lr, layer_decay):
    """
    Create optimizer parameter groups with layer-wise learning rate decay for Swin models.
    """
    # Define num_layers = 5 (PatchEmbed=0, Stages=1..4, Head=5)
    num_layers = 5
    
    param_group_names = {}
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for bias and norm layers
        if len(param.shape) == 1 or name.endswith(".bias") or (name.endswith(".weight") and "norm" in name):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        layer_id = get_swin_layer_id(name, num_layers)
        group_name = f"layer_{layer_id}_{group_name}"

        if group_name not in param_group_names:
            # lr formula: lr * (layer_decay ** (num_layers - layer_id))
            this_lr = lr * (layer_decay ** (num_layers - layer_id))
            
            param_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": this_lr
            }
            param_groups[group_name] = param_group_names[group_name]

        param_group_names[group_name]["params"].append(param)

    return list(param_groups.values())

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    import math
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def get_param_groups_with_head_lr(model, base_lr, head_multiplier, weight_decay):
    groups = {
        "head_decay": {"params": [], "lr": base_lr * head_multiplier, "weight_decay": weight_decay},
        "head_no_decay": {"params": [], "lr": base_lr * head_multiplier, "weight_decay": 0.0},
        "backbone_decay": {"params": [], "lr": base_lr, "weight_decay": weight_decay},
        "backbone_no_decay": {"params": [], "lr": base_lr, "weight_decay": 0.0},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        is_no_decay = len(param.shape) == 1 or name.endswith(".bias") or (name.endswith(".weight") and "norm" in name)
        
        if "head" in name:
            key = "head_no_decay" if is_no_decay else "head_decay"
            groups[key]["params"].append(param)
        else:
            key = "backbone_no_decay" if is_no_decay else "backbone_decay"
            groups[key]["params"].append(param)
            
    return list(groups.values())
