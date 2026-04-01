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

def get_vit_layer_id(name, num_layers):
    """
    Assign a conceptual layer id for varying depth parameter names in ViT.
    Patch embedding: layer 0
    Encoder blocks 0 to 11: layers 1 to 12
    Head/Norm (final layer): layer 13
    """
    if "head" in name or name.endswith("encoder.ln.weight") or name.endswith("encoder.ln.bias") or name.startswith("classifier") or "norm" in name and not "layers" in name:
        return num_layers
    elif "layers" in name:
        # Expected format: e.g., 'mae.vit.encoder.layers.5.xx'
        try:
            block_idx = int(name.split("layers.")[-1].split(".")[0])
            return block_idx + 1
        except ValueError:
            pass
    elif "patch_embed" in name or "pos_embed" in name or "cls_token" in name: 
        return 0
    
    return 0

def get_llrd_param_groups(model, weight_decay, lr, layer_decay):
    """
    Create optimizer parameter groups with layer-wise learning rate decay for ViT.
    """
    # ViT-Base has 12 blocks. So num_layers = 13 (PatchEmbed/Tokens=0, Blocks=1..12, Head/FinalNorm=13)
    num_layers = 13
    
    param_group_names = {}
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for bias and norm layers
        if len(param.shape) == 1 or name.endswith(".bias") or (name.endswith(".weight") and "norm" in name) or "ln" in name:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        layer_id = get_vit_layer_id(name, num_layers)
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
