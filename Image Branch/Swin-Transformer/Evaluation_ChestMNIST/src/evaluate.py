import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sklearn_auc, precision_recall_curve, average_precision_score
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from medmnist import Evaluator
from medmnist.info import INFO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import accuracy_score

from dataset import ChestMNISTDataset
from model import get_swin_classifier
from config import Config

def plot_roc_curve(y_true, y_score, output_dir, class_names, display_name, save_name):
    plt.figure(figsize=(10, 8))
    for i in range(y_true.shape[1]):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = sklearn_auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Swin-T with {display_name} on ChestMNIST')
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"SwinT_roc_curve_{save_name}.png"), dpi=300)
    plt.close()

def plot_pr_curve(y_true, y_score, output_dir, class_names, display_name, save_name):
    plt.figure(figsize=(10, 8))
    for i in range(y_true.shape[1]):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        ap = average_precision_score(y_true[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]} (AP = {ap:.3f})')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve for Swin-T with {display_name} on ChestMNIST')
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"SwinT_pr_curve_{save_name}.png"), dpi=300)
    plt.close()

def reshape_transform(tensor, height=7, width=7):
    # Swin-T final feature map is [B, 7, 7, 768] (for 224x224 input)
    # GradCAM needs [B, C, H, W]
    result = tensor.permute(0, 3, 1, 2)
    return result

def generate_gradcam(model, dataset, device, output_dir, class_names, samples_per_class=2):
    # For Swin-T, the optimal target is typically the last norm of the final block
    # In torchvision's swin_t: `model.encoder.features[-1][-1].norm2`
    target_layers = [model.encoder.features[-1][-1].norm2]
    
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    
    # Track how many samples we have generated for each class
    class_counts = {i: 0 for i in range(len(class_names))}
    
    # Iterate through dataset to find representative samples
    for i in tqdm(range(len(dataset)), desc="Generating GradCAM"):
        # Stop if we have collected enough samples for all classes
        if all(count >= samples_per_class for count in class_counts.values()):
            break
            
        img_tensor, target = dataset[i]
        
        # Get positive classes for the current image (ChestMNIST is multi-label)
        target_tensor = torch.as_tensor(target)
        positive_classes = torch.where(target_tensor == 1)[0].tolist()
        
        for c in positive_classes:
            if class_counts[c] < samples_per_class:
                class_counts[c] += 1
                
                # Reverse normalization for visualization
                img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)
                
                input_tensor = img_tensor.unsqueeze(0).to(device)
                
                # Target the specific positive class for GradCAM
                targets_cam = [ClassifierOutputTarget(c)]
                
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets_cam)
                grayscale_cam = grayscale_cam[0, :]
                
                cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                
                class_name_safe = class_names[c].replace(" ", "_").replace("/", "_")
                count_idx = class_counts[c]
                
                img_save_path = os.path.join(output_dir, f"{class_name_safe}_sample{count_idx}_idx{i}.png")
                cv2.imwrite(img_save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

def evaluate():
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    output_dir = os.path.join(Config.OUTPUT_DIR, "evaluation")
    gradcam_dir = os.path.join(output_dir, "GradCAM")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gradcam_dir, exist_ok=True)
    
    print("Loading test dataset...")
    if getattr(Config, 'USE_SUBSAMPLE', False):
        print(f"Using Subsampled Dataset (Binary Infiltration) - Testing against {Config.SUBSAMPLE_TEST} samples.")
    else:
        print("Using original ChestMNIST dataset (14 classes)")
    test_dataset = ChestMNISTDataset(split='test', stage='test', img_size=Config.IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    class_names = INFO['chestmnist']['label']
    
    if hasattr(Config, 'USE_SUBSAMPLE') and Config.USE_SUBSAMPLE:
        class_names_list = ['Infiltration']
    else:
        evaluator = Evaluator('chestmnist', 'test', size=Config.IMAGE_SIZE)
        class_names_list = [class_names[str(i)] for i in range(14)]
    
    exp_name = getattr(Config, 'exp_name', "dapt" if Config.USE_DAPT else "baseline")
    
    display_model_type = "DAPT" if Config.USE_DAPT else "Baseline (ImageNet)"
    if getattr(Config, 'USE_LORA', False):
        display_model_type += " + LoRA"
    elif hasattr(Config, 'USE_LLRD') and Config.USE_LLRD:
        display_model_type += " + LLRD"
        
    print(f"Initializing {display_model_type} Swin-T model...")
    model = get_swin_classifier(num_classes=Config.NUM_CLASSES, pretrained_path=Config.PRETRAIN_CHECKPOINT)
    
    if getattr(Config, 'USE_LORA', False):
        from peft import LoraConfig, get_peft_model
        print("Initializing PEFT LoRA formulation for evaluation...")
        lora_config = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            modules_to_save=["head"]
        )
        model = get_peft_model(model, lora_config)

    if os.path.exists(Config.CHECKPOINT_PATH):
        checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint.")
    else:
        print(f"Warning: No checkpoint found at {Config.CHECKPOINT_PATH}.")
        
    model.to(device)
    model.eval()
    
    y_true = []
    y_score = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            scores = torch.sigmoid(outputs).cpu().numpy()
            y_score.append(scores)
            y_true.append(targets.numpy())
            
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    
    if hasattr(Config, 'USE_SUBSAMPLE') and Config.USE_SUBSAMPLE:
        auc = sklearn_auc(*roc_curve(y_true[:, 0], y_score[:, 0])[:2])
        acc = accuracy_score(y_true[:, 0], y_score[:, 0] > 0.5)
    else:
        auc, acc = evaluator.evaluate(y_score)
        
    print(f"Test AUC: {auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    
    print("Plotting ROC curves...")
    plot_roc_curve(y_true, y_score, output_dir, class_names_list, display_model_type, exp_name)
    print(f"ROC curve saved to {os.path.join(output_dir, f'SwinT_roc_curve_{exp_name}.png')}")
    
    print("Plotting PR curves...")
    plot_pr_curve(y_true, y_score, output_dir, class_names_list, display_model_type, exp_name)
    print(f"PR curve saved to {os.path.join(output_dir, f'SwinT_pr_curve_{exp_name}.png')}")
    
    print("Exporting instance-level predictions to CSV...")
    pred_data = {}
    num_classes = 1 if (hasattr(Config, 'USE_SUBSAMPLE') and Config.USE_SUBSAMPLE) else 14
    for i in range(num_classes):
        class_name = class_names_list[i]
        pred_data[f"GroundTruth_{class_name}"] = y_true[:, i]
        pred_data[f"Prob_{class_name}"] = y_score[:, i]
        pred_data[f"Pred_{class_name}"] = (y_score[:, i] > 0.5).astype(int)
    
    df_preds = pd.DataFrame(pred_data)
    df_preds.to_csv(os.path.join(output_dir, f"test_predictions_{exp_name}.csv"), index=False)
    print(f"Predictions saved to {os.path.join(output_dir, f'test_predictions_{exp_name}.csv')}")

    print("Exporting metrics to CSV...")
    metrics_data = {"Class": class_names_list, "AUC": [], "AP": []}
    
    num_classes = 1 if (hasattr(Config, 'USE_SUBSAMPLE') and Config.USE_SUBSAMPLE) else 14
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        metrics_data["AUC"].append(sklearn_auc(fpr, tpr))
        metrics_data["AP"].append(average_precision_score(y_true[:, i], y_score[:, i]))
        
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.loc[len(df_metrics)] = ["Overall (Macro AUC)", auc, np.mean(metrics_data["AP"])]
    df_metrics.loc[len(df_metrics)] = ["Overall (Accuracy)", acc, np.nan]
    df_metrics.to_csv(os.path.join(output_dir, f"metrics_{exp_name}.csv"), index=False)
    print(f"Metrics saved to {os.path.join(output_dir, f'metrics_{exp_name}.csv')}")
    
    print("Generating GradCAM samples...")
    for param in model.parameters():
        param.requires_grad = True
        
    generate_gradcam(model, test_dataset, device, gradcam_dir, class_names_list, samples_per_class=2)
    print(f"GradCAM images saved to {gradcam_dir}")

if __name__ == '__main__':
    evaluate()
