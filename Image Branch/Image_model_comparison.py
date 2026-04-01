import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, average_precision_score

# ---------------------------------------------------------
# Configuration: Define the models to compare here
# ---------------------------------------------------------
# Each entry should specify a display name, the path to the predictions CSV, and a color.
# Paths should be relative to where you run this script (usually repository root).
# 
# Adjust 'prediction_file' values with the exact path based on your exp_name when running.
MODELS_TO_COMPARE = [
    {
        "name": "ResNet50_ImageNet",
        "prediction_file": "ResNet50/Evaluation_ChestMNIST/output_baseline_10k_binary/evaluation/test_predictions_baseline.csv",
    },
    {
        "name": "Swin-T_ImageNet",
        "prediction_file": "Swin-Transformer/Evaluation_ChestMNIST/output_baseline_10k_binary/evaluation/test_predictions_baseline_10k_binary.csv",
    },
    {
        "name": "Swin-T_DAPT",
        "prediction_file": "Swin-Transformer/Evaluation_ChestMNIST/output_dapt_10k_binary/evaluation/test_predictions_dapt_10k_binary.csv",
    },
    {
        "name": "Swin-T_DAPT_LLRD",
        "prediction_file": "Swin-Transformer/Evaluation_ChestMNIST/output_dapt_llrd_10k_binary/evaluation/test_predictions_dapt_llrd_10k_binary.csv",
    },
    {
        "name": "Swin-T_DAPT_LoRA",
        "prediction_file": "Swin-Transformer/Evaluation_ChestMNIST/output_dapt_lora_10k_binary/evaluation/test_predictions_dapt_lora_10k_binary.csv",
    },
    {
        "name": "ViT_ImageNet",
        "prediction_file": "Vision Transformer/Evaluation_ChestMNIST/output_baseline_10k_binary/evaluation/test_predictions_baseline_10k_binary.csv",
    },
    {
        "name": "ViT_DAPT",
        "prediction_file": "Vision Transformer/Evaluation_ChestMNIST/output_dapt_10k_binary/evaluation/test_predictions_dapt_10k_binary.csv",
    },
    {
        "name": "ViT_DAPT_LLRD",
        "prediction_file": "Vision Transformer/Evaluation_ChestMNIST/output_dapt_llrd_10k_binary/evaluation/test_predictions_dapt_llrd_10k_binary.csv",
    },
    {
        "name": "ViT_DAPT_LoRA",
        "prediction_file": "Vision Transformer/Evaluation_ChestMNIST/output_dapt_lora_10k_binary/evaluation/test_predictions_dapt_lora_10k_binary.csv",
    }
]

# Adjust if evaluating on the 14-class original ChestMNIST instead of 'Infiltration'
TARGET_CLASS = "Infiltration" 
OUTPUT_DIR = "image_comparison_results"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = []
    
    plt.figure(figsize=(10, 8))
    roc_ax = plt.gca()
    
    fig_pr, pr_ax = plt.subplots(figsize=(10, 8))
    
    # Color Mapping: Dynamically generate colors based on the number of models
    cmap = plt.get_cmap('tab10')
    num_models = len(MODELS_TO_COMPARE)
    colors = [cmap(i / max(1, num_models - 1) if num_models > 1 else 0) for i in range(num_models)]
    
    for idx, model_config in enumerate(MODELS_TO_COMPARE):
        name = model_config["name"]
        file_path = model_config["prediction_file"]
        color = colors[idx]
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found for {name} ({file_path}). Make sure you have run evaluate.py for this configuration. Skipping this model.")
            continue
            
        print(f"Processing {name}...")
        df = pd.read_csv(file_path)
        
        gt_col = f"GroundTruth_{TARGET_CLASS}"
        prob_col = f"Prob_{TARGET_CLASS}"
        pred_col = f"Pred_{TARGET_CLASS}"
        
        if gt_col not in df.columns or prob_col not in df.columns:
            print(f"Error: Target class columns ('{gt_col}', '{prob_col}') not found in {file_path}. Skipping.")
            continue
            
        y_true = df[gt_col].values
        y_prob = df[prob_col].values
        if pred_col in df.columns:
            y_pred = df[pred_col].values
        else:
            y_pred = (y_prob > 0.5).astype(int)
            
        # Hard Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        # ROC Curve & AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # PR Curve & AP
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "AUC": roc_auc,
            "AP (Avg Precision)": ap
        })
        
        # Plot to ROC Curve
        roc_ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Plot to PR Curve
        pr_ax.plot(recall_vals, precision_vals, color=color, lw=2, label=f'{name} (AP = {ap:.3f})')
        
    if not results:
        print("\nNo valid model results found. No comparison plots were generated.")
        return
        
    # Finalize ROC Curve Plot
    roc_ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5)
    roc_ax.set_xlim([0.0, 1.0])
    roc_ax.set_ylim([0.0, 1.0])
    roc_ax.set_xlabel('False Positive Rate', fontsize=12)
    roc_ax.set_ylabel('True Positive Rate', fontsize=12)
    roc_ax.set_title(f'Integrated ROC Curve - {TARGET_CLASS} in ChestMNIST', fontsize=14)
    roc_ax.legend(loc="lower right", fontsize=10)
    roc_ax.grid(alpha=0.3)
    roc_figure_path = os.path.join(OUTPUT_DIR, "Integrated_ROC_Curve.png")
    roc_ax.figure.savefig(roc_figure_path, dpi=300, bbox_inches='tight')
    plt.close(roc_ax.figure)
    print(f"Saved Integrated ROC Curve to: {roc_figure_path}")
    
    # Finalize PR Curve Plot
    pr_ax.set_xlim([0.0, 1.0])
    pr_ax.set_ylim([0.0, 1.05])
    pr_ax.set_xlabel('Recall', fontsize=12)
    pr_ax.set_ylabel('Precision', fontsize=12)
    pr_ax.set_title(f'Integrated PR Curve - {TARGET_CLASS} in ChestMNIST', fontsize=14)
    pr_ax.legend(loc="lower right", fontsize=10)
    pr_ax.grid(alpha=0.3)
    pr_figure_path = os.path.join(OUTPUT_DIR, "Integrated_PR_Curve.png")
    pr_ax.figure.savefig(pr_figure_path, dpi=300, bbox_inches='tight')
    plt.close(pr_ax.figure)
    print(f"Saved Integrated PR Curve to: {pr_figure_path}")
    
    # Text Table
    results_df = pd.DataFrame(results)
    results_df = results_df.round(4)
    print("\n============== Model Comparison Metrics ==============")
    print(results_df.to_string(index=False))
    print("======================================================\n")
    
    # CSV Table Output
    csv_path = os.path.join(OUTPUT_DIR, "Model_Comparison_Metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved metrics table to: {csv_path}")

    # Visual Table Format (PNG Output)
    fig_tbl, ax_tbl = plt.subplots(figsize=(10, 2 + len(results)*0.5))
    ax_tbl.axis('off')
    ax_tbl.axis('tight')
    
    table_data = results_df.values.tolist()
    col_labels = results_df.columns.tolist()
    
    table = ax_tbl.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Table header style
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#e0e0e0')
        table[(0, i)].set_text_props(weight='bold')
        
    plt.title(f'Model Comparison Metrics - {TARGET_CLASS}', fontsize=14, pad=20)
    plt.tight_layout()
    table_img_path = os.path.join(OUTPUT_DIR, "Model_Comparison_Metrics.png")
    plt.savefig(table_img_path, dpi=300, bbox_inches='tight')
    plt.close(fig_tbl)
    print(f"Saved metrics PNG table to: {table_img_path}\n")


if __name__ == "__main__":
    main()
