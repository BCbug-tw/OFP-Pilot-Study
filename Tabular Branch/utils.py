"""
Tabular Branch - Shared Utilities
=================================
Provides unified logging setup and loss curve plotting functions
used across all model training scripts (XGBoost, LightGBM, CatBoost, FT-Transformer).
"""

import os
import re
import logging
import matplotlib.pyplot as plt


def setup_logging(output_dir, log_filename='logger.log'):
    """
    Initialize a logger that writes to both a file and the console.

    Args:
        output_dir (str): Directory to save the log file.
        log_filename (str): Name of the log file (default: 'logger.log').

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)

    # Reset any existing handlers to avoid duplicate logs on re-runs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def plot_loss_from_dict(train_losses, val_losses, save_dir, title,
                        x_label='Iteration', show_plot=False):
    """
    Plot train/val loss curves from Python lists or arrays.
    Designed for GBDT models that expose evals_result dictionaries.

    Args:
        train_losses (list): Per-iteration training loss values.
        val_losses (list): Per-iteration validation loss values.
        save_dir (str): Directory to save the plot.
        title (str): Plot title (e.g., 'XGBoost Training Loss').
        x_label (str): X-axis label (default: 'Iteration').
        show_plot (bool): Whether to display the plot interactively.
    """
    os.makedirs(save_dir, exist_ok=True)

    iterations = range(1, len(val_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_losses, label='Train Loss', alpha=0.8)
    plt.plot(iterations, val_losses, label='Val Loss', alpha=0.8)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Loss (Logloss)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Loss curve saved to {save_path}")

    if show_plot:
        plt.show()
    plt.close()


def plot_loss_from_log(log_path, save_dir, title='Training and Validation Loss',
                       show_plot=False):
    """
    Parse a log file and plot train/val loss curves.
    Designed for FT-Transformer which logs per-epoch losses to a text file.
    
    Expected log format:
        ... Epoch [1/100] - Train Loss: 0.5478 - Val Loss: 0.5322

    Args:
        log_path (str): Path to the logger.log file.
        save_dir (str): Directory to save the plot.
        title (str): Plot title.
        show_plot (bool): Whether to display the plot interactively.
    """
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    epochs = []
    train_losses = []
    val_losses = []

    pattern = re.compile(
        r"Epoch \[(\d+)/\d+\] - Train Loss: ([\d.]+) - Val Loss: ([\d.]+)"
    )

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_losses.append(float(match.group(3)))

    if not epochs:
        print("No training data found in log.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', alpha=0.8)
    plt.plot(epochs, val_losses, label='Val Loss', alpha=0.8)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Loss curve saved to {save_path}")

    if show_plot:
        plt.show()
    plt.close()
