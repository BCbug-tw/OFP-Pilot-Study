import os
import re
import matplotlib.pyplot as plt
from config import Config

def plot_loss(log_path=None, save_dir=None, show_plot=False):
    if log_path is None:
        log_path = os.path.join(Config.OUTPUT_DIR, 'logger.log')
    if save_dir is None:
        save_dir = Config.OUTPUT_DIR

    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    epochs = []
    train_losses = []
    val_losses = []

    # Regex to match: ... Epoch [1/50] - Train Loss: 0.1657 - Val Loss: 0.1642
    pattern = re.compile(r"Epoch \[(\d+)/\d+\] - Train Loss: ([\d.]+) - Val Loss: ([\d.]+)")

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
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    
    plt.title(f'Training and Validation Loss ({Config.exp_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path)
    print(f"Loss curve saved to {save_path}")
    
    if show_plot:
        plt.show()
    plt.close()

if __name__ == '__main__':
    # When executed standalone, we show the plot.
    plot_loss(show_plot=True)
