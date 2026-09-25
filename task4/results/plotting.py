import json
import matplotlib.pyplot as plt
import os

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Skipping...")
        return [], [], []
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    epochs = [d['epoch'] for d in data]
    train_acc = [d['train_acc'] for d in data]
    val_acc = [d['val_acc'] for d in data]
    return epochs, train_acc, val_acc

def generate_plots():
    # Map labels to their corresponding JSON files
    datasets = {
        "Vanilla": "vanilla_training.json",
        "GCSC": "gcsc_training.json",
        "PROSER (Fine-Tuning)": "proser_finetuning.json",
        "RPL": "rpl_training.json"
    }

    train_curves = {}
    val_curves = {}

    # Extract data
    for label, filepath in datasets.items():
        epochs, train_acc, val_acc = load_data(filepath)
        if epochs:
            train_curves[label] = (epochs, train_acc)
            val_curves[label] = (epochs, val_acc)

    colors = {"Vanilla": "#1f77b4", "GCSC": "#ff7f0e", "PROSER (Fine-Tuning)": "#2ca02c", "RPL": "#d62728"}
    linestyles = {"Vanilla": "-", "GCSC": "-", "PROSER (Fine-Tuning)": "--", "RPL": "-"}

    # 1. Plot Training Accuracy
    plt.figure(figsize=(10, 6), dpi=300)
    for label, (epochs, acc) in train_curves.items():
        plt.plot(epochs, acc, label=label, color=colors[label], linestyle=linestyles[label], linewidth=2)
    
    plt.title('Training Accuracy across Epochs', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.savefig('training_acc.png')
    print("Successfully generated and saved 'training_acc.png'")
    plt.close()

    # 2. Plot Validation Accuracy
    plt.figure(figsize=(10, 6), dpi=300)
    for label, (epochs, acc) in val_curves.items():
        plt.plot(epochs, acc, label=label, color=colors[label], linestyle=linestyles[label], linewidth=2)
    
    plt.title('Validation Accuracy across Epochs', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.savefig('validation_acc.png')
    print("Successfully generated and saved 'validation_acc.png'")
    plt.close()

if __name__ == "__main__":
    generate_plots()
