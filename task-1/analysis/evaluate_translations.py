import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.metrics import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(TASK1_DIR, "data")
CHECKPOINT_DIR = os.path.join(TASK1_DIR, "checkpoints")
RESULTS_DIR = os.path.join(TASK1_DIR, "results")

import sys
sys.path.append(TASK1_DIR)
from data.transforms import base_transform, get_normalization, ReflectionTranslate
from models.backbones import get_backbone, LinearClassifierHead

class STL10TranslationWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, base_transform, delta, direction, norm_transform):
        self.subset = subset
        self.base_transform = base_transform
        self.intervention = ReflectionTranslate(delta, direction)
        self.norm_transform = norm_transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = self.base_transform(img)      # Base Resize(224)
        img = self.intervention(img)        # Apply pad & shifted crop
        img = self.norm_transform(img)      # ToTensor & Normalize
        return img, label

@torch.no_grad()
def get_predictions(dataloader, backbone, head, device, is_zeroshot=False):
    all_preds = []
    all_labels = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        if is_zeroshot:
            confs = backbone.zero_shot_predict(x)
            _, preds = confs.max(dim=1)
        else:
            feats = backbone(x)
            logits = head(feats)
            _, preds = logits.max(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Translation Invariance on device: {device}\n")
    
    splits_path = os.path.join(DATA_DIR, "splits_seed6304.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    
    test_dataset = datasets.STL10(root=DATA_DIR, split="test", download=False)
    test_subset = Subset(test_dataset, splits["test_subset_indices"])
    
    models_to_test = [
        ("resnet", False), 
        ("vit", False), 
        ("clip", False), 
        ("clip_zeroshot", True)
    ]
    
    deltas = [0, 8, 16, 32]
    directions = ['up', 'down', 'left', 'right']
    
    plot_data = {m[0]: {'acc': [], 'cons': []} for m in models_to_test}

    for model_name, is_zeroshot in models_to_test:
        print(f"--- {model_name.upper()} ---")
        
        base_model_name = "clip" if is_zeroshot else model_name
        norm_transform = get_normalization(base_model_name)
        
        backbone = get_backbone(base_model_name, device)
        backbone.eval()
        
        head = None
        if not is_zeroshot:
            head = LinearClassifierHead(backbone.feature_dim, num_classes=10).to(device)
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_head_best.pth")
            head.load_state_dict(torch.load(checkpoint_path, map_location=device))
            head.eval()
            
        # Delta = 0 (Clean Baseline)
        clean_ds = STL10TranslationWrapper(test_subset, base_transform, 0, 'up', norm_transform)
        clean_loader = DataLoader(clean_ds, batch_size=64, shuffle=False, num_workers=2)
        clean_preds, labels = get_predictions(clean_loader, backbone, head, device, is_zeroshot)
        
        clean_acc = accuracy_score(labels, clean_preds)
        plot_data[model_name]['acc'].append(clean_acc)
        plot_data[model_name]['cons'].append(1.0)
        
        print(f"\u03B4 =  0 | Acc: {clean_acc:.4f} | Consistency: 1.0000")
        
        # Delta = 8, 16, 32
        for delta in [8, 16, 32]:
            delta_accs = []
            delta_conss = []
            
            for direction in directions:
                ds = STL10TranslationWrapper(test_subset, base_transform, delta, direction, norm_transform)
                loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
                preds, _ = get_predictions(loader, backbone, head, device, is_zeroshot)
                
                delta_accs.append(accuracy_score(labels, preds))
                delta_conss.append(np.mean(clean_preds == preds))
                
            avg_acc = np.mean(delta_accs)
            avg_cons = np.mean(delta_conss)
            
            plot_data[model_name]['acc'].append(avg_acc)
            plot_data[model_name]['cons'].append(avg_cons)
            
            print(f"\u03B4 = {delta:2d} | Avg Acc: {avg_acc:.4f} | Avg Consistency: {avg_cons:.4f}")
        print("")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'resnet': '#1f77b4', 'vit': '#2ca02c', 'clip': '#ff7f0e', 'clip_zeroshot': '#d62728'}
    markers = {'resnet': 'o', 'vit': 's', 'clip': '^', 'clip_zeroshot': 'D'}
    labels_map = {'resnet': 'ResNet-50', 'vit': 'ViT-B/16', 'clip': 'CLIP (Linear)', 'clip_zeroshot': 'CLIP (Zero-Shot)'}

    for model_name in plot_data.keys():
        ax1.plot(deltas, plot_data[model_name]['acc'], marker=markers[model_name], 
                 color=colors[model_name], label=labels_map[model_name], linewidth=2, markersize=8)
        ax2.plot(deltas, plot_data[model_name]['cons'], marker=markers[model_name], 
                 color=colors[model_name], label=labels_map[model_name], linewidth=2, markersize=8)

    ax1.set_title("Translation Invariance: Accuracy", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Displacement \u03B4 (pixels)", fontsize=12)
    ax1.set_ylabel("Top-1 Accuracy", fontsize=12)
    ax1.set_xticks(deltas)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.set_title("Translation Invariance: Prediction Consistency", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Displacement \u03B4 (pixels)", fontsize=12)
    ax2.set_ylabel("Consistency vs. Clean Baseline", fontsize=12)
    ax2.set_xticks(deltas)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plot_path = os.path.join(RESULTS_DIR, "translation_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {plot_path}")
    
    json_path = os.path.join(RESULTS_DIR, "translation_results.json")
    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=4)

if __name__ == "__main__":
    main()
