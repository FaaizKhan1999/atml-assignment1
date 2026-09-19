import os
import json
import torch
import numpy as np
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
from data.transforms import base_transform, get_normalization, grayscale_intervention, hue_rotation_intervention
from models.backbones import get_backbone, LinearClassifierHead

class STL10InterventionWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, base_transform, intervention, norm_transform):
        self.subset = subset
        self.base_transform = base_transform
        self.intervention = intervention
        self.norm_transform = norm_transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        # Apply base resize -> intervention -> model-specific normalization[cite: 1]
        img = self.base_transform(img)
        if self.intervention is not None:
            img = self.intervention(img)
        img = self.norm_transform(img)
        return img, label

@torch.no_grad()
def get_predictions(dataloader, backbone, head, device, is_zeroshot=False):
    """Returns predictions and ground truth labels for a given configuration."""
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

def evaluate_intervention(clean_preds, trans_preds, labels):
    """Calculates accuracy change and prediction consistency relative to clean images[cite: 1]."""
    clean_acc = accuracy_score(labels, clean_preds)
    trans_acc = accuracy_score(labels, trans_preds)
    acc_change = trans_acc - clean_acc
    consistency = np.mean(clean_preds == trans_preds)
    
    return trans_acc, acc_change, consistency

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Color Bias on device: {device}\n")
    
    # Load test subset[cite: 1]
    splits_path = os.path.join(DATA_DIR, "splits_seed6304.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    
    test_dataset = datasets.STL10(root=DATA_DIR, split="test", download=False)
    test_subset = Subset(test_dataset, splits["test_subset_indices"])
    
    color_bias_results = {}
    
    models_to_test = [
        ("resnet", False), 
        ("vit", False), 
        ("clip", False), 
        ("clip_zeroshot", True)
    ]

    for model_name, is_zeroshot in models_to_test:
        print(f"--- {model_name.upper()} ---")
        
        base_model_name = "clip" if is_zeroshot else model_name
        norm_transform = get_normalization(base_model_name)
        
        # Initialize loaders for Clean, Grayscale, and Hue Rotation[cite: 1]
        clean_ds = STL10InterventionWrapper(test_subset, base_transform, None, norm_transform)
        gray_ds = STL10InterventionWrapper(test_subset, base_transform, grayscale_intervention, norm_transform)
        hue_ds = STL10InterventionWrapper(test_subset, base_transform, hue_rotation_intervention, norm_transform)
        
        clean_loader = DataLoader(clean_ds, batch_size=64, shuffle=False, num_workers=2)
        gray_loader = DataLoader(gray_ds, batch_size=64, shuffle=False, num_workers=2)
        hue_loader = DataLoader(hue_ds, batch_size=64, shuffle=False, num_workers=2)
        
        # Initialize model
        backbone = get_backbone(base_model_name, device)
        backbone.eval()
        
        head = None
        if not is_zeroshot:
            head = LinearClassifierHead(backbone.feature_dim, num_classes=10).to(device)
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_head_best.pth")
            head.load_state_dict(torch.load(checkpoint_path, map_location=device))
            head.eval()
            
        # Get predictions
        clean_preds, labels = get_predictions(clean_loader, backbone, head, device, is_zeroshot)
        gray_preds, _ = get_predictions(gray_loader, backbone, head, device, is_zeroshot)
        hue_preds, _ = get_predictions(hue_loader, backbone, head, device, is_zeroshot)
        
        # Calculate metrics[cite: 1]
        gray_acc, gray_delta, gray_cons = evaluate_intervention(clean_preds, gray_preds, labels)
        hue_acc, hue_delta, hue_cons = evaluate_intervention(clean_preds, hue_preds, labels)
        
        color_bias_results[model_name] = {
            "grayscale": {"acc": float(gray_acc), "acc_change": float(gray_delta), "consistency": float(gray_cons)},
            "hue_rotation": {"acc": float(hue_acc), "acc_change": float(hue_delta), "consistency": float(hue_cons)}
        }
        
        print(f"Grayscale    | Acc: {gray_acc:.4f} (\u0394 {gray_delta:+.4f}) | Consistency: {gray_cons:.4f}")
        print(f"Hue Rotation | Acc: {hue_acc:.4f} (\u0394 {hue_delta:+.4f}) | Consistency: {hue_cons:.4f}\n")

    # Save outputs
    output_file = os.path.join(RESULTS_DIR, "color_bias_results.json")
    with open(output_file, "w") as f:
        json.dump(color_bias_results, f, indent=4)
    print(f"Color bias results saved to {output_file}")

if __name__ == "__main__":
    main()
