import os
import json
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(TASK1_DIR, "configs", "base.yaml")
DATA_DIR = os.path.join(TASK1_DIR, "data")
CHECKPOINT_DIR = os.path.join(TASK1_DIR, "checkpoints")
RESULTS_DIR = os.path.join(TASK1_DIR, "results")

import sys
sys.path.append(TASK1_DIR)
from data.transforms import base_transform, get_normalization
from models.backbones import get_backbone, LinearClassifierHead

class STL10TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = self.transform(img)
        return img, label

@torch.no_grad()
def evaluate_model(dataloader, backbone, head, device, is_zeroshot=False):
    """Evaluates a model and returns top-1 accuracy, macro-F1, and mean max confidence."""
    all_preds = []
    all_labels = []
    all_confs = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        if is_zeroshot:
            confs = backbone.zero_shot_predict(x)
            max_confs, preds = confs.max(dim=1)
        else:
            feats = backbone(x)
            logits = head(feats)
            confs = torch.softmax(logits, dim=1)
            max_confs, preds = confs.max(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_confs.extend(max_confs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    mean_conf = np.mean(all_confs)

    return acc, macro_f1, mean_conf

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Clean Baseline on device: {device}\n")
    
    splits_path = os.path.join(DATA_DIR, "splits_seed6304.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    
    test_dataset = datasets.STL10(root=DATA_DIR, split="test", download=False)
    test_subset = Subset(test_dataset, splits["test_subset_indices"])
    
    results = {}

    for model_name in ["resnet", "vit", "clip"]:
        norm_transform = get_normalization(model_name)
        full_transform = T.Compose([base_transform, norm_transform])
        
        eval_subset = STL10TransformWrapper(test_subset, full_transform)
        dataloader = DataLoader(eval_subset, batch_size=64, shuffle=False, num_workers=2)
        
        backbone = get_backbone(model_name, device)
        backbone.eval()
        
        head = LinearClassifierHead(backbone.feature_dim, num_classes=10).to(device)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_head_best.pth")
        head.load_state_dict(torch.load(checkpoint_path, map_location=device))
        head.eval()
        
        acc, macro_f1, mean_conf = evaluate_model(dataloader, backbone, head, device, is_zeroshot=False)
        results[model_name] = {"acc": acc, "macro_f1": macro_f1, "mean_conf": float(mean_conf)}
        
        print(f"--- {model_name.upper()} (Trained Head) ---")
        print(f"Top-1 Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Mean Max Conf: {mean_conf:.4f}\n")

    clip_norm = get_normalization("clip")
    clip_transform = T.Compose([base_transform, clip_norm])
    clip_subset = STL10TransformWrapper(test_subset, clip_transform)
    clip_loader = DataLoader(clip_subset, batch_size=64, shuffle=False, num_workers=2)
    
    clip_backbone = get_backbone("clip", device)
    clip_backbone.eval()
    
    acc, macro_f1, mean_conf = evaluate_model(clip_loader, clip_backbone, None, device, is_zeroshot=True)
    results["clip_zeroshot"] = {"acc": acc, "macro_f1": macro_f1, "mean_conf": float(mean_conf)}
    
    print(f"--- CLIP (Zero-Shot) ---")
    print(f"Top-1 Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Mean Max Conf: {mean_conf:.4f}\n")

    output_file = os.path.join(RESULTS_DIR, "clean_baseline.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Clean baseline results saved to {output_file}")

if __name__ == "__main__":
    main()
