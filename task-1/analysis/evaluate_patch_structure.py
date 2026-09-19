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
from data.transforms import base_transform, get_normalization, patch_shuffle_intervention
from models.backbones import get_backbone, LinearClassifierHead

class STL10CleanWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, base_transform, norm_transform):
        self.subset = subset
        self.base_transform = base_transform
        self.norm_transform = norm_transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = self.base_transform(img)
        img = self.norm_transform(img)
        return img, label

class STL10PatchWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, base_transform, norm_transform, seed=6304):
        self.subset = subset
        self.base_transform = base_transform
        self.norm_transform = norm_transform
        
        # Pre-calculate 500 deterministic, non-identity permutations
        rng = np.random.RandomState(seed)
        self.permutations = []
        identity = np.arange(16)
        
        for _ in range(len(subset)):
            perm = rng.permutation(16)
            while np.array_equal(perm, identity):
                perm = rng.permutation(16)
            self.permutations.append(perm)
            
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = self.base_transform(img)
        img = patch_shuffle_intervention(img, self.permutations[idx])
        img = self.norm_transform(img)
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
    print(f"Evaluating Patch Structure Invariance on device: {device}\n")
    
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
    
    results = {}

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
            
        clean_ds = STL10CleanWrapper(test_subset, base_transform, norm_transform)
        patch_ds = STL10PatchWrapper(test_subset, base_transform, norm_transform, seed=6304)
        
        clean_loader = DataLoader(clean_ds, batch_size=64, shuffle=False, num_workers=2)
        patch_loader = DataLoader(patch_ds, batch_size=64, shuffle=False, num_workers=2)
        
        clean_preds, labels = get_predictions(clean_loader, backbone, head, device, is_zeroshot)
        patch_preds, _ = get_predictions(patch_loader, backbone, head, device, is_zeroshot)
        
        clean_acc = accuracy_score(labels, clean_preds)
        patch_acc = accuracy_score(labels, patch_preds)
        
        acc_drop = patch_acc - clean_acc
        consistency = np.mean(clean_preds == patch_preds)
        
        results[model_name] = {
            "clean_acc": float(clean_acc),
            "patch_acc": float(patch_acc),
            "acc_drop": float(acc_drop),
            "consistency": float(consistency)
        }
        
        print(f"Clean Acc: {clean_acc:.4f} | Patch Acc: {patch_acc:.4f}")
        print(f"Accuracy Drop: {acc_drop:+.4f} | Consistency: {consistency:.4f}\n")

    output_file = os.path.join(RESULTS_DIR, "patch_structure_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
