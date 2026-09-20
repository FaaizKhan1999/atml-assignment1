import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision.transforms.functional as T_F
import torch.nn.functional as F
from sklearn.manifold import TSNE
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(TASK1_DIR, "data")
CANDIDATES_DIR = os.path.join(DATA_DIR, "cue_conflicts_candidates")
RESULTS_DIR = os.path.join(TASK1_DIR, "results")

import sys
sys.path.append(TASK1_DIR)
from data.transforms import (
    base_transform, get_normalization, grayscale_intervention, 
    ReflectionTranslate, patch_shuffle_intervention
)
from models.backbones import get_backbone

# --- Dataset Wrappers ---

class InterventionWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, base_transform, intervention, norm_transform):
        self.subset = subset
        self.base_transform = base_transform
        self.intervention = intervention
        self.norm_transform = norm_transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = self.base_transform(img)
        if self.intervention:
            img = self.intervention(img)
        img = self.norm_transform(img)
        return img, label

class PatchWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, base_transform, norm_transform, seed=6304):
        self.subset = subset
        self.base_transform = base_transform
        self.norm_transform = norm_transform
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

class CueConflictFeatureWrapper(torch.utils.data.Dataset):
    def __init__(self, folder, base_transform, norm_transform, subset_idx_map):
        self.folder = folder
        self.files = [f for f in os.listdir(folder) if f.endswith('.png')]
        self.base_transform = base_transform
        self.norm_transform = norm_transform
        self.subset_idx_map = subset_idx_map
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        filename = self.files[idx]
        parts = filename.split('_')
        shape_cls = int(parts[0].replace('shape', ''))
        c_idx = int(parts[2])
        
        subset_idx = self.subset_idx_map[c_idx]
        
        img = Image.open(os.path.join(self.folder, filename)).convert('RGB')
        img = self.base_transform(img)
        img = self.norm_transform(img)
        
        return img, shape_cls, subset_idx

@torch.no_grad()
def get_features(dataloader, backbone, device, is_cue_conflict=False):
    features = []
    labels = []
    subset_indices = []
    
    for batch in dataloader:
        if is_cue_conflict:
            x, y, s_idx = batch
            subset_indices.extend(s_idx.numpy())
        else:
            x, y = batch
            
        x = x.to(device)
        feat = backbone(x)
        # Flatten features (e.g., for ResNet pooling)
        feat = feat.view(feat.size(0), -1)
        
        features.append(feat.cpu())
        labels.extend(y.numpy())
        
    features = torch.cat(features, dim=0)
    labels = np.array(labels)
    
    if is_cue_conflict:
        return features, labels, np.array(subset_indices)
    return features, labels

def calculate_stability(f_clean, f_trans):
    """Calculates Cosine Stability I_T exactly according to the formula."""
    return F.cosine_similarity(f_clean, f_trans, dim=-1).mean().item()

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Representation Analysis on device: {device}\n")
    
    splits_path = os.path.join(DATA_DIR, "splits_seed6304.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    
    test_subset_indices = splits["test_subset_indices"]
    subset_idx_map = {orig_idx: i for i, orig_idx in enumerate(test_subset_indices)}
    
    test_dataset = datasets.STL10(root=DATA_DIR, split="test", download=False)
    test_subset = Subset(test_dataset, test_subset_indices)
    
    # We only need to evaluate the core visual representations, zero-shot head is irrelevant here
    models_to_test = ["resnet", "vit", "clip"]
    
    results = {}

    for model_name in models_to_test:
        print(f"--- Extracting Features: {model_name.upper()} ---")
        norm_transform = get_normalization(model_name)
        backbone = get_backbone(model_name, device)
        backbone.eval()
        
        # Initialize datasets
        clean_ds = InterventionWrapper(test_subset, base_transform, None, norm_transform)
        gray_ds = InterventionWrapper(test_subset, base_transform, grayscale_intervention, norm_transform)
        trans_ds = InterventionWrapper(test_subset, base_transform, ReflectionTranslate(16, 'up'), norm_transform)
        patch_ds = PatchWrapper(test_subset, base_transform, norm_transform, seed=6304)
        cue_ds = CueConflictFeatureWrapper(CANDIDATES_DIR, base_transform, norm_transform, subset_idx_map)
        
        # Loaders
        clean_loader = DataLoader(clean_ds, batch_size=64, shuffle=False)
        gray_loader = DataLoader(gray_ds, batch_size=64, shuffle=False)
        trans_loader = DataLoader(trans_ds, batch_size=64, shuffle=False)
        patch_loader = DataLoader(patch_ds, batch_size=64, shuffle=False)
        cue_loader = DataLoader(cue_ds, batch_size=64, shuffle=False)
        
        # Extract all features
        f_clean, labels_clean = get_features(clean_loader, backbone, device)
        f_gray, _ = get_features(gray_loader, backbone, device)
        f_trans, _ = get_features(trans_loader, backbone, device)
        f_patch, _ = get_features(patch_loader, backbone, device)
        f_cue, labels_cue, cue_clean_indices = get_features(cue_loader, backbone, device, is_cue_conflict=True)
        
        # Calculate Cosine Stabilities
        I_gray = calculate_stability(f_clean, f_gray)
        I_trans = calculate_stability(f_clean, f_trans)
        I_patch = calculate_stability(f_clean, f_patch)
        
        # Cue conflict requires matching the curated subset to their clean pairs
        f_clean_paired_for_cue = f_clean[cue_clean_indices]
        I_cue = calculate_stability(f_clean_paired_for_cue, f_cue)
        
        results[model_name] = {
            "cosine_stability": {
                "grayscale": I_gray,
                "translation_d16": I_trans,
                "patch_shuffle": I_patch,
                "cue_conflict": I_cue
            }
        }
        print(f"Grayscale: {I_gray:.4f} | Translation: {I_trans:.4f} | Patch: {I_patch:.4f} | Cue: {I_cue:.4f}")
        
        # --- t-SNE Projection ---
        print(f"Fitting t-SNE for {model_name.upper()}... This may take a minute.")
        X_all = torch.cat([f_clean, f_gray, f_trans, f_patch, f_cue], dim=0).numpy()
        labels_all = np.concatenate([labels_clean, labels_clean, labels_clean, labels_clean, labels_cue])
        
        tsne = TSNE(n_components=2, random_state=6304, init='pca', learning_rate='auto')
        X_2d = tsne.fit_transform(X_all)
        
        # Slicing back the 2D coordinates
        n = len(f_clean)
        t_clean = X_2d[0:n]
        t_gray = X_2d[n:2*n]
        t_trans = X_2d[2*n:3*n]
        t_patch = X_2d[3*n:4*n]
        t_cue = X_2d[4*n:]
        
        # --- Plotting ---
        fig, axs = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f"{model_name.upper()} Representation Stability (t-SNE)", fontsize=18, fontweight='bold')
        
        cmap = plt.get_cmap('tab10')
        interventions = [
            (t_gray, "Grayscale", axs[0, 0]),
            (t_cue, "Cue Conflict", axs[0, 1]),
            (t_trans, "Translation (\u03B4=16)", axs[1, 0]),
            (t_patch, "Patch Shuffling", axs[1, 1])
        ]
        
        for t_feat, title, ax in interventions:
            ax.set_title(title, fontsize=14)
            # Plot transformed (X) first so clean (o) dots rest clearly on top
            ax.scatter(t_feat[:, 0], t_feat[:, 1], c=labels_all[-len(t_feat):], cmap=cmap, marker='X', s=50, alpha=0.5, label='Transformed')
            scatter = ax.scatter(t_clean[:, 0], t_clean[:, 1], c=labels_clean, cmap=cmap, marker='o', s=40, edgecolors='k', linewidth=0.5, alpha=0.9, label='Clean')
            ax.axis('off')
            
        # Unified legend for conditions
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12, markerscale=1.5)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plot_path = os.path.join(RESULTS_DIR, f"{model_name}_tsne.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Saved t-SNE plot to {plot_path}\n")

    json_path = os.path.join(RESULTS_DIR, "representation_stability.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"All representation analysis results saved to {json_path}")

if __name__ == "__main__":
    main()
