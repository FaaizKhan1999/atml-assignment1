import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(TASK1_DIR, "data")
CANDIDATES_DIR = os.path.join(DATA_DIR, "cue_conflicts_candidates")
CHECKPOINT_DIR = os.path.join(TASK1_DIR, "checkpoints")
RESULTS_DIR = os.path.join(TASK1_DIR, "results")

import sys
sys.path.append(TASK1_DIR)
from data.transforms import base_transform, get_normalization
from models.backbones import get_backbone, LinearClassifierHead

class CueConflictDataset(torch.utils.data.Dataset):
    def __init__(self, folder, base_transform, norm_transform):
        self.folder = folder
        self.files = [f for f in os.listdir(folder) if f.endswith('.png')]
        self.base_transform = base_transform
        self.norm_transform = norm_transform
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        filename = self.files[idx]
        
        # Extract ground truth classes from the filename format: shape0_texture1_idx1_idx2.png
        parts = filename.split('_')
        shape_cls = int(parts[0].replace('shape', ''))
        texture_cls = int(parts[1].replace('texture', ''))
        
        img = Image.open(os.path.join(self.folder, filename)).convert('RGB')
        img = self.base_transform(img)
        img = self.norm_transform(img)
        
        return img, shape_cls, texture_cls

@torch.no_grad()
def evaluate_conflicts(dataloader, backbone, head, device, is_zeroshot=False):
    shape_matches = 0
    texture_matches = 0
    neither_matches = 0
    total = 0

    for x, shape_labels, texture_labels in dataloader:
        x = x.to(device)
        shape_labels = shape_labels.numpy()
        texture_labels = texture_labels.numpy()
        
        if is_zeroshot:
            confs = backbone.zero_shot_predict(x)
            _, preds = confs.max(dim=1)
        else:
            feats = backbone(x)
            logits = head(feats)
            _, preds = logits.max(dim=1)

        preds = preds.cpu().numpy()
        
        for p, s, t in zip(preds, shape_labels, texture_labels):
            if p == s:
                shape_matches += 1
            elif p == t:
                texture_matches += 1
            else:
                neither_matches += 1
            total += 1
            
    coverage = (shape_matches + texture_matches) / total
    
    # Shape bias is only calculated on images where the model recognized at least one of the cues
    valid_decisions = shape_matches + texture_matches
    shape_bias = shape_matches / valid_decisions if valid_decisions > 0 else 0.0
    
    return coverage, shape_bias, shape_matches, texture_matches, neither_matches

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Shape vs. Texture Bias on device: {device}\n")
    print(f"Loaded {len(os.listdir(CANDIDATES_DIR))} cue conflict images.\n")
    
    models_to_test = [
        ("resnet", False), 
        ("vit", False), 
        ("clip", False), 
        ("clip_zeroshot", True)
    ]
    
    results = {}

    for model_name, is_zeroshot in models_to_test:
        base_model_name = "clip" if is_zeroshot else model_name
        norm_transform = get_normalization(base_model_name)
        
        dataset = CueConflictDataset(CANDIDATES_DIR, base_transform, norm_transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
        
        backbone = get_backbone(base_model_name, device)
        backbone.eval()
        
        head = None
        if not is_zeroshot:
            head = LinearClassifierHead(backbone.feature_dim, num_classes=10).to(device)
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_head_best.pth")
            head.load_state_dict(torch.load(checkpoint_path, map_location=device))
            head.eval()
            
        cov, bias, s_match, t_match, n_match = evaluate_conflicts(dataloader, backbone, head, device, is_zeroshot)
        
        results[model_name] = {
            "coverage": float(cov),
            "shape_bias": float(bias),
            "raw_counts": {
                "shape_chosen": s_match,
                "texture_chosen": t_match,
                "neither_chosen": n_match
            }
        }
        
        print(f"--- {model_name.upper()} ---")
        print(f"Coverage:   {cov:.4f} (Model understood {s_match + t_match} out of {s_match + t_match + n_match} images)")
        print(f"Shape Bias: {bias:.4f} (Chose shape {s_match} times, chose texture {t_match} times)\n")

    output_file = os.path.join(RESULTS_DIR, "shape_bias_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
