import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

@torch.no_grad()
def get_prediction(img, backbone, head, norm_transform, device, is_zeroshot=False):
    x = norm_transform(base_transform(img)).unsqueeze(0).to(device)
    if is_zeroshot:
        confs = backbone.zero_shot_predict(x)
        _, pred = confs.max(dim=1)
    else:
        feats = backbone(x)
        logits = head(feats)
        _, pred = logits.max(dim=1)
    return pred.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Scanning all images for hardest failures on device: {device}")

    models = [("resnet", False), ("vit", False), ("clip", False)]
    loaded_models = {}

    for model_name, is_zs in models:
        norm = get_normalization(model_name)
        backbone = get_backbone(model_name, device)
        backbone.eval()
        
        head = LinearClassifierHead(backbone.feature_dim, num_classes=10).to(device)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_head_best.pth")
        head.load_state_dict(torch.load(checkpoint_path, map_location=device))
        head.eval()
        
        loaded_models[model_name] = (backbone, head, norm)

    all_files = [f for f in os.listdir(CANDIDATES_DIR) if f.endswith('.png')]
    rng = np.random.RandomState(6304)
    rng.shuffle(all_files)

    agreements = []
    disagreements = []
    
    total_failures = []    # 3 models failed
    majority_failures = [] # 2 models failed
    single_failures = []   # 1 model failed

    print("Evaluating entire candidate folder...")
    for filename in all_files:
        img_path = os.path.join(CANDIDATES_DIR, filename)
        img = Image.open(img_path).convert('RGB')
        
        parts = filename.split('_')
        shape_idx = int(parts[0].replace('shape', ''))
        texture_idx = int(parts[1].replace('texture', ''))
        
        preds_idx = {}
        for m_name, (b_bone, head, norm) in loaded_models.items():
            preds_idx[m_name] = get_prediction(img, b_bone, head, norm, device)
            
        p_res = preds_idx['resnet']
        p_vit = preds_idx['vit']
        p_clip = preds_idx['clip']
        
        data_dict = {
            'img': img,
            'shape_cls': STL10_CLASSES[shape_idx],
            'texture_cls': STL10_CLASSES[texture_idx],
            'preds_str': {m: STL10_CLASSES[p] for m, p in preds_idx.items()}
        }
        
        valid_cues = [shape_idx, texture_idx]
        fail_count = sum(p not in valid_cues for p in [p_res, p_vit, p_clip])
        
        if fail_count == 3:
            total_failures.append(data_dict)
        elif fail_count == 2:
            majority_failures.append(data_dict)
        elif fail_count == 1:
            single_failures.append(data_dict)
        elif p_res == p_vit == p_clip:
            agreements.append(data_dict)
        else:
            disagreements.append(data_dict)

    # Assemble the final 5 images based on severity
    selected_failures = (total_failures + majority_failures + single_failures)[:2]
    selected_disagreements = disagreements[:2]
    selected_agreements = agreements[:1]
    
    selected = selected_agreements + selected_disagreements + selected_failures
    categories = (["Agreement"] * len(selected_agreements) + 
                  ["Disagreement"] * len(selected_disagreements) + 
                  ["Failure"] * len(selected_failures))

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    for i, data in enumerate(selected):
        ax = axes[i]
        ax.imshow(data['img'])
        ax.axis('off')
        
        title = f"[{categories[i].upper()}]\n"
        title += f"True Shape: {data['shape_cls']}\nTrue Texture: {data['texture_cls']}\n\n"
        title += f"ResNet: {data['preds_str']['resnet']}\n"
        title += f"ViT: {data['preds_str']['vit']}\n"
        title += f"CLIP: {data['preds_str']['clip']}"
        
        ax.set_title(title, fontsize=11, loc='left', pad=10)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "cue_conflict_visuals.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"\nFound {len(total_failures)} Total Failures, {len(majority_failures)} Majority Failures.")
    print(f"Saved specific outcome visual analysis to {plot_path}")

if __name__ == "__main__":
    main()
