import os
import json
import torch
import yaml
from collections import defaultdict

# Assuming execution from the repository root
from shared.pacs_protocol import get_dg_loaders
from task3.train import DomainGenModel
from task3.evaluation.source_domain_separability import compute_domain_separability
from task3.evaluation.sharpness import compute_sharpness_proxy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('task3/results', exist_ok=True)
    
    # Load base config for structural settings
    with open('task3/configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    _, source_val_loaders, _ = get_dg_loaders(
        split_path="shared/splits/pacs_sketch_seed6304.json", 
        num_workers=config['data'].get('num_workers', 4)
    )
    
    checkpoints = {
        'ERM': 'task3/checkpoints/source_only_best.pt',
        'DAN-DG': 'task3/checkpoints/dan_dg_best.pt',
        'SAM 0.05': 'task3/checkpoints/sam_05_best.pt',
        'SAM 0.01': 'task3/checkpoints/sam_01_best.pt',
        'SAM 0.1': 'task3/checkpoints/sam_1_best.pt'
    }
    
    results = defaultdict(dict)
    model = DomainGenModel(num_classes=config['model']['num_classes']).to(device)
    
    print(f"Running source-side diagnostics on {device}...")
    
    for method_name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"[!] Warning: {ckpt_path} not found. Skipping {method_name}.")
            continue
            
        print(f"\nEvaluating {method_name}...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        # 1. Source Domain Separability (Lower is better invariance)
        separability = compute_domain_separability(
            model, source_val_loaders, device, seed=config['seed']
        )
        print(f"  -> Domain Separability (Chance=33.3%): {separability * 100:.2f}%")
        
        # 2. Local Sharpness Proxy (Lower means flatter minima)[cite: 1]
        delta_sharp, loss_base, loss_adv = compute_sharpness_proxy(
            model, source_val_loaders, device, 
            rho=config['diagnostics']['sharpness_rho'], 
            seed=config['seed']
        )
        print(f"  -> Sharpness (Delta Loss): {delta_sharp:.4f} (Base: {loss_base:.4f} -> Adv: {loss_adv:.4f})")
        
        results[method_name] = {
            'domain_separability': float(separability),
            'delta_sharp': float(delta_sharp),
            'base_val_loss': float(loss_base),
            'adv_val_loss': float(loss_adv)
        }
        
    with open('task3/results/diagnostics_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\n[*] Diagnostics successfully saved to task3/results/diagnostics_results.json")

if __name__ == "__main__":
    main()
