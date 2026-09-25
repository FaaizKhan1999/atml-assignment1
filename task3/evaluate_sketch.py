import os
import json
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from shared.pacs import PACSDataset
from task3.train import DomainGenModel
from task3.evaluation.domain_metrics import compute_detailed_metrics

def evaluate_on_sketch(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            _, logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return compute_detailed_metrics(all_labels, all_preds, num_classes=7)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('task3/results', exist_ok=True)
    
    with open('task3/configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Strictly load only the Sketch indices for evaluation[cite: 1]
    with open("shared/splits/pacs_sketch_seed6304.json", "r") as f:
        splits = json.load(f)
        
    sketch_indices = splits['sketch']['target']
    sketch_dataset = PACSDataset(sketch_indices, is_train=False) # is_train=False applies center crop
    sketch_loader = DataLoader(sketch_dataset, batch_size=64, shuffle=False, num_workers=config['data'].get('num_workers', 4))
    
    checkpoints = {
        'ERM': 'task3/checkpoints/source_only_best.pt',
        'DAN-DG': 'task3/checkpoints/dan_dg_best.pt',
        'SAM 0.05': 'task3/checkpoints/sam_05_best.pt',
        'SAM 0.01': 'task3/checkpoints/sam_01_best.pt',
        'SAM 0.1': 'task3/checkpoints/sam_1_best.pt'
        }

    
    model = DomainGenModel(num_classes=config['model']['num_classes']).to(device)
    results = {}
    
    print(f"Running Unseen Target Evaluation (Sketch) on {device}...")
    
    for method_name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"[!] Warning: {ckpt_path} not found. Skipping {method_name}.")
            continue
            
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        metrics = evaluate_on_sketch(model, sketch_loader, device)
        
        results[method_name] = metrics
        
        print(f"\n[{method_name}] Sketch Performance:")
        print(f"  -> Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  -> Macro-F1: {metrics['macro_f1']:.4f}")
        
    # Compute relative changes against ERM[cite: 1]
    if 'ERM' in results:
        erm_acc = results['ERM']['accuracy']
        erm_per_class = results['ERM']['per_class_accuracy']
        
        print("\n=== Performance Changes Relative to ERM ===")
        for method in [k for k in results.keys() if k != 'ERM']:
            if method in results:
                delta_acc = results[method]['accuracy'] - erm_acc
                delta_per_class = results[method]['per_class_accuracy'] - erm_per_class
                
                print(f"\n[{method}] Overall Accuracy Delta: {delta_acc*100:+.2f}%")
                print(f"[{method}] Per-Class Deltas (Classes 0-6):")
                for c, d in enumerate(delta_per_class):
                    print(f"  Class {c}: {d*100:+.2f}%")
                    
                # Store deltas in results dictionary for saving
                results[method]['delta_acc_from_erm'] = float(delta_acc)
                results[method]['delta_per_class_from_erm'] = delta_per_class.tolist()
                
    # Prepare JSON serializable output
    serializable_results = {}
    for method, metrics in results.items():
        serializable_results[method] = {
            'accuracy': float(metrics['accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'per_class_accuracy': metrics['per_class_accuracy'].tolist(),
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
        }
        if 'delta_acc_from_erm' in metrics:
            serializable_results[method]['delta_acc_from_erm'] = metrics['delta_acc_from_erm']
            serializable_results[method]['delta_per_class_from_erm'] = metrics['delta_per_class_from_erm']

    with open('task3/results/sketch_evaluation.json', 'w') as f:
        json.dump(serializable_results, f, indent=4)
    print("\n[*] Final Sketch metrics successfully saved to task3/results/sketch_evaluation.json")

if __name__ == "__main__":
    main()
