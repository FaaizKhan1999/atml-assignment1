import torch
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
import json

from shared.pacs import PACSDataset
from task2.models.backbone import DomainAdaptationResNet
from task2.models.classifier_head import ClassifierHead
from task2.evaluation.domain_separability import compute_domain_separability

def get_eval_loaders():
    with open("shared/splits/pacs_sketch_seed6304.json", "r") as f:
        splits = json.load(f)
        
    source_val_loaders = {
        dom: DataLoader(PACSDataset(splits[dom]['val'], is_train=False), batch_size=64, shuffle=False)
        for dom in ['photo', 'art_painting', 'cartoon']
    }
    
    target_loader = DataLoader(PACSDataset(splits['sketch']['target'], is_train=False), batch_size=64, shuffle=False)
    
    return source_val_loaders, target_loader

def evaluate_model(checkpoint_path, source_val_loaders, target_loader, device):
    backbone = DomainAdaptationResNet().to(device)
    head = ClassifierHead().to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(ckpt['backbone'])
    head.load_state_dict(ckpt['head'])
    
    backbone.eval()
    head.eval()
    
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for imgs, lbls, _ in target_loader:
            imgs = imgs.to(device)
            features = backbone(imgs)
            logits = head(features)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_lbls.extend(lbls.numpy())
            
    target_acc = accuracy_score(all_lbls, all_preds)
    target_f1 = f1_score(all_lbls, all_preds, average='macro')
    report = classification_report(all_lbls, all_preds, output_dict=True)
    
    separability = compute_domain_separability(backbone, source_val_loaders, target_loader, device)
    
    return target_acc, target_f1, separability, report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    source_val_loaders, target_loader = get_eval_loaders()
    methods = ['source_only', 'dan', 'dann', 'cdan']
    
    results = {}
    
    for method in methods:
        ckpt_path = f'/kaggle/working/{method}_best.pt'
        try:
            acc, f1, sep, report = evaluate_model(ckpt_path, source_val_loaders, target_loader, args.device)
            results[method] = {
                'target_accuracy': acc,
                'target_macro_f1': f1,
                'domain_separability': sep,
                'per_class': {str(k): v['recall'] for k, v in report.items() if k.isdigit()}
            }
            print(f"--- {method.upper()} ---")
            print(f"Target Acc: {acc:.4f} | Target Macro-F1: {f1:.4f} | Separability: {sep:.4f}")
        except FileNotFoundError:
            print(f"Checkpoint for {method} not found at {ckpt_path}. Skipping.")
            
    with open('/kaggle/working/final_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
