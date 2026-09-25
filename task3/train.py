import os

import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, f1_score

# Assuming execution from the repository root
from shared.pacs_protocol import get_dg_loaders
from task3.methods.dan_dg import dan_dg_loss
from task3.methods.sam import SAM

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DomainGenModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.classifier_head = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier_head(features)
        return features, logits

def evaluate_sources(model, val_loaders, device):
    model.eval()
    domain_metrics = {}
    macro_f1s = []
    
    with torch.no_grad():
        for domain, loader in val_loaders.items():
            all_preds = []
            all_labels = []
            for images, labels, _ in loader:
                images, labels = images.to(device), labels.to(device)
                _, logits = model(images)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            acc = accuracy_score(all_labels, all_preds)
            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            
            domain_metrics[domain] = {'accuracy': acc, 'macro_f1': macro_f1}
            macro_f1s.append(macro_f1)
            
    mean_macro_f1 = np.mean(macro_f1s)
    return domain_metrics, mean_macro_f1

def main():
    parser = argparse.ArgumentParser(description="Task 3: Domain Generalization Training")
    parser.add_argument("--config", type=str, required=True, help="Path to method config YAML")
    args = parser.parse_args()

    # 1. Load Configurations
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(config['base_config'], 'r') as f:
        base_config = yaml.safe_load(f)
        
    # Merge configs (method config overrides base config)
    merged_config = {**base_config, **config}
    
    # 2. Setup
    set_seed(merged_config.get('seed', 6304))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('task3/checkpoints', exist_ok=True)
    
    # If this is the ERM baseline config, skip training
    if not merged_config.get('train', True):
        print(f"Skipping training for {merged_config['method']}. Use diagnostic scripts directly on {merged_config['checkpoint_name']}")
        return

    # 3. Data Loaders
    dg_train_iter, source_val_loaders, steps_per_epoch = get_dg_loaders(
        split_path="shared/splits/pacs_sketch_seed6304.json", 
        num_workers=merged_config['data'].get('num_workers', 4)
    )

    # 4. Model & Optimizer
    model = DomainGenModel(num_classes=merged_config['model']['num_classes']).to(device)
    
    base_optimizer = torch.optim.AdamW
    opt_kwargs = {
        'lr': merged_config['optimization']['lr'], 
        'weight_decay': merged_config['optimization']['weight_decay']
    }

    if merged_config['method'] == 'sam':
        optimizer = SAM(model.parameters(), base_optimizer, rho=merged_config['sam']['rho'], **opt_kwargs)
    else:
        optimizer = base_optimizer(model.parameters(), **opt_kwargs)

    # 5. Training Loop
    max_epochs = merged_config['optimization']['max_epochs']
    patience = merged_config['optimization']['early_stopping_patience']
    best_mean_f1 = -1.0
    epochs_without_improvement = 0
    save_path = os.path.join('task3/checkpoints', merged_config['checkpoint_name'])

    print(f"Starting training for {merged_config['method'].upper()} on {device}...")
    
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        
        # We manually step `steps_per_epoch` times since `dg_train_iter` is an infinite generator
        for step in range(steps_per_epoch):
            source_batches = next(dg_train_iter)
            
            # Combine the 3 source domains (8 imgs each) -> Batch of 24
            images = torch.cat([b[0] for b in source_batches], dim=0).to(device)
            labels = torch.cat([b[1] for b in source_batches], dim=0).to(device)
            
            # Assignment Constraint: Freeze BatchNorm running stats[cite: 1]
            model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
            
            if merged_config['method'] == 'sam':
                # --- SAM 2-Step Execution ---[cite: 1]
                # Pass 1
                features, logits = model(images)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # Pass 2
                features_adv, logits_adv = model(images)
                loss_adv = F.cross_entropy(logits_adv, labels)
                loss_adv.backward()
                optimizer.second_step(zero_grad=True)
                
                epoch_loss += loss.item()
                
            else:
                # --- DAN-DG Execution ---[cite: 1]
                features, logits = model(images)
                
                if merged_config['method'] == 'dan_dg':
                    loss, ce_loss, mmd_loss = dan_dg_loss(features, logits, labels, merged_config)
                else: # Fallback just in case
                    loss = F.cross_entropy(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

        # 6. Validation & Early Stopping[cite: 1]
        domain_metrics, mean_macro_f1 = evaluate_sources(model, source_val_loaders, device)
        avg_train_loss = epoch_loss / steps_per_epoch
        
        print(f"Epoch [{epoch+1}/{max_epochs}] - Train Loss: {avg_train_loss:.4f} | Mean Val Macro-F1: {mean_macro_f1:.4f}")
        for dom, mets in domain_metrics.items():
            print(f"   -> {dom.capitalize()}: Acc={mets['accuracy']:.4f}, F1={mets['macro_f1']:.4f}")

        if mean_macro_f1 > best_mean_f1:
            best_mean_f1 = mean_macro_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"[*] New best model saved to {save_path}")
        else:
            epochs_without_improvement += 1
            print(f"[!] No improvement for {epochs_without_improvement} epoch(s).")
            
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

if __name__ == "__main__":
    main()
