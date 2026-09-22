import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
from sklearn.metrics import f1_score
import json
import os

from shared.pacs_protocol import get_uda_loaders
from task2.models.backbone import DomainAdaptationResNet
from task2.models.classifier_head import ClassifierHead
from task2.models.domain_discriminator import DomainDiscriminator
from task2.models.grl import grad_reverse
from task2.methods.mmd import mmd_rbf_loss

def set_seed(seed=6304):
    """Enforce strict reproducibility across all operations."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(args, backbone, head, discriminator, uda_iter, optimizer, epoch, steps_per_epoch):
    backbone.train() 
    head.train()
    if discriminator:
        discriminator.train()
        
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    for step in range(steps_per_epoch):
        source_batches, target_batch = next(uda_iter)
        
        # Aggregate the 3 source domains (24 images total)
        src_imgs = torch.cat([b[0] for b in source_batches], dim=0).to(args.device)
        src_lbls = torch.cat([b[1] for b in source_batches], dim=0).to(args.device)
        
        tgt_imgs = target_batch[0].to(args.device)
        
        optimizer.zero_grad()
        
        # 1. Classification Forward Pass (Source Only)
        src_features = backbone(src_imgs)
        src_logits = head(src_features)
        loss_cls = ce_loss(src_logits, src_lbls)
        total_loss = loss_cls
        
        # 2. Adaptation Objectives
        if args.method == 'dan':
            tgt_features = backbone(tgt_imgs)
            loss_mmd = args.lambda_mmd * mmd_rbf_loss(src_features, tgt_features)
            total_loss += loss_mmd
            
        elif args.method in ['dann', 'cdan']:
            tgt_features = backbone(tgt_imgs)
            
            # Strict gradient reversal schedule: alpha(p) = (2 / (1 + exp(-10p))) - 1
            p = (epoch * steps_per_epoch + step) / (args.epochs * steps_per_epoch)
            alpha = args.max_grl_strength * ((2.0 / (1.0 + np.exp(-10 * p))) - 1.0)
            
            src_feat_rev = grad_reverse(src_features, alpha)
            tgt_feat_rev = grad_reverse(tgt_features, alpha)
            
            if args.method == 'cdan':
                # Tensor product g(x) = vec(f x p)
                src_probs = torch.softmax(src_logits.detach(), dim=1)
                src_disc_input = torch.bmm(src_feat_rev.unsqueeze(2), src_probs.unsqueeze(1)).view(src_feat_rev.size(0), -1)
                
                tgt_logits = head(tgt_features)
                tgt_probs = torch.softmax(tgt_logits.detach(), dim=1)
                tgt_disc_input = torch.bmm(tgt_feat_rev.unsqueeze(2), tgt_probs.unsqueeze(1)).view(tgt_feat_rev.size(0), -1)
            else:
                src_disc_input = src_feat_rev
                tgt_disc_input = tgt_feat_rev
                
            disc_input = torch.cat([src_disc_input, tgt_disc_input], dim=0)
            
            # Domain Labels: Source = 1, Target = 0
            domain_labels = torch.cat([
                torch.ones(src_imgs.size(0), 1),
                torch.zeros(tgt_imgs.size(0), 1)
            ], dim=0).to(args.device)
            
            disc_logits = discriminator(disc_input)
            loss_domain = bce_loss(disc_logits, domain_labels)
            total_loss += loss_domain

        total_loss.backward()
        optimizer.step()

def validate(backbone, head, val_loaders, device):
    """Evaluates on each source domain separately and returns the mean macro-F1."""
    backbone.eval()
    head.eval()
    
    domain_f1_scores = {}
    
    with torch.no_grad():
        for domain, loader in val_loaders.items():
            all_preds, all_lbls = [], []
            for imgs, lbls, _ in loader:
                imgs = imgs.to(device)
                features = backbone(imgs)
                logits = head(features)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_lbls.extend(lbls.numpy())
                
            macro_f1 = f1_score(all_lbls, all_preds, average='macro')
            domain_f1_scores[domain] = macro_f1
            
    mean_f1 = np.mean(list(domain_f1_scores.values()))
    return mean_f1, domain_f1_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['source_only', 'dan', 'dann', 'cdan'], required=True)
    parser.add_argument('--lambda_mmd', type=float, default=1.0)
    parser.add_argument('--max_grl_strength', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Fixed assignment constraints
    args.epochs = 30
    args.lr = 1e-4
    args.wd = 1e-4
    args.seed = 6304
    
    set_seed(args.seed)

    backbone = DomainAdaptationResNet().to(args.device)
    head = ClassifierHead().to(args.device)
    
    discriminator = None
    if args.method in ['dann', 'cdan']:
        in_feat = 512 * 7 if args.method == 'cdan' else 512
        discriminator = DomainDiscriminator(in_features=in_feat).to(args.device)

    params = list(backbone.parameters()) + list(head.parameters())
    if discriminator:
        params += list(discriminator.parameters())
        
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    
    uda_iter, val_loaders, steps_per_epoch = get_uda_loaders()
    
    best_f1 = 0.0
    patience_counter = 0
    save_path = f'/kaggle/working/{args.method}_best.pt'
    history = []
    
    print(f"Starting {args.method.upper()} Training...")
    print(f"Steps per epoch resolved to: {steps_per_epoch}")
    
    for epoch in range(args.epochs):
        train_epoch(args, backbone, head, discriminator, uda_iter, optimizer, epoch, steps_per_epoch)
        mean_val_f1, domain_scores = validate(backbone, head, val_loaders, args.device)
        
        log_entry = {
            "epoch": epoch + 1,
            "mean_macro_f1": float(mean_val_f1),
            "domains": {k: float(v) for k, v in domain_scores.items()}
        }
        history.append(log_entry)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Mean Val F1: {mean_val_f1:.4f} | "
              f"Photo: {domain_scores['photo']:.4f}, Art: {domain_scores['art_painting']:.4f}, Cartoon: {domain_scores['cartoon']:.4f}")
        
        if mean_val_f1 > best_f1:
            best_f1 = mean_val_f1
            patience_counter = 0
            torch.save({'backbone': backbone.state_dict(), 'head': head.state_dict()}, save_path)
            print(f"  -> Checkpoint saved to {save_path}")
        else:
            patience_counter += 1
            
        if patience_counter >= 5:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Mean Macro-F1: {best_f1:.4f}")
            break
            
    # Save metrics history
    with open(f'/kaggle/working/{args.method}_history.json', 'w') as f:
        json.dump(history, f, indent=4)
