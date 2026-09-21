import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .manifold_mixup import mixup_data

def proser_loss_cp(logits, targets, num_known=10):
    """Classifier Placeholder Loss: standard CE + dummy penalty."""
    # Standard Cross Entropy on known classes
    ce_loss = F.cross_entropy(logits[:, :num_known], targets)
    
    # Dummy penalty: encourage dummy classes to be second highest
    batch_size = logits.size(0)
    
    # Mask out the correct known class logit so it isn't penalized
    mask = torch.ones_like(logits[:, :num_known], dtype=torch.bool)
    mask[torch.arange(batch_size), targets] = False
    
    known_logits_except_target = logits[:, :num_known][mask].view(batch_size, num_known - 1)
    dummy_logits = logits[:, num_known:]
    
    # LogSumExp for numerical stability representing the negative log ratio
    lse_known = torch.logsumexp(known_logits_except_target, dim=1)
    lse_dummy = torch.logsumexp(dummy_logits, dim=1)
    
    combined = torch.stack([lse_dummy, lse_known], dim=1)
    dummy_loss = (torch.logsumexp(combined, dim=1) - lse_dummy).mean()
    
    return ce_loss, dummy_loss

def proser_loss_dp(logits, num_known=10):
    """Data Placeholder Loss: encourage mixed features to be classified as dummy."""
    lse_all = torch.logsumexp(logits, dim=1)
    lse_dummy = torch.logsumexp(logits[:, num_known:], dim=1)
    return (lse_all - lse_dummy).mean()

def train_proser(model, train_loader, val_loader, optimizer, scheduler, config, device):
    """PROSER training loop splitting batches and executing manifold mixup."""
    epochs = config["epochs"]
    num_known = 10
    beta = config["proser"]["beta"]
    gamma = config["proser"]["gamma"]
    
    save_dir = "./task4/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "proser_best.pth")
    
    vanilla_path = os.path.join(save_dir, "vanilla_best.pth")
    if not os.path.exists(vanilla_path):
        raise FileNotFoundError(f"Missing {vanilla_path}. Train Vanilla first.")
        
    vanilla_state = torch.load(vanilla_path, map_location=device, weights_only=True)
    
    # Carefully load the 10-class Vanilla weights into the 15-class PROSER model
    model_state = model.state_dict()
    for name, param in vanilla_state.items():
        if "fc" in name:
            model_state[name][:num_known] = param 
            # The appended 5 dummy classifiers retain their random initialization
        else:
            model_state[name] = param
            
    model.load_state_dict(model_state)
    print("Initialized PROSER from Vanilla checkpoint.")
    
    best_acc = 0.0
    print(f"Starting PROSER fine-tuning for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            if batch_size < 2:
                continue
                
            split_idx = batch_size // 2
            x1, y1 = inputs[:split_idx], targets[:split_idx]
            x2, y2 = inputs[split_idx:], targets[split_idx:]
            
            optimizer.zero_grad()
            
            # --- Part 1: Classifier Placeholders ---
            logits1 = model(x1)
            ce_loss, dummy_loss = proser_loss_cp(logits1, y1, num_known)
            loss_cp = ce_loss + beta * dummy_loss
            
            _, predicted = logits1[:, :num_known].max(1)
            total += y1.size(0)
            correct += predicted.eq(y1).sum().item()
            
            # --- Part 2: Data Placeholders (Manifold Mixup) ---
            with torch.no_grad():
                # Perform manifold mixup after layer2 and before layer3
                h = model.forward_up_to_layer2(x2)
            
            h.requires_grad_(True)
            mixed_h, _, _, _ = mixup_data(h, y2, alpha=2.0)
            
            logits2, _ = model.forward_from_layer3(mixed_h)
            loss_dp = proser_loss_dp(logits2, num_known)
            
            total_loss = loss_cp + gamma * loss_dp
            total_loss.backward()
            
            optimizer.step()
            
        scheduler.step()
        train_acc = 100. * correct / total
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                _, predicted = outputs[:, :num_known].max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best PROSER model to {save_path}")
