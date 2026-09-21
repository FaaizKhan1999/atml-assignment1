import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ReciprocalPoints(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, embed_dim))
        
    def forward(self, x):
        """Computes the squared distance from features to all reciprocal points."""
        x_sq = torch.sum(x ** 2, dim=1, keepdim=True)
        c_sq = torch.sum(self.centers ** 2, dim=1).unsqueeze(0)
        xc = torch.matmul(x, self.centers.t())
        return x_sq + c_sq - 2 * xc

def rpl_loss(distances, targets, margin=1.0, weight=0.1):
    """Distance-based classification + Open Space Regularization."""
    ce_loss = F.cross_entropy(distances, targets)
    
    open_space_reg = torch.mean(torch.relu(distances - margin))
    
    return ce_loss + weight * open_space_reg

def train_rpl(model, train_loader, val_loader, optimizer, scheduler, config, device):
    """RPL training loop using distance evaluation instead of a linear classifier."""
    epochs = config["epochs"]
    embed_dim = model.backbone.fc.in_features
    
    rp_module = ReciprocalPoints(num_classes=10, embed_dim=embed_dim).to(device)
    optimizer.add_param_group({'params': rp_module.parameters()})
    
    save_dir = "./task4/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "rpl_best.pth")
    
    best_acc = 0.0
    print(f"Starting RPL training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        rp_module.train()
        correct, total = 0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            _, features = model(inputs, return_features=True)
            distances = rp_module(features)
            
            loss = rpl_loss(distances, targets)
            loss.backward()
            optimizer.step()
            
            _, predicted = distances.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        train_acc = 100. * correct / total
        
        # Validation Phase
        model.eval()
        rp_module.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, features = model(inputs, return_features=True)
                distances = rp_module(features)
                
                _, predicted = distances.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'rp_module': rp_module.state_dict()
            }, save_path)
