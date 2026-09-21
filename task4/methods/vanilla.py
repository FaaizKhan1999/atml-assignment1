import torch
import torch.nn as nn
import os

def train_vanilla(model, train_loader, val_loader, optimizer, scheduler, config, device):
    """
    Standard cross-entropy training loop for the CIFAR-10 known classes.
    """
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    epochs = config["epochs"]
    
    save_dir = "./task4/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vanilla_best.pth")
    
    print(f"Starting Vanilla training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        train_acc = 100. * correct / total
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path}")
            
    print(f"Vanilla training completed. Best Val Acc: {best_acc:.2f}%")
