import torch
import torch.nn as nn
import os
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_gcsc_train_loader(config):
    """Rebuilds the training loader with RandAugment inserted."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    full_train_dataset = datasets.CIFAR10(
        root=config["data_root"], train=True, download=True, transform=train_transform
    )
    
    split_file = f"./task4/data/cifar10_split_seed{config['seed']}.json"
    with open(split_file, "r") as f:
        splits = json.load(f)
        
    train_subset = Subset(full_train_dataset, splits["train"])
    return DataLoader(
        train_subset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

def train_gcsc(model, val_loader, optimizer, scheduler, config, device):
    """
    GCSC training loop: Identical to vanilla but uses the RandAugment loader.
    """
    train_loader = get_gcsc_train_loader(config)
    
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    epochs = config["epochs"]
    
    save_dir = "./task4/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "gcsc_best.pth")
    
    print(f"Starting GCSC training for {epochs} epochs...")
    
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
            
    print(f"GCSC training completed. Best Val Acc: {best_acc:.2f}%")
