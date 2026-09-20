import os
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets
import torchvision.transforms as T

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(TASK1_DIR, "configs", "base.yaml")
SPLITS_PATH = os.path.join(TASK1_DIR, "data", "splits_seed6304.json")
DATA_ROOT = os.path.join(TASK1_DIR, "data")
CHECKPOINT_DIR = os.path.join(TASK1_DIR, "checkpoints")

import sys
sys.path.append(TASK1_DIR)
from data.transforms import base_transform, get_normalization
from models.backbones import get_backbone, LinearClassifierHead

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def extract_features(dataset, backbone, device, batch_size=128):
    """Precomputes features to accelerate linear head training."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    features, labels = [], []
    for x, y in loader:
        x = x.to(device)
        feats = backbone(x)
        features.append(feats.cpu())
        labels.append(y)
    return torch.cat(features), torch.cat(labels)

def train_and_evaluate(model_name, train_dataset, val_dataset, config, device):
    print(f"\n--- Training Linear Head for {model_name.upper()} ---")
    
    # Setup Backbone and Precompute Features
    backbone = get_backbone(model_name, device)
    backbone.eval()
    
    print("Extracting training features...")
    train_feats, train_labels = extract_features(train_dataset, backbone, device)
    print("Extracting validation features...")
    val_feats, val_labels = extract_features(val_dataset, backbone, device)
    
    train_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_feats, val_labels), batch_size=128, shuffle=False)
    
    # Initialize Head, Optimizer, and Loss
    torch.manual_seed(config["seed"]) 
    head = LinearClassifierHead(backbone.feature_dim, num_classes=10).to(device)
    
    optimizer = torch.optim.AdamW(
        head.parameters(), 
        lr=config["training"]["learning_rate"], 
        weight_decay=config["training"]["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()
    
    max_epochs = config["training"]["max_epochs"]
    patience_limit = config["training"]["early_stopping_patience"]
    
    best_val_acc = 0.0
    patience_counter = 0
    best_state_dict = None
    
    # Training Loop with Early Stopping
    for epoch in range(1, max_epochs + 1):
        head.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = head(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * feats.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += feats.size(0)
            
        head.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                outputs = head(feats)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += feats.size(0)
                
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch:02d} | Train Acc: {train_correct/train_total:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state_dict = head.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered at epoch {epoch}.")
            break
            
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_head_best.pth")
    torch.save(best_state_dict, save_path)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}. Saved to {save_path}")

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(SPLITS_PATH, "r") as f:
        splits = json.load(f)
        
    base_dataset = datasets.STL10(root=DATA_ROOT, split="train", download=False)
    
    class STL10TransformWrapper(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __len__(self):
            return len(self.subset)
            
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            img = self.transform(img)
            return img, label

    for model_name in ["resnet", "vit", "clip"]:
        norm_transform = get_normalization(model_name)
        full_transform = T.Compose([base_transform, norm_transform])
        
        train_subset = STL10TransformWrapper(Subset(base_dataset, splits["train_indices"]), full_transform)
        val_subset = STL10TransformWrapper(Subset(base_dataset, splits["val_indices"]), full_transform)
        
        train_and_evaluate(model_name, train_subset, val_subset, config, device)

if __name__ == "__main__":
    main()
