import os
import json
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from task4.models.resnet_cifar import CIFAR_ResNet18
from task4.data.cifar10 import get_cifar10_loaders
from task4.data.cifar100_unknowns import get_cifar100_unknown_loaders

def get_unaugmented_train_loader(config):
    """Mahalanobis scoring strictly requires unaugmented training features."""
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    full_train = datasets.CIFAR10(root=config["data_root"], train=True, download=False, transform=eval_transform)
    split_file = f"./task4/data/cifar10_split_seed{config['seed']}.json"
    
    with open(split_file, "r") as f:
        train_idx = json.load(f)["train"]
        
    train_subset = Subset(full_train, train_idx)
    return DataLoader(train_subset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

def extract_and_save(model, loader, device, save_path):
    """Runs inference and caches penultimate features and logits."""
    all_logits, all_features, all_targets = [], [], []
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits, features = model(inputs, return_features=True)
            
            all_logits.append(logits.cpu())
            all_features.append(features.cpu())
            all_targets.append(targets)
            
    torch.save({
        "logits": torch.cat(all_logits, dim=0),
        "features": torch.cat(all_features, dim=0),
        "targets": torch.cat(all_targets, dim=0)
    }, save_path)
    print(f"Saved {len(all_targets) * loader.batch_size} samples to {save_path}")

def main():
    with open("./task4/configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting on: {device}")
    
    # Load frozen Vanilla model
    model = CIFAR_ResNet18(num_classes=10)
    ckpt_path = "./task4/checkpoints/vanilla_best.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing {ckpt_path}. Run training first.")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    
    os.makedirs("./task4/cache", exist_ok=True)
    
    print("Loading datasets...")
    unaug_train_loader = get_unaugmented_train_loader(config)
    _, val_loader, test_loader = get_cifar10_loaders("./task4/configs/base.yaml")
    near_loader, far_loader = get_cifar100_unknown_loaders("./task4/configs/base.yaml")
    
    print("Extracting CIFAR-10 Unaugmented Train (for Mahalanobis)...")
    extract_and_save(model, unaug_train_loader, device, "./task4/cache/cifar10_train.pt")
    
    print("Extracting CIFAR-10 Validation (for threshold calibration)...")
    extract_and_save(model, val_loader, device, "./task4/cache/cifar10_val.pt")
    
    print("Extracting CIFAR-10 Test (Knowns)...")
    extract_and_save(model, test_loader, device, "./task4/cache/cifar10_test.pt")
    
    print("Extracting CIFAR-100 Near (Unknowns)...")
    extract_and_save(model, near_loader, device, "./task4/cache/cifar100_near.pt")
    
    print("Extracting CIFAR-100 Far (Unknowns)...")
    extract_and_save(model, far_loader, device, "./task4/cache/cifar100_far.pt")

if __name__ == "__main__":
    main()
