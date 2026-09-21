import json
import os
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_cifar10_loaders(config_path="./task4/configs/base.yaml"):
    """Returns Train, Validation, and Test loaders for the 10 known classes."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_root = config["data_root"]
    batch_size = config["batch_size"]
    seed = config["seed"]
    split_file = f"./task4/data/cifar10_split_seed{seed}.json"
    
    # Random crop to 32x32 with 4-pixel padding and horizontal flip
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    full_train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=eval_transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_transform)
    
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file missing at {split_file}. Run make_splits.py first.")
        
    with open(split_file, "r") as f:
        splits = json.load(f)
        
    train_subset = Subset(full_train_dataset, splits["train"])
    val_subset = Subset(val_dataset, splits["val"])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader
