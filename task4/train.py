import argparse
import yaml
import torch
import torch.optim as optim
import random
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from task4.models.resnet_cifar import CIFAR_ResNet18
from task4.data.cifar10 import get_cifar10_loaders
from task4.methods.vanilla import train_vanilla
from task4.methods.gcsc import train_gcsc
from task4.methods.proser import train_proser
from task4.methods.rpl import train_rpl

def set_seed(seed):
    """Enforces reproducibility for all randomized processes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Task 4 OSR Training")
    parser.add_argument("--config", type=str, required=True, help="Path to the method-specific YAML config")
    args = parser.parse_args()
    
    # Load and merge base.yaml with the method-specific yaml
    with open("./task4/configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(args.config, "r") as f:
        method_config = yaml.safe_load(f)
    config.update(method_config)
    
    # Fix seed to 6304 as required
    set_seed(config["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")
    
    # DataLoaders: Defaults to checking base.yaml internally
    train_loader, val_loader, _ = get_cifar10_loaders("./task4/configs/base.yaml")
    
    # Model Setup
    model = CIFAR_ResNet18(num_classes=config["num_classes"]).to(device)
    
    # Optimizer configuration using parameters mandated by the assignment
    opt_cfg = config["optimizer"]
    optimizer = optim.SGD(
        model.parameters(), 
        lr=opt_cfg["lr"], 
        momentum=opt_cfg["momentum"], 
        weight_decay=opt_cfg["weight_decay"]
    )
    
    # Cosine decay schedule over the full epoch count
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    # Method Dispatch
    method = config["method"]
    if method == "vanilla":
        train_vanilla(model, train_loader, val_loader, optimizer, scheduler, config, device)
    elif method == "gcsc":
        train_gcsc(model, val_loader, optimizer, scheduler, config, device)
    elif method == "proser":
        train_proser(model, train_loader, val_loader, optimizer, scheduler, config, device)
    elif method == "rpl":
        train_rpl(model, train_loader, val_loader, optimizer, scheduler, config, device)
    else:
        raise ValueError(f"Unknown method specified: {method}")

if __name__ == "__main__":
    main()
