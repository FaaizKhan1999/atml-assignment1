import json
import os
import yaml
from torchvision import datasets
from sklearn.model_selection import train_test_split

def generate_splits(config_path="./task4/configs/base.yaml", save_dir="./task4/data"):
    """Creates a 90/10 stratified split of CIFAR-10 training data."""

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_root = config["data_root"]
    seed = config["seed"]

    os.makedirs(data_root, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = datasets.CIFAR10(root=data_root, train=True, download=True)
    targets = dataset.targets
    indices = list(range(len(targets)))
    
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=seed, stratify=targets
    )
    
    split_file = os.path.join(save_dir, f"cifar10_split_seed{seed}.json")
    with open(split_file, "w") as f:
        json.dump({"train": train_idx, "val": val_idx}, f)
    
    print(f"Splits generated and saved to {split_file}")

if __name__ == "__main__":
    generate_splits()
