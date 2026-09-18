import os
import json
import yaml
import numpy as np
from torchvision import datasets

def load_config(config_path="../configs/base.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    seed = config["seed"]
    train_split_ratio = config["dataset"]["train_val_split"]
    test_subset_size = config["dataset"]["subset_size"]
    
    # Initialize random state to guarantee exact reproducibility
    rng = np.random.RandomState(seed)
    
    # Ensure data directory exists
    os.makedirs("./", exist_ok=True)
    
    print("Downloading/Loading STL-10 dataset...")
    # STL-10 uses 'train' and 'test' for its labeled splits
    train_dataset = datasets.STL10(root="./", split="train", download=True)
    test_dataset = datasets.STL10(root="./", split="test", download=True)
    
    # 1. Stratified Train/Validation Split
    train_labels = np.array(train_dataset.labels)
    classes = np.unique(train_labels)
    
    train_indices = []
    val_indices = []
    
    for c in classes:
        c_indices = np.where(train_labels == c)[0]
        rng.shuffle(c_indices)
        
        split_idx = int(len(c_indices) * train_split_ratio)
        train_indices.extend(c_indices[:split_idx].tolist())
        val_indices.extend(c_indices[split_idx:].tolist())
        
    # 2. Class-Balanced Test Subset
    test_labels = np.array(test_dataset.labels)
    test_indices = []
    samples_per_class = test_subset_size // len(classes)
    
    for c in classes:
        c_indices = np.where(test_labels == c)[0]
        rng.shuffle(c_indices)
        
        # Select the required number of samples for the balanced subset
        test_indices.extend(c_indices[:samples_per_class].tolist())
        
    # Save the exact dataset identifiers so all models evaluate identical images
    splits = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_subset_indices": test_indices
    }
    
    output_file = f"./splits_seed{seed}.json"
    with open(output_file, "w") as f:
        json.dump(splits, f, indent=4)
        
    print(f"Splits saved to {output_file}")
    print(f"Train size: {len(train_indices)}")
    print(f"Val size:   {len(val_indices)}")
    print(f"Test size:  {len(test_indices)}")

if __name__ == "__main__":
    main()
