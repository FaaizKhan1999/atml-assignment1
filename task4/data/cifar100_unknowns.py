import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_cifar100_unknown_loaders(config_path="./task4/configs/base.yaml"):
    """Filters CIFAR-100 test classes into fixed near/far semantic unknown loaders."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_root = config["data_root"]
    batch_size = config["batch_size"]
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load ONLY the test partition to strictly prevent training leakage
    test_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=eval_transform)
    
    # Defined semantic groupings
    near_classes = ['bus', 'pickup_truck', 'motorcycle', 'tractor', 'wolf', 'fox', 'leopard', 'camel']
    far_classes = ['bottle', 'bowl', 'chair', 'clock', 'keyboard', 'mushroom', 'sunflower', 'wardrobe']
    
    class_to_idx = test_dataset.class_to_idx
    near_indices = {class_to_idx[c] for c in near_classes}
    far_indices = {class_to_idx[c] for c in far_classes}
    
    near_dataset_idx = [i for i, target in enumerate(test_dataset.targets) if target in near_indices]
    far_dataset_idx = [i for i, target in enumerate(test_dataset.targets) if target in far_indices]
    
    near_subset = Subset(test_dataset, near_dataset_idx)
    far_subset = Subset(test_dataset, far_dataset_idx)
    
    near_loader = DataLoader(near_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    far_loader = DataLoader(far_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return near_loader, far_loader
