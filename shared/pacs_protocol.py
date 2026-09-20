import json
import torch
from torch.utils.data import DataLoader
from shared.pacs import PACSDataset, get_pacs_transforms

SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"

def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch

class MultiSourceBatchIterator:
    def __init__(self, loaders: list[DataLoader]):
        self.loaders = loaders
        self.iterators = [cycle_loader(loader) for loader in loaders]
        self.max_len = max(len(loader) for loader in loaders)

    def __iter__(self):
        self.current_step = 0
        return self

    def __next__(self):
        if self.current_step >= self.max_len:
            raise StopIteration
        self.current_step += 1

        batches = [next(it) for it in self.iterators]
        return {
            "image": torch.cat([b["image"] for b in batches], dim=0),
            "label": torch.cat([b["label"] for b in batches], dim=0),
            "domain": torch.cat([b["domain"] for b in batches], dim=0),
        }
        
    def __len__(self):
        return self.max_len

def load_splits(json_path: str = "shared/splits/pacs_sketch_seed6304.json") -> dict:
    with open(json_path, "r") as f:
        return json.load(f)

def get_source_train_loader(json_path: str = "shared/splits/pacs_sketch_seed6304.json"):
    splits = load_splits(json_path)
    train_transform, _ = get_pacs_transforms()
    loaders = []
    
    for domain_id, domain in enumerate(SOURCE_DOMAINS):
        ds = PACSDataset(splits["sources"][domain]["train"], train_transform, domain_id)
        loaders.append(DataLoader(ds, batch_size=8, shuffle=True, drop_last=True))
        
    return MultiSourceBatchIterator(loaders)

def get_target_train_loader(json_path: str = "shared/splits/pacs_sketch_seed6304.json"):
    splits = load_splits(json_path)
    train_transform, _ = get_pacs_transforms()
    ds = PACSDataset(splits["target"][TARGET_DOMAIN], train_transform, domain_id=3)
    return DataLoader(ds, batch_size=24, shuffle=True, drop_last=True)

def get_source_val_loaders(json_path: str = "shared/splits/pacs_sketch_seed6304.json"):
    splits = load_splits(json_path)
    _, eval_transform = get_pacs_transforms()
    loaders = {}
    
    for domain_id, domain in enumerate(SOURCE_DOMAINS):
        ds = PACSDataset(splits["sources"][domain]["val"], eval_transform, domain_id)
        loaders[domain] = DataLoader(ds, batch_size=32, shuffle=False)
        
    return loaders

def get_target_eval_loader(json_path: str = "shared/splits/pacs_sketch_seed6304.json"):
    splits = load_splits(json_path)
    _, eval_transform = get_pacs_transforms()
    ds = PACSDataset(splits["target"][TARGET_DOMAIN], eval_transform, domain_id=3)
    return DataLoader(ds, batch_size=32, shuffle=False)
