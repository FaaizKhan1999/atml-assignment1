%%writefile shared/pacs_protocol.py
import json
from torch.utils.data import DataLoader
from shared.pacs import PACSDataset

def cycle_loader(iterable):
    while True:
        for x in iterable:
            yield x

class UDADataIter:
    def __init__(self, source_loaders, target_loader):
        self.source_iters = [cycle_loader(loader) for loader in source_loaders]
        self.target_iter = cycle_loader(target_loader)
        
    def __next__(self):
        source_batches = [next(i) for i in self.source_iters]
        target_batch = next(self.target_iter)
        return source_batches, target_batch

def get_uda_loaders(split_path="shared/splits/pacs_sketch_seed6304.json", num_workers=2):
    with open(split_path, "r") as f:
        splits = json.load(f)
        
    source_domains = ['photo', 'art_painting', 'cartoon']
    
    # Train Loaders
    source_train_loaders = [
        DataLoader(PACSDataset(splits[dom]['train'], is_train=True), 
                   batch_size=8, shuffle=True, num_workers=num_workers, drop_last=True)
        for dom in source_domains
    ]
    target_train_loader = DataLoader(
        PACSDataset(splits['sketch']['target'], is_train=True), 
        batch_size=24, shuffle=True, num_workers=num_workers, drop_last=True
    )
    
    # Validation Loaders
    source_val_loaders = {
        dom: DataLoader(PACSDataset(splits[dom]['val'], is_train=False), 
                        batch_size=32, shuffle=False, num_workers=num_workers)
        for dom in source_domains
    }
    
    uda_train_iter = UDADataIter(source_train_loaders, target_train_loader)
    
    # Calculate exact steps required to complete one full pass over the largest domain
    lengths = [len(loader) for loader in source_train_loaders] + [len(target_train_loader)]
    steps_per_epoch = max(lengths)
    
    return uda_train_iter, source_val_loaders, steps_per_epoch
