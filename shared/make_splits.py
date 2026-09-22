import json
import random
from collections import defaultdict
from datasets import load_dataset

def create_pacs_splits(seed=6304, save_path="shared/splits/pacs_sketch_seed6304.json"):
    dataset = load_dataset("flwrlabs/pacs", split="train")
    
    # Organize indices by domain and label for stratification
    domain_label_indices = defaultdict(lambda: defaultdict(list))
    for i, example in enumerate(dataset):
        domain = example['domain']
        label = example['label']
        domain_label_indices[domain][label].append(i)
        
    splits = {}
    random.seed(seed)
    
    source_domains = ['photo', 'art_painting', 'cartoon']
    
    for domain in source_domains:
        splits[domain] = {'train': [], 'val': []}
        for label, indices in domain_label_indices[domain].items():
            random.shuffle(indices)
            split_idx = int(len(indices) * 0.8)
            splits[domain]['train'].extend(indices[:split_idx])
            splits[domain]['val'].extend(indices[split_idx:])
            
    # Sketch is completely unlabeled target data during adaptation
    splits['sketch'] = {'target': []}
    for label, indices in domain_label_indices['sketch'].items():
        splits['sketch']['target'].extend(indices)
        
    with open(save_path, "w") as f:
        json.dump(splits, f, indent=4)
    print(f"Splits successfully saved to {save_path}")

if __name__ == "__main__":
    create_pacs_splits()
