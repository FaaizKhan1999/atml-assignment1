import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_pacs_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    return train_transform, eval_transform

class PACSDataset(Dataset):
    def __init__(self, samples: list[dict], transform=None, domain_id: int = 0):
        self.samples = samples
        self.transform = transform
        self.domain_id = domain_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        entry = self.samples[idx]
        with open(entry["path"], "rb") as f:
            image = Image.open(f).convert("RGB")
            
        if self.transform is not None:
            image = self.transform(image)
            
        return {
            "image": image,
            "label": torch.tensor(entry["label"], dtype=torch.long),
            "domain": torch.tensor(self.domain_id, dtype=torch.long),
        }
