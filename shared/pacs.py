from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

def get_transforms(is_train=True):
    # Standard ImageNet normalization for ResNet18_Weights.IMAGENET1K_V1
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

class PACSDataset(Dataset):
    def __init__(self, indices, is_train=True):
        """
        Loads the HF dataset and subsets it using the provided indices.
        """
        self.dataset = load_dataset("flwrlabs/pacs", split="train").select(indices)
        self.transform = get_transforms(is_train)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example['image'].convert('RGB')
        label = example['label']
        domain = example['domain']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, domain
