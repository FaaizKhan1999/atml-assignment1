import torchvision.transforms as T

base_transform = T.Compose([
    T.Resize((224, 224))
])

imagenet_norm = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

clip_norm = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711])
])

def get_normalization(model_name: str):
    """Returns the appropriate tensor conversion and normalization sequence."""
    if model_name in ["resnet", "vit"]:
        return imagenet_norm
    elif model_name == "clip":
        return clip_norm
    else:
        raise ValueError(f"Unknown model normalization requested: {model_name}")
