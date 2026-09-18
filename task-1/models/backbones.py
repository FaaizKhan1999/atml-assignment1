import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
import open_clip

# STL-10 class labels in their official order
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer", 
    "dog", "horse", "monkey", "ship", "truck"
]

class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Use torchvision ResNet-50 with ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Replace the final fully connected layer with Identity to extract the global-average-pooled ResNet feature
        self.model.fc = nn.Identity()
        self.feature_dim = 2048
        
        # Freeze every backbone parameter
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.model(x)

class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Use torchvision ViT-B/16 with ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Replace the classifier head with Identity to extract the final ViT class token
        self.model.heads = nn.Identity()
        self.feature_dim = 768
        
        # Freeze every backbone parameter
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.model(x)

class CLIPBackbone(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        # Use OpenCLIP ViT-B-32 with pretrained='openai'
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.model = model.to(device)
        self.feature_dim = 512
        self.device = device
        
        # Freeze every backbone parameter
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Precompute zero-shot text embeddings using the fixed prompt
        prompts = [f"a photo of a {c}." for c in STL10_CLASSES]
        text_tokens = open_clip.tokenize(prompts).to(device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
    def forward(self, x):
        # Extract the normalized CLIP image embedding
        image_features = self.model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
        
    def zero_shot_predict(self, x):
        """Computes confidence from the softmax over scaled class similarities[cite: 1]."""
        image_features = self.forward(x)
        logit_scale = self.model.logit_scale.exp()
        # Scaled class similarities
        logits = logit_scale * image_features @ self.text_features.T
        confidences = logits.softmax(dim=-1)
        return confidences

class LinearClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.classifier(x)

def get_backbone(model_name: str, device="cpu"):
    """Returns the requested frozen backbone mapped to the correct device."""
    if model_name == "resnet":
        return ResNet50Backbone().to(device)
    elif model_name == "vit":
        return ViTBackbone().to(device)
    elif model_name == "clip":
        return CLIPBackbone(device=device)
    else:
        raise ValueError(f"Unknown backbone: {model_name}")
