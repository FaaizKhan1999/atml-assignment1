import os
import json
import urllib.request
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import datasets
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(TASK1_DIR, "data")
CANDIDATES_DIR = os.path.join(DATA_DIR, "cue_conflicts_candidates")

# --- AdaIN Implementation (Mathematical Port of xunhuang1995/AdaIN-style) ---

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3))
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(), 
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(), 
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(), 
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(), 
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(), 
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(), 
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(), 
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(), 
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU()   
)

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4]) 
        self.enc_2 = nn.Sequential(*enc_layers[4:11]) 
        self.enc_3 = nn.Sequential(*enc_layers[11:18]) 
        self.enc_4 = nn.Sequential(*enc_layers[18:31]) 
        self.decoder = decoder
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
                
        for param in self.decoder.parameters():
            param.requires_grad = False

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def forward(self, content, style, alpha=1.0):
        content_feat = self.encode(content)
        style_feat = self.encode(style)
        
        t = adaptive_instance_normalization(content_feat, style_feat)
        t = alpha * t + (1 - alpha) * content_feat

        return self.decoder(t)

def download_weights(data_dir):
    enc_url = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"
    dec_url = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"
    enc_path = os.path.join(data_dir, "vgg_normalised.pth")
    dec_path = os.path.join(data_dir, "decoder.pth")
    
    if not os.path.exists(enc_path):
        print("Downloading AdaIN VGG encoder...")
        urllib.request.urlretrieve(enc_url, enc_path)
    if not os.path.exists(dec_path):
        print("Downloading AdaIN decoder...")
        urllib.request.urlretrieve(dec_url, dec_path)
        
    return enc_path, dec_path

def main():
    os.makedirs(CANDIDATES_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Cue Conflicts on device: {device}")

    enc_path, dec_path = download_weights(DATA_DIR)
    
    vgg.load_state_dict(torch.load(enc_path), strict=False)
    decoder.load_state_dict(torch.load(dec_path))
    
    model = Net(vgg, decoder).to(device)
    model.eval()

    splits_path = os.path.join(DATA_DIR, "splits_seed6304.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
        
    test_dataset = datasets.STL10(root=DATA_DIR, split="test", download=False)
    
    class_to_indices = {i: [] for i in range(10)}
    for idx in splits["test_subset_indices"]:
        _, label = test_dataset[idx]
        class_to_indices[label].append(idx)

    # Use BICUBIC interpolation to reduce upscaling blur artifacts from the 96x96 images
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    
    images_per_pair = 5
    metadata = {}
    
    rng = np.random.RandomState(6304)

    for content_cls in range(10):
        for style_cls in range(10):
            if content_cls == style_cls:
                continue
            
            content_candidates = rng.choice(class_to_indices[content_cls], images_per_pair, replace=False)
            style_candidates = rng.choice(class_to_indices[style_cls], images_per_pair, replace=False)
            
            for i in range(images_per_pair):
                c_idx, s_idx = content_candidates[i], style_candidates[i]
                c_img, _ = test_dataset[c_idx]
                s_img, _ = test_dataset[s_idx]
                
                c_tensor = transform(c_img).unsqueeze(0).to(device)
                s_tensor = transform(s_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # Reduced alpha to 0.45 to heavily prioritize structure retention
                    output = model(c_tensor, s_tensor, alpha=0.7)
                    
                output = output.cpu().squeeze(0).clamp(0, 1)
                output_pil = T.ToPILImage()(output)
                
                filename = f"shape{content_cls}_texture{style_cls}_{c_idx}_{s_idx}.png"
                filepath = os.path.join(CANDIDATES_DIR, filename)
                output_pil.save(filepath)
                
                metadata[filename] = {
                    "shape_class": int(content_cls),
                    "texture_class": int(style_cls)
                }
                
            print(f"Generated {images_per_pair} candidates for Shape: {content_cls} | Texture: {style_cls}")

    meta_path = os.path.join(DATA_DIR, "cue_conflicts_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nSuccessfully generated 450 candidates in {CANDIDATES_DIR}.")

if __name__ == "__main__":
    main()
