import torchvision.transforms as T
import torchvision.transforms.functional as F

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

grayscale_intervention = T.Grayscale(num_output_channels=3)

hue_rotation_intervention = T.Lambda(lambda img: F.adjust_hue(img, 0.5))

class ReflectionTranslate:
    """
    Applies reflection padding followed by a shifted crop to translate the image.
    Evaluates translation invariance for delta displacements in cardinal directions.
    """
    def __init__(self, delta, direction):
        self.delta = delta
        self.direction = direction.lower()

    def __call__(self, img):
        if self.delta == 0:
            return img
            
        # Pad the image by delta on all 4 sides using reflection padding
        padded = F.pad(img, self.delta, padding_mode='reflect')
        
        # Determine the starting (top, left) coordinates for the 224x224 crop.
        top, left = self.delta, self.delta
        
        # Shifting the content up means sliding the crop box down, and vice versa.
        if self.direction == 'up':
            top = 2 * self.delta
        elif self.direction == 'down':
            top = 0
        elif self.direction == 'left':
            left = 2 * self.delta
        elif self.direction == 'right':
            left = 0
            
        return F.crop(padded, top, left, 224, 224)

def patch_shuffle_intervention(img, permutation):
    """
    Divides a 224x224 PIL image into a 4x4 grid and shuffles patches.
    """
    grid_size = 4
    patch_size = 224 // grid_size  # 56x56 pixels
    
    # Extract the 16 patches
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            box = (j * patch_size, i * patch_size, (j + 1) * patch_size, (i + 1) * patch_size)
            patches.append(img.crop(box))
            
    # Reassemble them according to the permutation
    shuffled_img = Image.new('RGB', (224, 224))
    for idx, p_idx in enumerate(permutation):
        i = idx // grid_size
        j = idx % grid_size
        box = (j * patch_size, i * patch_size, (j + 1) * patch_size, (i + 1) * patch_size)
        shuffled_img.paste(patches[p_idx], box)
        
    return shuffled_img
