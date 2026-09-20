import os
from datasets import load_dataset

# Load the dataset
ds = load_dataset("flwrlabs/pacs", split="train")

# Create the base directory
base_dir = "data"
os.makedirs(base_dir, exist_ok=True)

# The Hugging Face dataset has 'domain' and 'label' features. 
# We need to map the integer labels back to their class names.
labels = ds.features["label"].names

for item in ds:
    domain = item["domain"]
    class_name = labels[item["label"]]
    image = item["image"]
    
    # Create domain/class folders (e.g., pacs_dataset/art_painting/dog)
    save_path = os.path.join(base_dir, domain, class_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Save the image
    # Generate a unique filename using a hash or index to prevent overwriting
    filename = f"{hash(image.tobytes())}.jpg" 
    image.save(os.path.join(save_path, filename))

print("Done extracting raw images!")
