import argparse
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"

def find_domain_folder(base_dir: Path, domain_name: str) -> Path:
    candidates = [
        base_dir / domain_name,
        base_dir / domain_name.replace("_", " "),
        base_dir / domain_name.title(),
        base_dir / domain_name.replace("_", ""),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"Could not find folder for domain '{domain_name}'")

def scan_domain(domain_path: Path, class_to_idx: dict) -> list[dict]:
    samples = []
    for img_path in domain_path.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            class_name = img_path.parent.name
            if class_name in class_to_idx:
                samples.append({
                    "path": str(img_path.resolve()),
                    "label": class_to_idx[class_name]
                })
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to PACS root")
    parser.add_argument("--out", type=str, default="shared/splits/pacs_sketch_seed6304.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Automatically identify the 7 classes from the first domain
    sample_domain = find_domain_folder(data_dir, SOURCE_DOMAINS[0])
    class_names = sorted([d.name for d in sample_domain.iterdir() if d.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    split_data = {"classes": class_names, "class_to_idx": class_to_idx, "sources": {}, "target": {}}

    # Stratified 80/20 split on seed 6304 for source domains
    for domain in SOURCE_DOMAINS:
        domain_folder = find_domain_folder(data_dir, domain)
        samples = scan_domain(domain_folder, class_to_idx)
        labels = [s["label"] for s in samples]

        train_samples, val_samples = train_test_split(
            samples, test_size=0.20, random_state=6304, stratify=labels
        )
        split_data["sources"][domain] = {"train": train_samples, "val": val_samples}

    # Load Sketch target domain completely unsplit
    target_folder = find_domain_folder(data_dir, TARGET_DOMAIN)
    split_data["target"][TARGET_DOMAIN] = scan_domain(target_folder, class_to_idx)

    with open(out_path, "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"Splits saved to {out_path}")

if __name__ == "__main__":
    main()
