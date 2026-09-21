import torch
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from task4.scores.mls import compute_mls
from task4.evaluation.metrics import get_threshold
from torchvision import datasets

def main():
    print("--- Running Failure Analysis (Vanilla MLS) ---")
    val = torch.load("./task4/cache/cifar10_val.pt", weights_only=True)
    near = torch.load("./task4/cache/cifar100_near.pt", weights_only=True)
    far = torch.load("./task4/cache/cifar100_far.pt", weights_only=True)
    
    val_mls = compute_mls(val["logits"]).numpy()
    near_mls = compute_mls(near["logits"]).numpy()
    far_mls = compute_mls(far["logits"]).numpy()
    
    tau = get_threshold(val_mls, 95)
    print(f"Calibrated 95% Rejection Threshold (Tau): {tau:.4f}\n")
    
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    dataset_100 = datasets.CIFAR100(root="./data_cache", train=False, download=False)
    
    def inspect_failures(unknown_data, unknown_mls, group_name):
        # Find indices where unknownness score is <= tau (incorrectly accepted)
        accepted_idx = (unknown_mls <= tau).nonzero()[0]
        print(f"Top 5 {group_name} Unknowns incorrectly accepted as known:")
        
        count = 0
        for idx in accepted_idx:
            if count >= 5:
                break
                
            true_label_idx = unknown_data["targets"][idx].item()
            true_class = dataset_100.classes[true_label_idx]
            
            pred_class_idx = unknown_data["logits"][idx].argmax().item()
            pred_class = cifar10_classes[pred_class_idx]
            
            score = unknown_mls[idx]
            print(f"  - True Class: {true_class:<15} | Predicted As: {pred_class:<10} | MLS Score: {score:.4f}")
            count += 1
        print("-" * 60)

    inspect_failures(near, near_mls, "Near")
    inspect_failures(far, far_mls, "Far")

if __name__ == "__main__":
    main()
