import os
import json
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(TASK1_DIR, "results")

def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def main():
    models = ["resnet", "vit", "clip", "clip_zeroshot"]
    labels = ["ResNet-50", "ViT-B/16", "CLIP (Linear)", "CLIP (Zero-Shot)"]
    x = np.arange(len(models))
    width = 0.2

    clean_data = load_json("clean_baseline.json")
    color_data = load_json("color_bias_results.json")
    patch_data = load_json("patch_structure_results.json")

    clean_accs = [clean_data.get(m, {}).get("acc", 0) for m in models]
    gray_accs = [color_data.get(m, {}).get("grayscale", {}).get("acc", 0) for m in models]
    hue_accs = [color_data.get(m, {}).get("hue_rotation", {}).get("acc", 0) for m in models]
    patch_accs = [patch_data.get(m, {}).get("patch_acc", 0) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, clean_accs, width, label='Clean', color='#1f77b4')
    ax.bar(x - 0.5*width, gray_accs, width, label='Grayscale', color='#7f7f7f')
    ax.bar(x + 0.5*width, hue_accs, width, label='Hue Rotation', color='#ff7f0e')
    ax.bar(x + 1.5*width, patch_accs, width, label='Patch Shuffle', color='#d62728')

    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title('Performance Comparison Across Interventions')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, "performance_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    shape_data = load_json("shape_bias_results.json")
    coverage = [shape_data.get(m, {}).get("coverage", 0) for m in models]
    shape_bias = [shape_data.get(m, {}).get("shape_bias", 0) for m in models]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, coverage, width, label='Coverage', color='#2ca02c')
    ax.bar(x + width/2, shape_bias, width, label='Shape Bias', color='#9467bd')

    ax.set_ylabel('Metric Score')
    ax.set_title('Shape Bias and Cue Coverage')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, "shape_bias_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()

    rep_data = load_json("representation_stability.json")
    models_rep = ["resnet", "vit", "clip"]
    labels_rep = ["ResNet-50", "ViT-B/16", "CLIP"]
    x_rep = np.arange(len(models_rep))
    
    gray_stabs = [rep_data.get(m, {}).get("cosine_stability", {}).get("grayscale", 0) for m in models_rep]
    trans_stabs = [rep_data.get(m, {}).get("cosine_stability", {}).get("translation_d16", 0) for m in models_rep]
    patch_stabs = [rep_data.get(m, {}).get("cosine_stability", {}).get("patch_shuffle", 0) for m in models_rep]
    cue_stabs = [rep_data.get(m, {}).get("cosine_stability", {}).get("cue_conflict", 0) for m in models_rep]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_rep - 1.5*width, gray_stabs, width, label='Grayscale', color='#7f7f7f')
    ax.bar(x_rep - 0.5*width, trans_stabs, width, label='Translation (\u03B4=16)', color='#17becf')
    ax.bar(x_rep + 0.5*width, patch_stabs, width, label='Patch Shuffle', color='#d62728')
    ax.bar(x_rep + 1.5*width, cue_stabs, width, label='Cue Conflict', color='#e377c2')

    ax.set_ylabel('Cosine Stability (I_T)')
    ax.set_title('Representation Stability Across Interventions')
    ax.set_xticks(x_rep)
    ax.set_xticklabels(labels_rep)
    ax.legend(loc='lower left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, "representation_stability_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Successfully generated summary plots in task1/results/")

if __name__ == "__main__":
    main()
