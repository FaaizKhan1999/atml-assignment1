import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from task4.scores.msp import compute_msp
from task4.scores.mls import compute_mls
from task4.scores.energy import compute_energy
from task4.scores.mahalanobis import MahalanobisScorer
from task4.evaluation.metrics import get_threshold, compute_auroc, compute_rates
from task4.models.resnet_cifar import CIFAR_ResNet18
from task4.data.cifar10 import get_cifar10_loaders
from task4.data.cifar100_unknowns import get_cifar100_unknown_loaders

from task4.methods.rpl import ReciprocalPoints

def compute_proser_score(logits, num_known=10):
    """
    PROSER placeholder-based detection score.
    Score = max(dummy_logits) - max(known_logits)
    Larger value = stronger dummy response = more unknown.
    """
    known_logits = logits[:, :num_known]
    dummy_logits = logits[:, num_known:]
    max_known, _ = torch.max(known_logits, dim=1)
    max_dummy, _ = torch.max(dummy_logits, dim=1)
    return (max_dummy - max_known).cpu()

def load_cache(name):
    return torch.load(f"./task4/cache/{name}.pt", map_location="cpu", weights_only=True)

def quick_inference(model, loader, device="cpu", rp_module=None):
    """
    Runs a fast local CPU forward pass.
    If rp_module is provided, it extracts features and computes distances for RPL.
    """
    model.eval()
    if rp_module:
        rp_module.eval()
        
    all_logits, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            if rp_module:
                _, features = model(inputs, return_features=True)
                logits = rp_module(features)
            else:
                logits = model(inputs)
            all_logits.append(logits.cpu())
            all_targets.append(targets)
            
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)

def main():
    print("--- Loading Vanilla Caches ---")
    val = load_cache("cifar10_val")
    test = load_cache("cifar10_test")
    near = load_cache("cifar100_near")
    far = load_cache("cifar100_far")
    train = load_cache("cifar10_train")

    maha = MahalanobisScorer()
    maha.fit(train["features"], train["targets"])

    scores_dict = {
        "MSP": compute_msp,
        "MLS": compute_mls,
        "Energy": compute_energy
    }

    # Initialize JSON dictionary
    json_output = {
        "table_1_vanilla_osr_scores": {},
        "table_2_model_interventions": {}
    }

    # =================================================================================
    # TABLE 1: Vanilla Model OSR Scores Comparison
    # =================================================================================
    print("\n" + "="*80)
    print("TABLE 1: Vanilla Model OSR Scores Comparison")
    print(f"{'Score':<15} | {'Near AUROC':<12} | {'Far AUROC':<12} | {'All AUROC':<12} | {'FPR@95TPR (Near)':<16} | {'FPR@95TPR (Far)':<16}")
    print("-" * 80)

    all_unknowns = torch.cat([near["logits"], far["logits"]], dim=0)
    all_unknown_feats = torch.cat([near["features"], far["features"]], dim=0)
    plot_data = {}

    for name, func in scores_dict.items():
        val_s = func(val["logits"]).numpy()
        test_s = func(test["logits"]).numpy()
        near_s = func(near["logits"]).numpy()
        far_s = func(far["logits"]).numpy()
        all_s = func(all_unknowns).numpy()
        
        tau = get_threshold(val_s, 95)
        n_auc = compute_auroc(test_s, near_s)
        f_auc = compute_auroc(test_s, far_s)
        a_auc = compute_auroc(test_s, all_s)
        _, n_fpr = compute_rates(test_s, near_s, tau)
        _, f_fpr = compute_rates(test_s, far_s, tau)
        
        plot_data[name] = {"test": test_s, "near": near_s, "far": far_s}
        print(f"{name:<15} | {n_auc:.4f}       | {f_auc:.4f}       | {a_auc:.4f}       | {n_fpr:.4f}           | {f_fpr:.4f}")
        
        json_output["table_1_vanilla_osr_scores"][name] = {
            "near_auroc": round(n_auc, 4), "far_auroc": round(f_auc, 4), "all_auroc": round(a_auc, 4),
            "fpr_95tpr_near": round(n_fpr, 4), "fpr_95tpr_far": round(f_fpr, 4)
        }

    # Mahalanobis
    m_val = maha.score(val["features"]).numpy()
    m_test = maha.score(test["features"]).numpy()
    m_near = maha.score(near["features"]).numpy()
    m_far = maha.score(far["features"]).numpy()
    m_all = maha.score(all_unknown_feats).numpy()
    
    m_tau = get_threshold(m_val, 95)
    m_n_auc, m_f_auc, m_a_auc = compute_auroc(m_test, m_near), compute_auroc(m_test, m_far), compute_auroc(m_test, m_all)
    _, m_n_fpr = compute_rates(m_test, m_near, m_tau)
    _, m_f_fpr = compute_rates(m_test, m_far, m_tau)
    
    print(f"{'Mahalanobis':<15} | {m_n_auc:.4f}       | {m_f_auc:.4f}       | {m_a_auc:.4f}       | {m_n_fpr:.4f}           | {m_f_fpr:.4f}")
    plot_data["Mahalanobis"] = {"test": m_test, "near": m_near, "far": m_far}
    
    json_output["table_1_vanilla_osr_scores"]["Mahalanobis"] = {
        "near_auroc": round(m_n_auc, 4), "far_auroc": round(m_f_auc, 4), "all_auroc": round(m_a_auc, 4),
        "fpr_95tpr_near": round(m_n_fpr, 4), "fpr_95tpr_far": round(m_f_fpr, 4)
    }

    # =================================================================================
    # TABLE 2: Model Interventions (MLS Score) + PROSER Custom + RPL
    # =================================================================================
    print("\n" + "="*80)
    print("TABLE 2: Model Interventions (MLS Score)")
    print(f"{'Model':<15} | {'CSA (%)':<10} | {'Near AUROC':<12} | {'Far AUROC':<12} | {'FPR@95TPR (Near)':<16} | {'FPR@95TPR (Far)':<16}")
    print("-" * 80)
    
    _, val_loader, test_loader = get_cifar10_loaders("./task4/configs/base.yaml")
    near_loader, far_loader = get_cifar100_unknown_loaders("./task4/configs/base.yaml")

    # Added RPL to the loop
    for model_name, num_classes in [("vanilla", 10), ("gcsc", 10), ("proser", 15), ("rpl", 10)]:
        ckpt_path = f"./task4/checkpoints/{model_name}_best.pth"
        if not os.path.exists(ckpt_path):
            continue
            
        model = CIFAR_ResNet18(num_classes=num_classes)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        
        rp_module = None
        if "rp_module" in ckpt:
            # RPL checkpoint structure
            model.load_state_dict(ckpt['model'])
            rp_module = ReciprocalPoints(num_classes=10, embed_dim=512)
            rp_module.load_state_dict(ckpt['rp_module'])
        else:
            # Standard checkpoint structure
            model.load_state_dict(ckpt)
        
        # CPU Inference
        val_l, _ = quick_inference(model, val_loader, rp_module=rp_module)
        test_l, test_y = quick_inference(model, test_loader, rp_module=rp_module)
        near_l, _ = quick_inference(model, near_loader, rp_module=rp_module)
        far_l, _ = quick_inference(model, far_loader, rp_module=rp_module)
        
        # 1. Standard MLS Evaluation
        csa = (test_l[:, :10].max(1)[1] == test_y).float().mean().item() * 100
        
        v_s = compute_mls(val_l[:, :10]).numpy()
        t_s = compute_mls(test_l[:, :10]).numpy()
        n_s = compute_mls(near_l[:, :10]).numpy()
        f_s = compute_mls(far_l[:, :10]).numpy()
        
        tau = get_threshold(v_s, 95)
        n_auc, f_auc = compute_auroc(t_s, n_s), compute_auroc(t_s, f_s)
        _, n_fpr = compute_rates(t_s, n_s, tau)
        _, f_fpr = compute_rates(t_s, f_s, tau)
        
        print(f"{model_name.upper():<15} | {csa:<10.2f} | {n_auc:.4f}       | {f_auc:.4f}       | {n_fpr:.4f}           | {f_fpr:.4f}")
        json_output["table_2_model_interventions"][model_name.upper()] = {
            "csa_percent": round(csa, 2), "near_auroc": round(n_auc, 4), "far_auroc": round(f_auc, 4),
            "fpr_95tpr_near": round(n_fpr, 4), "fpr_95tpr_far": round(f_fpr, 4)
        }

        # 2. PROSER Placeholder Detection Score (Row 4)
        if model_name == "proser":
            v_p = compute_proser_score(val_l).numpy()
            t_p = compute_proser_score(test_l).numpy()
            n_p = compute_proser_score(near_l).numpy()
            f_p = compute_proser_score(far_l).numpy()
            
            tau_p = get_threshold(v_p, 95)
            pn_auc, pf_auc = compute_auroc(t_p, n_p), compute_auroc(t_p, f_p)
            _, pn_fpr = compute_rates(t_p, n_p, tau_p)
            _, pf_fpr = compute_rates(t_p, f_p, tau_p)
            
            print(f"{'PROSER-CUSTOM':<15} | {'-':<10} | {pn_auc:.4f}       | {pf_auc:.4f}       | {pn_fpr:.4f}           | {pf_fpr:.4f}")
            json_output["table_2_model_interventions"]["PROSER-CUSTOM"] = {
                "csa_percent": None, "near_auroc": round(pn_auc, 4), "far_auroc": round(pf_auc, 4),
                "fpr_95tpr_near": round(pn_fpr, 4), "fpr_95tpr_far": round(pf_fpr, 4)
            }

    # =================================================================================
    # SAVE ARTIFACTS
    # =================================================================================
    os.makedirs("./results", exist_ok=True)
    
    # Save JSON
    with open("./results/osr_results.json", "w") as f:
        json.dump(json_output, f, indent=4)
    print("\nSaved tabular findings to results/osr_results.json")

    # Generate Multi-Panel Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_scores = ["MSP", "MLS", "Mahalanobis"]
    for i, s in enumerate(plot_scores):
        axes[i].hist(plot_data[s]["test"], bins=50, alpha=0.5, density=True, label="Known (CIFAR-10)")
        axes[i].hist(plot_data[s]["near"], bins=50, alpha=0.5, density=True, label="Near Unknown")
        axes[i].hist(plot_data[s]["far"], bins=50, alpha=0.5, density=True, label="Far Unknown")
        axes[i].set_title(f"{s} Score Distribution")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig("./results/task4_score_distributions.png")
    print("Saved score distributions to results/task4_score_distributions.png")

if __name__ == "__main__":
    main()
