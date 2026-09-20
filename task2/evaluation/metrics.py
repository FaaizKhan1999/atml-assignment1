import torch
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def compute_metrics(true_labels: list, pred_labels: list) -> dict:
    """Computes accuracy and macro-F1 for a single domain."""
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}

def evaluate_source_domains(model, classifier, val_loaders, device) -> dict:
    """
    Evaluates the model across the source domains and computes the mean macro-F1.
    """
    model.eval()
    classifier.eval()
    
    results = {}
    macro_f1s = []
    
    with torch.no_grad():
        for domain_name, loader in val_loaders.items():
            all_preds = []
            all_labels = []
            
            for batch in loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                features = model(images)
                logits = classifier(features)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            domain_metrics = compute_metrics(all_labels, all_preds)
            results[domain_name] = domain_metrics
            macro_f1s.append(domain_metrics["macro_f1"])
            
    results["mean_macro_f1"] = np.mean(macro_f1s)
    return results
