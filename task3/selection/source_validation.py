import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def evaluate_sources(model, val_loaders, device):
    """
    Evaluates the model across the three source validation domains.
    Returns per-domain metrics and the mean macro-F1 used for checkpoint selection.
    """
    model.eval()
    domain_metrics = {}
    macro_f1s = []
    
    with torch.no_grad():
        for domain, loader in val_loaders.items():
            all_preds = []
            all_labels = []
            for images, labels, _ in loader:
                images, labels = images.to(device), labels.to(device)
                _, logits = model(images)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            acc = accuracy_score(all_labels, all_preds)
            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            
            domain_metrics[domain] = {'accuracy': acc, 'macro_f1': macro_f1}
            macro_f1s.append(macro_f1)
            
    mean_macro_f1 = np.mean(macro_f1s)
    return domain_metrics, mean_macro_f1
