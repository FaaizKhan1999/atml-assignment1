import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def compute_detailed_metrics(true_labels, predictions, num_classes=7):
    """
    Calculates overall accuracy, macro-F1, and per-class accuracy.
    """
    acc = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    
    cm = confusion_matrix(true_labels, predictions, labels=np.arange(num_classes))
    # Avoid division by zero for classes that might not appear
    class_totals = cm.sum(axis=1)
    per_class_acc = np.divide(cm.diagonal(), class_totals, out=np.zeros_like(class_totals, dtype=float), where=class_totals!=0)
    
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm
    }
