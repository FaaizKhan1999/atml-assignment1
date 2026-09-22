%%writefile task2/evaluation/domain_separability.py
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def compute_domain_separability(backbone, source_val_loaders, target_loader, device, seed=6304):
    backbone.eval()
    
    source_features = []
    target_features = []
    
    with torch.no_grad():
        # Collect source validation features
        for loader in source_val_loaders.values():
            for imgs, _, _ in loader:
                imgs = imgs.to(device)
                feats = backbone(imgs).cpu().numpy()
                source_features.append(feats)
                
        # Collect target features
        for imgs, _, _ in target_loader:
            imgs = imgs.to(device)
            feats = backbone(imgs).cpu().numpy()
            target_features.append(feats)
            
    source_features = np.concatenate(source_features, axis=0)
    target_features = np.concatenate(target_features, axis=0)
    
    # Subsample to ensure equal numbers of source and target features
    min_len = min(len(source_features), len(target_features))
    np.random.seed(seed)
    src_idx = np.random.choice(len(source_features), min_len, replace=False)
    tgt_idx = np.random.choice(len(target_features), min_len, replace=False)
    
    X = np.vstack((source_features[src_idx], target_features[tgt_idx]))
    # 1 for Source, 0 for Target
    y = np.hstack((np.ones(min_len), np.zeros(min_len)))
    
    # 70/30 split using seed 6304
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    
    clf = LogisticRegression(C=1.0, random_state=seed, max_iter=1000)
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    return accuracy
