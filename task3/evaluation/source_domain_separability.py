import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def compute_domain_separability(model, val_loaders, device, seed=6304):
    """
    Extracts frozen features, balances them, and trains a logistic regression 
    model to predict the source domain (chance = 33.3%).
    """
    model.eval()
    
    domain_to_label = {'photo': 0, 'art_painting': 1, 'cartoon': 2}
    features_dict = {0: [], 1: [], 2: []}
    
    with torch.no_grad():
        for domain, loader in val_loaders.items():
            dom_label = domain_to_label[domain]
            for images, _, _ in loader:
                images = images.to(device)
                # Extract 512-dim features directly from backbone
                feats = model.backbone(images)
                features_dict[dom_label].append(feats.cpu().numpy())
                
    # Flatten and balance the features across the three domains
    for dom_label in features_dict:
        features_dict[dom_label] = np.concatenate(features_dict[dom_label], axis=0)
        
    min_samples = min(len(feats) for feats in features_dict.values())
    
    # Subsample to strictly balance the domains
    rng = np.random.default_rng(seed)
    X, y = [], []
    for dom_label, feats in features_dict.items():
        indices = rng.choice(len(feats), min_samples, replace=False)
        X.append(feats[indices])
        y.extend([dom_label] * min_samples)
        
    X = np.concatenate(X, axis=0)
    y = np.array(y)
    
    # Create 70/30 split using the required seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    
    # Train multinomial logistic regression with C=1
    clf = LogisticRegression(C=1.0, multi_class='multinomial', max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    
    test_preds = clf.predict(X_test)
    separability_score = accuracy_score(y_test, test_preds)
    
    return separability_score
