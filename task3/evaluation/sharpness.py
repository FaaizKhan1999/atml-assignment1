import torch
import torch.nn.functional as F

def compute_sharpness_proxy(model, val_loaders, device, rho=0.05, seed=6304):
    """
    Measures the increase in loss after a normalized gradient-ascent perturbation.
    Uses exactly 32 samples per source domain.
    """
    model.eval()
    
    # Collect exactly 32 samples per domain using the fixed seed
    rng = torch.Generator().manual_seed(seed)
    images_list, labels_list = [], []
    
    for domain, loader in val_loaders.items():
        dom_images, dom_labels = [], []
        for imgs, lbls, _ in loader:
            dom_images.append(imgs)
            dom_labels.append(lbls)
            
        dom_images = torch.cat(dom_images, dim=0)
        dom_labels = torch.cat(dom_labels, dim=0)
        
        # Randomly select 32 samples
        indices = torch.randperm(len(dom_images), generator=rng)[:32]
        images_list.append(dom_images[indices])
        labels_list.append(dom_labels[indices])
        
    X = torch.cat(images_list, dim=0).to(device)
    Y = torch.cat(labels_list, dim=0).to(device)
    
    model.zero_grad()
    
    # 1. Compute Base Loss
    features, logits = model(X)
    loss_base = F.cross_entropy(logits, Y)
    loss_base.backward()
    
    # 2. Compute Global Gradient Norm
    grad_norm = torch.norm(
        torch.stack([
            p.grad.norm(p=2).to(device) 
            for p in model.parameters() if p.grad is not None
        ]), 
        p=2
    )
    
    # 3. Apply normalized perturbation: epsilon = rho * (grad / ||grad||_2)
    perturbations = {}
    with torch.no_grad():
        scale = rho / (grad_norm + 1e-12)
        for name, p in model.named_parameters():
            if p.grad is not None:
                e_w = p.grad * scale
                p.add_(e_w)
                perturbations[name] = e_w
                
    # 4. Compute Perturbed Loss
    model.zero_grad()
    with torch.no_grad():
        _, logits_adv = model(X)
        loss_adv = F.cross_entropy(logits_adv, Y)
        
    # 5. Restore original parameters
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in perturbations:
                p.sub_(perturbations[name])
                
    delta_sharp = (loss_adv - loss_base).item()
    return delta_sharp, loss_base.item(), loss_adv.item()
