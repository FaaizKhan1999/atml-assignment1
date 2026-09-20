import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path

from shared.pacs_protocol import get_source_train_loader, get_source_val_loaders
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import LinearClassifier
from task2.utils import set_seed, set_batchnorm_eval, EarlyStopping
from task2.evaluation.metrics import evaluate_source_domains

def train_source_only():
    set_seed(6304)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading source datasets...")
    train_loader = get_source_train_loader()
    val_loaders = get_source_val_loaders()
    
    backbone = ResNet18Backbone().to(device)
    classifier = LinearClassifier(in_features=512, num_classes=7).to(device)
    
    optimizer = AdamW(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    
    early_stopping = EarlyStopping(patience=5, mode="max")
    checkpoint_dir = Path("task2/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "source_only_best.pth"
    
    max_epochs = 30
    
    for epoch in range(max_epochs):
        backbone.train()
        classifier.train()
        
        set_batchnorm_eval(backbone)
        
        total_loss = 0.0
        
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            features = backbone(images)
            logits = classifier(features)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        val_results = evaluate_source_domains(backbone, classifier, val_loaders, device)
        mean_f1 = val_results["mean_macro_f1"]
        
        print(f"Epoch {epoch+1:02d}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | Mean Val F1: {mean_f1:.4f}")
        for domain in ["photo", "art_painting", "cartoon"]:
            print(f"  -> {domain} Val F1: {val_results[domain]['macro_f1']:.4f}")
            
        improved = mean_f1 > early_stopping.best_score
        if early_stopping.step(mean_f1, backbone):
            pass # We handle the actual early stopping break below
            
        if improved:
            torch.save({
                'backbone_state_dict': backbone.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'epoch': epoch,
                'mean_macro_f1': mean_f1
            }, checkpoint_path)
            print(f"  [*] New best checkpoint saved to {checkpoint_path}")
            
        if early_stopping.epochs_without_improvement >= early_stopping.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break
            
    print("\nTraining Complete.")
    print(f"Best Mean Macro-F1: {early_stopping.best_score:.4f}")
    print(f"Checkpoint saved at: {checkpoint_path}")

if __name__ == "__main__":
    train_source_only()
