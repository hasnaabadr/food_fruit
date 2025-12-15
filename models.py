import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================================
# PHASE A: BINARY CLASSIFIER (Food vs Fruit)
# ============================================================================

class PhaseABinaryClassifier(nn.Module):
    """ResNet18 for binary classification (Food=0, Fruit=1)"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        return self.resnet18(x)

if __name__ == "__main__":
    print("Testing model architectures...\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Phase A
    print("1. Phase A Binary Classifier:")
    model_a = PhaseABinaryClassifier().to(device)
    x_a = torch.randn(4, 3, 224, 224).to(device)
    out_a = model_a(x_a)
    print(f"   Input: {x_a.shape}, Output: {out_a.shape}")
    print(f"   Trainable params: {sum(p.numel() for p in model_a.parameters() if p.requires_grad)}")
    
    