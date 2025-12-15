"""
Utility Classes and Functions
==============================
Shared components for all training phases:
- Dataset classes with phase-specific augmentation
- Weighted sampling for Phase C
- Training utilities
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm


# ============================================================================
# AUGMENTATION TRANSFORMS
# ============================================================================

def get_phase_a_transforms(augment: bool = True):
    """Phase A: Binary classifier augmentation (224×224)"""
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


# ============================================================================
# DATASET CLASSES
# ============================================================================

class PhaseABinaryDataset(Dataset):
    """Phase A: Binary Food vs Fruit Classifier"""
    def __init__(self, split: str = 'train', augment: bool = True):
        """
        Args:
            split: 'train' or 'val'
            augment: Apply augmentations
        """
        self.augment = augment
        self.transform = get_phase_a_transforms(augment)
        self.image_paths = []
        self.labels = []
        
        base_path = Path("data/processed/phaseA") / split
        
        # Load food images (label=0)
        food_dir = base_path / "food"
        if food_dir.exists():
            for img_path in food_dir.glob("*.jpg"):
                self.image_paths.append(img_path)
                self.labels.append(0)
        
        # Load fruit images (label=1)
        fruit_dir = base_path / "fruit"
        if fruit_dir.exists():
            for img_path in fruit_dir.glob("*.jpg"):
                self.image_paths.append(img_path)
                self.labels.append(1)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


if __name__ == "__main__":
    print("Testing dataset classes...")
    
    # Test Phase A
    print("\n1. Phase A Binary Dataset:")
    ds_a = PhaseABinaryDataset(split='train', augment=True)
    print(f"   - Total samples: {len(ds_a)}")
    print(f"   - Sample batch shape: {ds_a[0][0].shape}")