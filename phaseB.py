# phaseB_full.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import random

# ==========================
# Dataset Phase B (Flat folder)
# ==========================
class PhaseBFoodDatasetFlat(Dataset):
    """Siamese Dataset for Phase B (Flat folder, no class subfolders)"""
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform
        self.data = list(self.folder_path.glob("*.jpg"))
        if len(self.data) == 0:
            raise ValueError(f"No images found in {folder_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path1 = self.data[idx]

        # Positive or Negative pair randomly (dummy)
        if random.random() < 0.5:
            img_path2 = random.choice(self.data)  # positive
            label = 1
        else:
            img_path2 = random.choice(self.data)  # negative
            label = 0

        img1 = Image.open(img_path1).convert("RGB")
        img2 = Image.open(img_path2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), torch.tensor(label, dtype=torch.float32)

# ==========================
# Transformations
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),         
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# ==========================
# Siamese Network
# ==========================
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, embedding_dim)
        self.backbone = backbone

    def forward(self, x1, x2):
        out1 = self.backbone(x1)
        out2 = self.backbone(x2)
        return out1, out2

# ==========================
# Contrastive Loss
# ==========================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        # Euclidean distance
        dist = torch.nn.functional.pairwise_distance(out1, out2)
        loss_pos = label * dist.pow(2)
        loss_neg = (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        loss = torch.mean(loss_pos + loss_neg)
        return loss

# ==========================
# Main Training Script
# ==========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to processed images
    train_path = "data/processed/phaseB/train/food"
    val_path = "data/processed/phaseB/val/food"

    # Datasets
    train_dataset = PhaseBFoodDatasetFlat(train_path, transform=transform)
    val_dataset = PhaseBFoodDatasetFlat(val_path, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model, Loss, Optimizer
    model = SiameseNetwork(embedding_dim=128).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        # ======== Training ========
        model.train()
        train_loss = 0
        for (img1, img2), labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ======== Validation ========
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (img1, img2), labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
#----model saving----#
torch.save(model.state_dict(), "phaseB_model.pt")
print("✓ Model saved as phaseB_model.pt")
