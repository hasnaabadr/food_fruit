import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import random
import torch.nn.functional as F

# ==========================
# Dataset Phase B (Folder per class)
# ==========================
class PhaseBFoodDatasetByClass(Dataset):
    """Siamese Dataset for Phase B (Folder per class)"""
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform
        self.data = []

        for class_dir in self.folder_path.iterdir():
            if class_dir.is_dir():
                imgs = list(class_dir.glob("*.jpg"))
                for img in imgs:
                    self.data.append((img, class_dir.name))

        if len(self.data) == 0:
            raise ValueError(f"No images found in {folder_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path1, class1 = self.data[idx]

        # Decide positive / negative
        same_class_imgs = [p for p, c in self.data if c == class1 and p != img_path1]
        diff_class_imgs = [p for p, c in self.data if c != class1]

        if random.random() < 0.5 and len(same_class_imgs) > 0:
            img_path2 = random.choice(same_class_imgs)
            label = 1
        else:
            img_path2 = random.choice(diff_class_imgs)
            label = 0

        img1 = Image.open(img_path1).convert("RGB")
        img2 = Image.open(img_path2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), torch.tensor(label, dtype=torch.float32)

# ==========================
# Transforms
# ==========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, embedding_dim)
        self.backbone = backbone

    def forward_once(self, x):
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# ==========================
# Contrastive Loss
# ==========================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = F.pairwise_distance(out1, out2)
        loss_pos = label * dist.pow(2)
        loss_neg = (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return torch.mean(loss_pos + loss_neg)

# ==========================
# Accuracy (for monitoring only)
# ==========================
def compute_accuracy(out1, out2, label):
    dist = F.pairwise_distance(out1, out2)
    threshold = dist.mean()
    pred = (dist < threshold).float()
    correct = (pred == label).sum().item()
    return correct / len(label)

# ==========================
# Training Script
# ==========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = "data/processed/phaseB/train/food"
    val_path   = "data/processed/phaseB/val/food"

    train_dataset = PhaseBFoodDatasetByClass(train_path, transform=train_transform)
    val_dataset   = PhaseBFoodDatasetByClass(val_path,   transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = SiameseNetwork(embedding_dim=128).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        # Training
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

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for (img1, img2), labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, labels)
                val_loss += loss.item()
                val_acc += compute_accuracy(out1, out2, labels) * len(labels)

        val_loss /= len(val_loader)
        val_acc /= len(val_dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"- Train Loss: {train_loss:.4f} "
              f"- Val Loss: {val_loss:.4f} "
              f"- Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "phaseB_model.pt")
    print("✓ Model saved as phaseB_model.pt")
