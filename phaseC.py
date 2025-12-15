# phaseC.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# ==========================
# Dataset Phase C - Fruit Classification (Single Folder)
# ==========================
class PhaseCFruitDataset(Dataset):
    """Dataset for Phase C - Fruit Classification
    Assumes all images are in one folder, and class name is before first '_' in filename.
    Example: Apple_Gala_1.jpg -> class 'Apple'
    """
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform
        self.data = list(self.folder_path.glob("*.jpg"))
        if len(self.data) == 0:
            raise ValueError(f"No images found in {folder_path}")

        # Extract class names from filenames
        self.labels = [img_file.name.split("_")[0] for img_file in self.data]
        self.class_names = sorted(list(set(self.labels)))
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label_name = self.labels[idx]
        label = self.class_to_idx[label_name]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ==========================
# Transformations
# ==========================
transform = transforms.Compose([
    transforms.Resize((192, 256)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================
# Model
# ==========================
class FruitClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# ==========================
# Training Script
# ==========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    train_path = r"data/processed/phaseC/train/fruit"
    val_path   = r"data/processed/phaseC/val/fruit"

    # Datasets
    train_dataset = PhaseCFruitDataset(train_path, transform=transform)
    val_dataset   = PhaseCFruitDataset(val_path, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model, Loss, Optimizer
    model = FruitClassifier(num_classes=len(train_dataset.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        # ======== Training ========
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss /= len(train_loader)
        train_acc = correct / total

        # ======== Validation ========
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # ==== Save Model ====
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx": train_dataset.class_to_idx,
        "idx_to_class": train_dataset.idx_to_class
    }, "phaseC_model.pt")
    print("✓ Model saved as phaseC_model.pt")
