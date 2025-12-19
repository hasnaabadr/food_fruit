
import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import random
import pickle
import os
import torch.nn.functional as F
import torch.nn as nn

# ==========================
# Transform (same as validation)
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# ==========================
# Siamese Network (same as training)
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
# Compute mean embedding per class (Case 1)
# ==========================
def compute_train_embeddings(train_dir, model, device, cache_file="train_embeddings.pkl"):
    train_dir = Path(train_dir)

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            embeddings, labels = pickle.load(f)
        return embeddings, labels

    model.eval()
    embeddings = []
    labels = []

    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*.jpg"))
        if len(images) == 0:
            continue

        selected_imgs = random.sample(images, min(5, len(images)))
        embs = []

        for img_path in selected_imgs:
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.forward_once(img).cpu()
            embs.append(emb)

        class_emb = torch.mean(torch.vstack(embs), dim=0, keepdim=True)
        embeddings.append(class_emb)
        labels.append(class_dir.name)

    embeddings = torch.vstack(embeddings)

    with open(cache_file, "wb") as f:
        pickle.dump((embeddings, labels), f)

    return embeddings, labels

# ==========================
# Part B - Case 1
# ==========================
def test_partB_case1(
    test_dir,
    train_dir=r"data\processed\phaseB\train\food",
    model_path="phaseB_model.pt",
    device=None,
    output_file="partB_case1_results.txt"
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork(embedding_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    train_embeddings, train_labels = compute_train_embeddings(train_dir, model, device)

    test_dir = Path(test_dir)
    results = []

    for img_file in test_dir.glob("*.jpg"):
        img = Image.open(img_file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            test_emb = model.forward_once(img_tensor).cpu()

        distances = torch.norm(train_embeddings - test_emb, dim=1)
        min_idx = torch.argmin(distances).item()
        predicted_class = train_labels[min_idx]

        results.append((img_file.name, predicted_class))

    with open(output_file, "w") as f:
        for img_name, pred_class in results:
            f.write(f"{img_name}: {pred_class}\n")

    print(f"✓ Case 1 results saved to {output_file}")
    return results

# ==========================
# Part B - Case 2
# ==========================
def test_partB_case2(anchor_folder, model_path="phaseB_model.pt", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork(embedding_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    anchor_folder = Path(anchor_folder)
    anchor_path = anchor_folder / "Anchor.jpg"

    if not anchor_path.exists():
        print("Anchor image not found!")
        return

    anchor_img = Image.open(anchor_path).convert("RGB")
    anchor_tensor = transform(anchor_img).unsqueeze(0).to(device)

    with torch.no_grad():
        anchor_emb = model.forward_once(anchor_tensor)

    other_images = [f for f in anchor_folder.glob("*.jpg") if f.name != "Anchor.jpg"]
    if len(other_images) == 0:
        print("No other images found!")
        return

    results = {}
    for img_path in other_images:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            img_emb = model.forward_once(img_tensor)

        dist = F.pairwise_distance(anchor_emb, img_emb)
        results[img_path.name] = dist.item()

    closest_img = min(results, key=results.get)

    print(f"Anchor image: {anchor_path.name}")
    print(f"Most similar image: {closest_img} (distance={results[closest_img]:.4f})")

    output_file = anchor_folder / "partB_case2_results.txt"
    with open(output_file, "w") as f:
        f.write(f"Anchor image: {anchor_path.name}\n")
        f.write(f"Most similar image: {closest_img} (distance={results[closest_img]:.4f})\n")

    print(f"✓ Case 2 results saved to {output_file}")
    return results

# ==========================
# Test Part C - Fruit Classification
# ==========================
import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

import torch
from torchvision import models
from pathlib import Path
from PIL import Image
from torchvision import transforms

class FruitClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        num_ftrs = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(num_ftrs, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

def test_partC(test_folder, model_path="phaseC_model.pt", device=None, output_file="partC_results.txt"):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    checkpoint = torch.load(model_path, map_location=device)
    idx_to_class = checkpoint["idx_to_class"]
    num_classes = len(idx_to_class)

    
    model = FruitClassifier(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_folder = Path(test_folder)
    results = []

    for img_path in test_folder.glob("*.jpg"):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred_idx = torch.max(outputs, 1)
            predicted_class = idx_to_class[pred_idx.item()]
        results.append((img_path.name, predicted_class))

    with open(output_file, "w") as f:
        for img_name, pred_class in results:
            f.write(f"{img_name}: {pred_class}\n")

    print(f"✓ Part C results saved to {output_file}")
    return results
