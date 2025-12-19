import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

# ====================================================================
# FRUIT TYPE TO COLOR MAPPING - DEFINE YOUR FRUITS HERE
# ====================================================================
# Format: class_id: (color_rgb, fruit_name)
FRUIT_COLORS = {
    0: ([0, 0, 0], "Background"),
    1: ([255, 255, 0], "Banana"),           # Yellow
    2: ([255, 165, 0], "Orange"),           # Orange
    3: ([255, 200, 0], "Mango"),            # Golden Yellow
    4: ([255, 0, 0], "Apple (Red)"),        # Red
    5: ([0, 255, 0], "Apple (Green)"),      # Green
    6: ([255, 20, 147], "Dragon Fruit"),    # Pink
    7: ([138, 43, 226], "Plum"),            # Purple
    8: ([255, 105, 180], "Peach"),          # Pink-Orange
    9: ([34, 139, 34], "Kiwi"),             # Dark Green
    10: ([255, 69, 0], "Tomato"),           # Red-Orange
    11: ([220, 20, 60], "Strawberry"),      # Crimson
    12: ([154, 205, 50], "Pear"),           # Yellow-Green
    13: ([255, 215, 0], "Lemon"),           # Bright Yellow
    14: ([50, 205, 50], "Lime"),            # Lime Green
    15: ([138, 54, 15], "Coconut"),         # Brown
    16: ([255, 99, 71], "Watermelon"),      # Red
    17: ([186, 85, 211], "Passion Fruit"),  # Purple
    18: ([255, 228, 181], "Papaya"),        # Light Orange
    19: ([240, 128, 128], "Guava"),         # Light Coral
    20: ([218, 112, 214], "Grape"),         # Orchid
    # Add more as needed up to 31
}

# Fill remaining classes with gray if you have more
for i in range(21, 31):
    FRUIT_COLORS[i] = ([128, 128, 128], f"Unknown_{i}")

# ====================================================================
# MODEL ARCHITECTURE - Multi-class U-Net
# ====================================================================
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=31):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.bottleneck = self.conv_block(256, 512)
        
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self.conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        self.out = nn.Conv2d(32, num_classes, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.upconv4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.upconv3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.upconv2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.upconv1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.out(d1)

# ====================================================================
# DATASET - Multi-class Segmentation
# ====================================================================
class FruitSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=128):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png'))])
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') 
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        mask_np = np.array(mask)
        # Keep class labels as-is (0-30)
        mask_np = np.where(mask_np > 30, 30, mask_np) 

        image = self.transform(image)
        mask = torch.from_numpy(mask_np).long()
        return image, mask

# ====================================================================
# COLORIZE BASED ON FRUIT TYPE
# ====================================================================
def colorize_by_class(class_mask):
    """
    Color the mask based on fruit class
    Each fruit TYPE gets its consistent color
    """
    h, w = class_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Get unique classes in the mask
    unique_classes = np.unique(class_mask)
    
    for class_id in unique_classes:
        if class_id in FRUIT_COLORS:
            color, _ = FRUIT_COLORS[class_id]
            colored[class_mask == class_id] = color
        else:
            # Unknown class - use gray
            colored[class_mask == class_id] = [128, 128, 128]
    
    return colored

def get_class_legend():
    """Create a legend showing which color = which fruit"""
    legend_lines = ["CLASS ID -> COLOR -> FRUIT NAME"]
    legend_lines.append("="*40)
    for class_id in sorted(FRUIT_COLORS.keys()):
        color, name = FRUIT_COLORS[class_id]
        if class_id == 0:
            legend_lines.append(f"Class {class_id:2d}: RGB{color} - {name}")
        else:
            legend_lines.append(f"Class {class_id:2d}: RGB{color} - {name}")
    return "\n".join(legend_lines)

# ====================================================================
# TRAINING
# ====================================================================
def train_model(train_img, train_mask, val_img, val_mask, epochs=10, batch_size=4, model_path="fruit_multiclass_model.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    train_ds = FruitSegmentationDataset(train_img, train_mask)
    val_ds = FruitSegmentationDataset(val_img, val_mask)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = UNet(num_classes=31).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == masks).sum().item()
                total += masks.numel()
        
        val_acc = (correct / total) * 100
        print(f"Epoch {epoch+1}: Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to: {model_path}")
    return model

# ====================================================================
# INFERENCE WITH FRUIT TYPE COLORS
# ====================================================================
def predict_with_fruit_colors(model, image_path, output_path, img_size=128):
    """
    Predict and color based on FRUIT TYPE
    Banana always yellow, Apple always red, etc.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Load image
    original_img = Image.open(image_path).convert('RGB')
    original_size = original_img.size
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # Resize back to original size
    pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), original_size, 
                                   interpolation=cv2.INTER_NEAREST)
    
    # Color based on fruit CLASS (not random)
    colored_mask = colorize_by_class(pred_mask_resized)
    
    # Get detected fruits
    unique_classes = np.unique(pred_mask_resized)
    detected_fruits = []
    for class_id in unique_classes:
        if class_id > 0 and class_id in FRUIT_COLORS:
            _, fruit_name = FRUIT_COLORS[class_id]
            detected_fruits.append(fruit_name)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(colored_mask)
    title = f'Fruit Type Segmentation\n'
    if detected_fruits:
        title += f'Detected: {", ".join(detected_fruits)}'
    axes[1].set_title(title, fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    original_np = np.array(original_img)
    overlay = cv2.addWeighted(original_np, 0.5, colored_mask, 0.5, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return detected_fruits

# ====================================================================
# MAIN
# ====================================================================
if __name__ == '__main__':
    
    print("="*60)
    print("MULTI-CLASS FRUIT SEGMENTATION")
    print("Each fruit type gets its own consistent color!")
    print("="*60)
    print("\n" + get_class_legend() + "\n")
    print("="*60 + "\n")
    
    # TRAIN THE MODEL
    print("TRAINING MODEL...")
    model = train_model(
        train_img = r"C:\Users\hp\Desktop\cvproject\food_fruit\data\processed\phaseDE\train\images",
        train_mask = r"C:\Users\hp\Desktop\cvproject\food_fruit\data\processed\phaseDE\train\masks_relabeled",
        val_img = r"C:\Users\hp\Desktop\cvproject\food_fruit\data\processed\phaseDE\val\images",
        val_mask = r"C:\Users\hp\Desktop\cvproject\food_fruit\data\processed\phaseDE\val\masks_relabeled",
        epochs=10,
        model_path="fruit_multiclass_model.pt"
    )
    
    # TEST ON SINGLE IMAGE
    print("\n" + "="*60)
    print("TESTING...")
    print("="*60)
    
    test_image = r"C:\Users\hp\Desktop\cvproject\food_fruit\test_image.jpg"
    if os.path.exists(test_image):
        fruits = predict_with_fruit_colors(model, test_image, "result_fruit_types.png")
        print(f"✓ Detected: {', '.join(fruits)}")
    else:
        print(f"⚠ Test image not found: {test_image}")