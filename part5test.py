import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# ====================================================================
# FRUIT TYPE TO COLOR MAPPING - MUST MATCH TRAINING
# ====================================================================
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
}

for i in range(21, 31):
    FRUIT_COLORS[i] = ([128, 128, 128], f"Unknown_{i}")

# ====================================================================
# MODEL ARCHITECTURE
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
# COLORIZE BY FRUIT TYPE
# ====================================================================
def colorize_by_class(class_mask):
    """Color based on fruit CLASS - consistent colors"""
    h, w = class_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    unique_classes = np.unique(class_mask)
    
    for class_id in unique_classes:
        if class_id in FRUIT_COLORS:
            color, _ = FRUIT_COLORS[class_id]
            colored[class_mask == class_id] = color
        else:
            colored[class_mask == class_id] = [128, 128, 128]
    
    return colored

# ====================================================================
# PREDICTION
# ====================================================================
def predict_with_fruit_colors(model, image_path, output_path, img_size=128):
    """Predict with consistent fruit type colors"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    original_img = Image.open(image_path).convert('RGB')
    original_size = original_img.size
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(original_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), original_size, 
                                   interpolation=cv2.INTER_NEAREST)
    
    colored_mask = colorize_by_class(pred_mask_resized)
    
    # Get detected fruits
    unique_classes = np.unique(pred_mask_resized)
    detected_fruits = []
    for class_id in unique_classes:
        if class_id > 0 and class_id in FRUIT_COLORS:
            _, fruit_name = FRUIT_COLORS[class_id]
            detected_fruits.append((class_id, fruit_name))
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(colored_mask)
    title = 'Fruit Type Segmentation\n'
    if detected_fruits:
        fruit_names = [f"{name} (Class {cid})" for cid, name in detected_fruits]
        title += '\n'.join(fruit_names)
    axes[1].set_title(title, fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
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
# TEST RANDOM IMAGES
# ====================================================================
def test_random_images(model_path, val_dir, output_dir, num_images=10, img_size=128):
    """Test with consistent fruit type colors"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading model from: {model_path}")
    model = UNet(num_classes=31).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded!\n")
    
    # Get images
    img_dir = os.path.join(val_dir, "images")
    if not os.path.exists(img_dir):
        print(f"Error: Directory not found: {img_dir}")
        return
    
    all_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(all_images) == 0:
        print(f"Error: No images found in {img_dir}")
        return
    
    num_to_select = min(num_images, len(all_images))
    selected_images = random.sample(all_images, num_to_select)
    
    print(f"Testing {num_to_select} random images")
    print(f"Each fruit TYPE gets its CONSISTENT COLOR:")
    print("  • Banana = Yellow")
    print("  • Mango = Golden Yellow")
    print("  • Apple (Red) = Red")
    print("  • Orange = Orange")
    print("  etc.\n")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for idx, img_name in enumerate(selected_images, 1):
        print(f"\n[{idx}/{num_to_select}] Processing: {img_name}")
        
        img_path = os.path.join(img_dir, img_name)
        output_path = os.path.join(output_dir, f"result_{idx}_{img_name}")
        
        try:
            detected = predict_with_fruit_colors(model, img_path, output_path, img_size)
            fruit_list = [name for _, name in detected]
            print(f"  ✓ Detected: {', '.join(fruit_list) if fruit_list else 'No fruits'}")
            print(f"  ✓ Saved to: {output_path}")
            results.append((img_name, fruit_list, "Success"))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append((img_name, [], f"Error: {e}"))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    successful = sum(1 for r in results if r[2] == "Success")
    print(f"Successfully processed: {successful}/{num_to_select} images")
    print(f"Results saved to: {output_dir}")
    
    print("\nDetected fruits:")
    for img_name, fruits, status in results:
        if status == "Success" and fruits:
            print(f"  • {img_name}: {', '.join(fruits)}")
    
    print("\n✓ Testing complete!")
    print("Note: Same fruit type always has the SAME COLOR across all images!")

# ====================================================================
# MAIN
# ====================================================================
if __name__ == '__main__':
    MODEL_PATH = "fruit_multiclass_model.pt"
    VAL_DIR = r"C:\Users\hp\Desktop\cvproject\food_fruit\data\processed\phaseDE\val"
    OUTPUT_DIR = r"C:\Users\hp\Desktop\cvproject\food_fruit\test_results_consistent"
    NUM_IMAGES = 10
    
    print("="*60)
    print("MULTI-CLASS FRUIT SEGMENTATION TEST")
    print("Consistent colors per fruit type!")
    print("="*60 + "\n")
    
    test_random_images(
        model_path=MODEL_PATH,
        val_dir=VAL_DIR,
        output_dir=OUTPUT_DIR,
        num_images=NUM_IMAGES
    )