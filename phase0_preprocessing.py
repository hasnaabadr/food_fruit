import os
import shutil
from pathlib import Path
from PIL import Image
import warnings
from tqdm import tqdm
import shutil
warnings.filterwarnings('ignore')

# Configuration
RAW_DATA_DIR = Path("Project Data")
PROCESSED_DATA_DIR = Path("data/processed")

# ==========================
# Create necessary directories
# ==========================
def create_phase_directories():
    phases = {
        'phaseA': ['train/food', 'train/fruit', 'val/food', 'val/fruit'],
        'phaseB': ['train/food', 'val/food'],
        'phaseC': ['train/fruit', 'val/fruit'],
        'phaseDE': ['train/images', 'train/masks', 'val/images', 'val/masks']
    }
    
    for phase, subdirs in phases.items():
        for subdir in subdirs:
            (PROCESSED_DATA_DIR / phase / subdir).mkdir(parents=True, exist_ok=True)
    
    print("✓ Phase directories created")

############################
#refresh run
############################


if PROCESSED_DATA_DIR.exists():
    shutil.rmtree(PROCESSED_DATA_DIR)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Phase A preprocessing
# ==========================
def preprocess_phase_a():
    print("\nPHASE A: Binary Classifier (224x224)")

    # --------- Food Train ---------
    food_train_dir = RAW_DATA_DIR / "Food" / "Train"
    dest_food_train = PROCESSED_DATA_DIR / "phaseA/train/food"
    count = 0
    for class_dir in tqdm(food_train_dir.iterdir(), desc="Food Train"):
        if class_dir.is_dir():
            for img_file in class_dir.glob("*.jpg"):
                try:
                    img = Image.open(img_file).convert('RGB').resize((224,224), Image.LANCZOS)
                    img.save(dest_food_train / f"{class_dir.name}_{img_file.stem}.jpg", quality=85, optimize=True)
                    count += 1
                except: pass
    print(f"Food Train: {count} images processed")

    # --------- Food Val ---------
    food_val_dir = RAW_DATA_DIR / "Food" / "Validation"
    dest_food_val = PROCESSED_DATA_DIR / "phaseA/val/food"
    count = 0
    for class_dir in tqdm(food_val_dir.iterdir(), desc="Food Val"):
        if class_dir.is_dir():
            for img_file in class_dir.glob("*.jpg"):
                try:
                    img = Image.open(img_file).convert('RGB').resize((224,224), Image.LANCZOS)
                    img.save(dest_food_val / f"{class_dir.name}_{img_file.stem}.jpg", quality=85, optimize=True)
                    count += 1
                except: pass
    print(f"Food Val: {count} images processed")

    # --------- Fruit Train ---------
    fruit_train_dir = RAW_DATA_DIR / "Fruit" / "Train"
    dest_fruit_train = PROCESSED_DATA_DIR / "phaseA/train/fruit"
    count = 0
    for class_dir in tqdm(fruit_train_dir.iterdir(), desc="Fruit Train"):
        images_dir = class_dir / "images"
        if images_dir.exists():
            for img_file in images_dir.glob("*.jpg"):
                try:
                    img = Image.open(img_file).convert('RGB').resize((224,224), Image.LANCZOS)
                    img.save(dest_fruit_train / f"{class_dir.name}_{img_file.stem}.jpg", quality=85, optimize=True)
                    count += 1
                except: pass
    print(f"Fruit Train: {count} images processed")

    # --------- Fruit Val ---------
    fruit_val_dir = RAW_DATA_DIR / "Fruit" / "Validation"
    dest_fruit_val = PROCESSED_DATA_DIR / "phaseA/val/fruit"
    count = 0
    for class_dir in tqdm(fruit_val_dir.iterdir(), desc="Fruit Val"):
        images_dir = class_dir / "images"
        if images_dir.exists():
            for img_file in images_dir.glob("*.jpg"):
                try:
                    img = Image.open(img_file).convert('RGB').resize((224,224), Image.LANCZOS)
                    img.save(dest_fruit_val / f"{class_dir.name}_{img_file.stem}.jpg", quality=85, optimize=True)
                    count += 1
                except: pass
    print(f"Fruit Val: {count} images processed")


# ==========================
# Fill Phase B/C from Phase A 
# ==========================
def fill_phase_b_c():
    # --------- Phase B: organize by class ---------
    for split in ["train","val"]:
        src = PROCESSED_DATA_DIR / f"phaseA/{split}/food"
        dest_root = PROCESSED_DATA_DIR / f"phaseB/{split}/food"
        dest_root.mkdir(parents=True, exist_ok=True)

        for img_file in src.glob("*.jpg"):
 
            class_name = img_file.stem.split("_")[0]
            dest_class_dir = dest_root / class_name
            dest_class_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_file, dest_class_dir / img_file.name)

    # --------- Phase C: copy flat without changing logic ---------
    for split in ["train","val"]:
        src = PROCESSED_DATA_DIR / f"phaseA/{split}/fruit"
        dest_root = PROCESSED_DATA_DIR / f"phaseC/{split}/fruit"
        dest_root.mkdir(parents=True, exist_ok=True)

        for img_file in src.glob("*.jpg"):
            shutil.copy(img_file, dest_root / img_file.name)

    print("✓ Phase B organized by class, Phase C kept flat")


# ==========================
# Phase DE preprocessing (Segmentation)
# ==========================
def preprocess_phase_de():
    print("\nPHASE D/E: Segmentation (384x384)")

    splits = {
        "train": RAW_DATA_DIR / "Fruit" / "Train",
        "val":   RAW_DATA_DIR / "Fruit" / "Validation"
    }

    for split_name, split_dir in splits.items():
        dest_images = PROCESSED_DATA_DIR / f"phaseDE/{split_name}/images"
        dest_masks  = PROCESSED_DATA_DIR / f"phaseDE/{split_name}/masks"

        dest_images.mkdir(parents=True, exist_ok=True)
        dest_masks.mkdir(parents=True, exist_ok=True)

        count_img = 0
        count_mask = 0

        for class_dir in tqdm(split_dir.iterdir(), desc=f"{split_name}"):
            images_dir = class_dir / "Images"
            masks_dir  = class_dir / "Mask"

            if not images_dir.exists() or not masks_dir.exists():
                continue

            for img_file in images_dir.glob("*.jpg"):
                mask_file = masks_dir / f"{img_file.stem}_mask.png"

                if not mask_file.exists():
                    print(f"Mask not found for {img_file.name}")
                    continue

                try:
                    # فتح الصورة وتغيير الحجم
                    img = Image.open(img_file).convert("RGB").resize((384, 384), Image.LANCZOS)
                    mask = Image.open(mask_file).convert("L").resize((384, 384), Image.NEAREST)

                    # حفظ الصور والماسكات
                    img_save_path = dest_images / f"{class_dir.name}_{img_file.name}"
                    mask_save_path = dest_masks / f"{class_dir.name}_{mask_file.name}"

                    if not img_save_path.exists():
                        img.save(img_save_path, quality=85, optimize=True)
                    if not mask_save_path.exists():
                        mask.save(mask_save_path)

                    count_img += 1
                    count_mask += 1

                except Exception as e:
                    print(f"Error processing {img_file.name}: {e}")

        print(f"{split_name}: {count_img} images, {count_mask} masks processed")

# ==========================
# Print Summary
# ==========================
def print_summary():
    print("\nPREPROCESSING COMPLETE - SUMMARY")
    for phase in ["phaseA","phaseB","phaseC","phaseDE"]:
        phase_dir = PROCESSED_DATA_DIR / phase
        if phase_dir.exists():
            img_count = len(list(phase_dir.glob("**/*.jpg")))
            mask_count = len(list(phase_dir.glob("**/*.png")))
            print(f"{phase}: Images={img_count}, Masks={mask_count}")

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    print("\n==== PHASE 0: HYBRID PREPROCESSING ====")
    create_phase_directories()
    preprocess_phase_a()
    fill_phase_b_c()
    preprocess_phase_de()
    print_summary()
