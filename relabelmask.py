import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

# ====================================================================
# FRUIT NAME TO CLASS ID MAPPING
# ====================================================================
FRUIT_NAME_TO_CLASS = {
    'banana': 1,
    'orange': 2,
    'mango': 3,
    'apple': 4,
    'red_apple': 4,
    'green_apple': 5,
    'dragon_fruit': 6,
    'dragonfruit': 6,
    'plum': 7,
    'peach': 8,
    'kiwi': 9,
    'tomato': 10,
    'strawberry': 11,
    'pear': 12,
    'lemon': 13,
    'lime': 14,
    'coconut': 15,
    'watermelon': 16,
    'passion_fruit': 17,
    'papaya': 18,
    'guava': 19,
    'grape': 20,
    'hog_plum': 5,  # Example - assign to green category
    'green_coconut': 5,
}

def detect_fruit_from_filename(filename):
    """
    Detect fruit type from filename
    Returns class_id or None if not detected
    """
    filename_lower = filename.lower()
    
    for fruit_name, class_id in FRUIT_NAME_TO_CLASS.items():
        if fruit_name in filename_lower:
            return class_id, fruit_name
    
    return None, None

def relabel_masks(image_dir, mask_dir, output_mask_dir, backup=True):
    """
    Relabel binary masks to multi-class based on image filenames
    
    Args:
        image_dir: Directory with fruit images
        mask_dir: Directory with binary masks (0=background, 1=fruit)
        output_mask_dir: Where to save relabeled masks
        backup: Whether to backup original masks
    """
    
    print("="*60)
    print("AUTOMATIC MASK RELABELING")
    print("="*60 + "\n")
    
    # Backup original masks
    if backup:
        backup_dir = mask_dir + "_backup"
        if not os.path.exists(backup_dir):
            print(f"Creating backup at: {backup_dir}")
            shutil.copytree(mask_dir, backup_dir)
            print("✓ Backup created\n")
    
    # Create output directory
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Get files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks\n")
    
    if len(image_files) != len(mask_files):
        print("⚠️ Warning: Different number of images and masks!")
    
    # Process each mask
    stats = {
        'processed': 0,
        'detected': 0,
        'not_detected': 0,
        'class_counts': {}
    }
    
    print("Processing masks...")
    for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files)):
        
        # Detect fruit type from image filename
        class_id, fruit_name = detect_fruit_from_filename(img_file)
        
        # Load mask
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path))
        
        if class_id is not None:
            # Relabel: wherever mask=1 (fruit), change to class_id
            new_mask = np.where(mask > 0, class_id, 0)
            stats['detected'] += 1
            
            if class_id not in stats['class_counts']:
                stats['class_counts'][class_id] = 0
            stats['class_counts'][class_id] += 1
        else:
            # Couldn't detect fruit type - keep as binary or use default
            new_mask = mask
            stats['not_detected'] += 1
            print(f"⚠️ Could not detect fruit type: {img_file}")
        
        # Save relabeled mask
        output_path = os.path.join(output_mask_dir, mask_file)
        Image.fromarray(new_mask.astype(np.uint8)).save(output_path)
        
        stats['processed'] += 1
    
    # Print statistics
    print("\n" + "="*60)
    print("RELABELING COMPLETE")
    print("="*60)
    print(f"\nProcessed: {stats['processed']} masks")
    print(f"Detected: {stats['detected']} masks")
    print(f"Not detected: {stats['not_detected']} masks")
    
    print("\nClass distribution:")
    for class_id in sorted(stats['class_counts'].keys()):
        count = stats['class_counts'][class_id]
        # Find fruit name
        fruit_name = [name for name, cid in FRUIT_NAME_TO_CLASS.items() if cid == class_id][0]
        print(f"  Class {class_id} ({fruit_name}): {count} images")
    
    print(f"\n✓ Relabeled masks saved to: {output_mask_dir}")
    
    if stats['not_detected'] > 0:
        print(f"\n⚠️ {stats['not_detected']} masks could not be automatically labeled")
        print("Please check the filenames or manually label these masks")

def relabel_all_splits(base_dir):
    """
    Relabel masks for train, val, and test splits
    """
    splits = ['train', 'val']
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"PROCESSING {split.upper()} SPLIT")
        print('='*60 + '\n')
        
        image_dir = os.path.join(base_dir, split, 'images')
        mask_dir = os.path.join(base_dir, split, 'masks')
        output_mask_dir = os.path.join(base_dir, split, 'masks_relabeled')
        
        if os.path.exists(image_dir) and os.path.exists(mask_dir):
            relabel_masks(image_dir, mask_dir, output_mask_dir, backup=True)
        else:
            print(f"⚠️ Skipping {split}: directories not found")

def preview_relabeled_masks(original_mask_dir, relabeled_mask_dir, output_dir, num_samples=5):
    """
    Create side-by-side comparison of original vs relabeled masks
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    mask_files = sorted([f for f in os.listdir(original_mask_dir) if f.endswith('.png')])
    
    print(f"\nCreating comparison visualizations...")
    
    for i, mask_file in enumerate(mask_files[:num_samples]):
        original_path = os.path.join(original_mask_dir, mask_file)
        relabeled_path = os.path.join(relabeled_mask_dir, mask_file)
        
        if not os.path.exists(relabeled_path):
            continue
        
        original = np.array(Image.open(original_path))
        relabeled = np.array(Image.open(relabeled_path))
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title(f'Original Mask\nValues: {np.unique(original)}')
        axes[0].axis('off')
        
        axes[1].imshow(relabeled, cmap='nipy_spectral')
        axes[1].set_title(f'Relabeled Mask\nValues: {np.unique(relabeled)}')
        axes[1].axis('off')
        
        output_path = os.path.join(output_dir, f"comparison_{i+1}_{mask_file}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {output_path}")

if __name__ == '__main__':
    
    # ====================================================================
    # OPTION 1: Relabel all splits automatically
    # ====================================================================
    BASE_DIR = r"C:\Users\hp\Desktop\cvproject\food_fruit\data\processed\phaseDE"
    
    print("="*60)
    print("AUTOMATIC MASK RELABELING TOOL")
    print("="*60)
    print("\nThis will:")
    print("1. Backup your original masks")
    print("2. Detect fruit type from image filenames")
    print("3. Relabel masks with correct class IDs")
    print("4. Create new 'masks_relabeled' folders")
    print("\n" + "="*60 + "\n")
    
    relabel_all_splits(BASE_DIR)
    
    # ====================================================================
    # Create comparison visualizations
    # ====================================================================
    print("\n" + "="*60)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*60)
    
    TRAIN_MASK_DIR = os.path.join(BASE_DIR, "train", "masks")
    RELABELED_MASK_DIR = os.path.join(BASE_DIR, "train", "masks_relabeled")
    COMPARISON_DIR = r"C:\Users\hp\Desktop\cvproject\food_fruit\relabeling_comparison"
    
    if os.path.exists(RELABELED_MASK_DIR):
        preview_relabeled_masks(TRAIN_MASK_DIR, RELABELED_MASK_DIR, COMPARISON_DIR, num_samples=5)
    
    print("\n" + "="*60)
    print("✓ ALL DONE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the comparison images in:", COMPARISON_DIR)
    print("2. Update your training script to use 'masks_relabeled' folders")
    print("3. Re-run training with the new multi-class masks")