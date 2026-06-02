import os
import shutil
import random
from pathlib import Path

def setup_real_baseline(base_dir, annotations_dir, out_dir, seed=42):
    """
    Phase 1: Locks the real dataset splits.
    Stratifies ladder vs no_ladder so distributions remain identical in base.
    """
    random.seed(seed)
    base_path = Path(base_dir)
    out_path = Path(out_dir)
    
    # standard YOLO format folder inside the annotations
    ann_dir = Path(annotations_dir) / "labels_bbox" 
    if not ann_dir.exists():
        ann_dir = Path(annotations_dir) # fallback if directly in the root
        
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            (out_path / split / subdir).mkdir(parents=True, exist_ok=True)

    def process_class(cls_name):
        cls_dir = base_path / cls_name
        if not cls_dir.exists():
            return
            
        imgs = sorted(list(cls_dir.glob("*.*")))
        imgs = [img for img in imgs if img.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        random.shuffle(imgs)
        
        n = len(imgs)
        n_train, n_val = int(n * 0.7), int(n * 0.15)
        
        splits = {
            'train': imgs[:n_train],
            'val': imgs[n_train:n_train + n_val],
            'test': imgs[n_train + n_val:]
        }
        
        for split, split_imgs in splits.items():
            for img in split_imgs:
                shutil.copy2(img, out_path / split / 'images' / img.name)
                
                # Copy or touch label
                label_name = img.stem + ".txt"
                label_target = out_path / split / 'labels' / label_name
                
                # Label studio adds an 8-character prefix + "-" to the original filename
                # Use rglob so we match files in nested folders (Label Studio exports may be nested)
                label_matches = list(ann_dir.rglob(f"*{img.stem}.txt"))
                
                if label_matches and cls_name != 'no_ladder':
                    shutil.copy2(label_matches[0], label_target)
                else:
                    # Empty text file for negative samples (no_ladder) or missing labels
                    label_target.touch()

    # Process both distinct real classes
    process_class('ladder')
    process_class('no_ladder')

def inject_synthetic_to_train(baseline_dir, synth_images_dir, synth_annotations_dir, synth_percentage):
    """
    Phase 2: Creates a new dataset specifically for training with MoBI data.
    Copies the locked baseline, then injects a SPECIFIC PERCENTAGE of synthetic data relative to the real train size.
    """
    baseline_dir = Path(baseline_dir)
    synth_images_dir = Path(synth_images_dir)
    synth_annotations_dir = Path(synth_annotations_dir)
    
    new_dataset_dir = baseline_dir.parent / f"dataset_mobi_augmented_{int(synth_percentage*100)}pct"
    
    # 1. Copy the entire baseline to preserve exact val/test arrays
    if new_dataset_dir.exists():
        shutil.rmtree(new_dataset_dir)
    shutil.copytree(baseline_dir, new_dataset_dir)
    
    # 2. Inject synthetic data into the TRAIN split only
    train_img_dir = new_dataset_dir / 'train' / 'images'
    train_label_dir = new_dataset_dir / 'train' / 'labels'
    
    # Find original distribution
    original_train_labels = list((baseline_dir / 'train' / 'labels').glob("*.txt"))
    base_no_ladder_count = 0
    base_ladder_count = 0
    
    for lbl in original_train_labels:
        if lbl.stat().st_size == 0:
            base_no_ladder_count += 1
        else:
            base_ladder_count += 1
            
    total_real_train = base_ladder_count + base_no_ladder_count
    
    # Calculate how many synthetic images map to the requested percentage of the total real database
    target_synth_count = int(total_real_train * synth_percentage)
    
    synth_imgs = sorted([f for f in synth_images_dir.glob("*.*") if f.suffix.lower() in ['.jpg', '.png']])
    random.seed(42)
    random.shuffle(synth_imgs)
    
    # Slice only the requested amount
    synth_imgs_to_inject = synth_imgs[:target_synth_count]
    
    synth_added = 0
    for img in synth_imgs_to_inject:
        shutil.copy2(img, train_img_dir / img.name)
        
        # Look for MoBI annotations
        label_matches = list(synth_annotations_dir.rglob(f"*{img.stem}.txt"))
        if label_matches:
            shutil.copy2(label_matches[0], train_label_dir / (img.stem + ".txt"))
        synth_added += 1

    # Create proper dataset.yaml for the variant (use relative paths for cross-platform compatibility)
    with open(new_dataset_dir / 'dataset.yaml', 'w') as f:
        f.write('path: .\n')
        f.write('train: train/images\n')
        f.write('val: val/images\n')
        f.write('test: test/images\n')
        f.write('nc: 1\n')
        f.write('names:\n  0: ladder\n')

    # Print logic
    new_ladder_count = base_ladder_count + synth_added
    target_no_ladder_count = int(new_ladder_count * (base_no_ladder_count / base_ladder_count)) if base_ladder_count > 0 else 0
    backgrounds_needed = target_no_ladder_count - base_no_ladder_count
    
    print(f"\n--- Variant: {int(synth_percentage*100)}% Synthetic Injection ---")
    print(f"Created: {new_dataset_dir.name}")
    print(f"Total Base Real Images in Train: {total_real_train}")
    print(f"Requested {synth_percentage*100}% Synthetic: Injected {synth_added} MoBI images.")
    print(f"To re-balance back to base ratio, you need to add {backgrounds_needed} extra 'no_ladder' background images into: {train_img_dir.name}")

if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent.resolve()
    ROBO_IMAGES = CURRENT_DIR / 'robo_images'
    REAL_ANNS = ROBO_IMAGES / 'yolo_annotations'
    SYNTH_ANNS = ROBO_IMAGES / 'yolo_annotations_mobi'
    BASELINE_OUTPUT = CURRENT_DIR / 'dataset_real_baseline'
    
    # Phase 2 Paths
    SYNTH_MOBI_IMGS = CURRENT_DIR / 'MoBI_outputs' / 'selected_synthetic_images'
    SYNTH_MOBI_ANNS = ROBO_IMAGES / 'yolo_annotations_mobi'
    
    # Execute Phase 1: Lock Evaluation datasets based ONLY on real images
    setup_real_baseline(ROBO_IMAGES, REAL_ANNS, BASELINE_OUTPUT)
    
    # Ensure the script dynamically counts and sets the limits
    synth_pool_size = len([f for f in Path(SYNTH_MOBI_IMGS).glob("*.*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
    
    base_train_images_dir = Path(BASELINE_OUTPUT) / 'train' / 'images'
    real_train_size = len(list(base_train_images_dir.glob("*.*"))) if base_train_images_dir.exists() else 0
    
    if real_train_size > 0:
        max_pct = synth_pool_size / real_train_size
        
        # Create dataset variants at explicit intervals
        inject_synthetic_to_train(BASELINE_OUTPUT, SYNTH_MOBI_IMGS, SYNTH_MOBI_ANNS, synth_percentage=0.05)
        inject_synthetic_to_train(BASELINE_OUTPUT, SYNTH_MOBI_IMGS, SYNTH_MOBI_ANNS, synth_percentage=0.08)
        # Use whatever the maximum percentage of synthetic images we have available
        #inject_synthetic_to_train(BASELINE_OUTPUT, SYNTH_MOBI_IMGS, SYNTH_MOBI_ANNS, synth_percentage=max_pct)

