import shutil
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def process_class(cls_name, base_path, ann_dir, out_path):
    cls_dir = base_path / cls_name
    if not cls_dir.exists():
        return

    imgs = sorted(list(cls_dir.glob("*.*")))
    imgs = [img for img in imgs if img.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    random.shuffle(imgs)

    n = len(imgs)
    n_train, n_val = int(n * 0.7), int(n * 0.15)

    split_data = {
        'train': imgs[:n_train],
        'val': imgs[n_train:n_train + n_val],
        'test': imgs[n_train + n_val:]
    }

    for split, split_imgs in split_data.items():
        for img in split_imgs:
            shutil.copy2(img, out_path / split / 'images' / img.name)

            label_target = out_path / split / 'labels' / (img.stem + ".txt")
            label_matches = list(ann_dir.rglob(f"*{img.stem}.txt"))

            if label_matches and cls_name != 'no_ladder':
                shutil.copy2(label_matches[0], label_target)
            else:
                label_target.touch()


def setup_real_baseline(base_dir, annotations_dir, out_dir, seed):
    random.seed(seed)
    base_path = Path(base_dir)
    out_path = Path(out_dir)

    ann_dir = Path(annotations_dir) / "labels_bbox"
    if not ann_dir.exists():
        ann_dir = Path(annotations_dir)

    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            (out_path / split / subdir).mkdir(parents=True, exist_ok=True)

    process_class('ladder', base_path, ann_dir, out_path)
    process_class('no_ladder', base_path, ann_dir, out_path)
    try:
        with open(out_path / 'dataset.yaml', 'w') as f:
            f.write(f'path: {out_path.name}\n')
            f.write('train: train/images\n')
            f.write('val: val/images\n')
            f.write('test: test/images\n')
            f.write('nc: 1\n')
            f.write('names:\n  0: ladder\n')
    except Exception as e:
        print(f"yaml write failed: {e}")


def augment_images(source_imgs, target_img_dir, target_lbl_dir, count, seed, is_ladder, baseline_label_dir):
    rng = random.Random(seed)
    selected = [rng.choice(source_imgs) for _ in range(count)]

    for i, src in enumerate(selected):
        rng_aug = random.Random(seed + i)
        np.random.seed(seed + i)
        torch_seed = seed + i

        img = Image.open(src).convert("RGB")
        w, h = img.size

        # Geometric transforms
        if rng_aug.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        angle = rng_aug.choice([-1, 1]) * rng_aug.uniform(5, 25)
        img = img.rotate(angle, expand=False)

        crop_frac = rng_aug.uniform(0.6, 0.9)
        cw, ch = int(w * crop_frac), int(h * crop_frac)
        left = rng_aug.randint(0, w - cw)
        top = rng_aug.randint(0, h - ch)
        img = img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)

        # Colour jitter
        random.seed(torch_seed)
        jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
        img = jitter(img)

        # Cutout mosaic: multiple random black squares
        arr = np.array(img)
        n_cuts = rng_aug.randint(2, 6)
        for _ in range(n_cuts):
            cut_w = rng_aug.randint(int(w * 0.05), int(w * 0.15))
            cut_h = rng_aug.randint(int(h * 0.05), int(h * 0.15))
            cx = rng_aug.randint(0, w - cut_w)
            cy = rng_aug.randint(0, h - cut_h)
            arr[cy:cy + cut_h, cx:cx + cut_w] = 0
        img = Image.fromarray(arr)

        prefix = "aug_ladder" if is_ladder else "aug_bg"
        aug_name = f"{prefix}_{i:04d}_{src.stem}{src.suffix}"
        img.save(target_img_dir / aug_name)

        aug_label = target_lbl_dir / f"{prefix}_{i:04d}_{src.stem}.txt"
        if is_ladder:
            label_src = baseline_label_dir / (src.stem + ".txt")
            if label_src.exists():
                shutil.copy2(label_src, aug_label)
        else:
            aug_label.touch()


def inject_synthetic_to_train(baseline_dir, synth_images_dir, synth_annotations_dir, synth_percentage, aug_percentage, seed):
    baseline_dir = Path(baseline_dir)
    synth_images_dir = Path(synth_images_dir)
    synth_annotations_dir = Path(synth_annotations_dir)

    output_root = baseline_dir.parent / "dataset_mobi_synth_aug"
    output_root.mkdir(exist_ok=True)
    new_dataset_dir = output_root / f"dataset_mobi_synth{int(synth_percentage*100)}pct_aug{int(aug_percentage*100)}pct"

    if new_dataset_dir.exists():
        shutil.rmtree(new_dataset_dir)
    shutil.copytree(baseline_dir, new_dataset_dir)

    train_img_dir = new_dataset_dir / 'train' / 'images'
    train_label_dir = new_dataset_dir / 'train' / 'labels'

    original_train_labels = list((baseline_dir / 'train' / 'labels').glob("*.txt"))
    base_no_ladder_count = 0
    base_ladder_count = 0

    for lbl in original_train_labels:
        if lbl.stat().st_size == 0:
            base_no_ladder_count += 1
        else:
            base_ladder_count += 1

    total_real_train = base_ladder_count + base_no_ladder_count
    target_synth_count = int(total_real_train * synth_percentage)

    synth_imgs = sorted([f for f in synth_images_dir.glob("*.*") if f.suffix.lower() in ['.jpg', '.png']])
    rng = random.Random(seed)
    rng.shuffle(synth_imgs)
    synth_imgs_to_inject = synth_imgs[:target_synth_count]

    synth_added = 0
    for img in synth_imgs_to_inject:
        shutil.copy2(img, train_img_dir / img.name)
        label_matches = list(synth_annotations_dir.rglob(f"*{img.stem}.txt"))
        if label_matches:
            shutil.copy2(label_matches[0], train_label_dir / (img.stem + ".txt"))
        synth_added += 1

    x = round(aug_percentage * base_ladder_count)
    z = round((base_no_ladder_count * (synth_added + x)) / base_ladder_count)

    ladder_src = [f for f in (baseline_dir / 'train' / 'images').glob("*.*")
                  if (baseline_dir / 'train' / 'labels' / (f.stem + ".txt")).stat().st_size > 0]
    bg_src = [f for f in (baseline_dir / 'train' / 'images').glob("*.*")
              if (baseline_dir / 'train' / 'labels' / (f.stem + ".txt")).stat().st_size == 0]

    baseline_label_dir = baseline_dir / 'train' / 'labels'
    augment_images(ladder_src, train_img_dir, train_label_dir, x, seed, is_ladder=True, baseline_label_dir=baseline_label_dir)
    augment_images(bg_src, train_img_dir, train_label_dir, z, seed, is_ladder=False, baseline_label_dir=baseline_label_dir)

    with open(new_dataset_dir / 'dataset.yaml', 'w') as f:
        f.write(f'path: dataset_mobi_synth_aug/{new_dataset_dir.name}\n')
        f.write('train: train/images\n')
        f.write('val: val/images\n')
        f.write('test: test/images\n')
        f.write('nc: 1\n')
        f.write('names:\n  0: ladder\n')

    print(f"\n--- Variant: synth {int(synth_percentage*100)}% aug {int(aug_percentage*100)}% ---")
    print(f"Created: {new_dataset_dir.name}")
    print(f"Real train: {total_real_train} | Synthetic injected: {synth_added} | Ladder aug: {x} | Background aug: {z}")


if __name__ == "__main__":
    SEED = 42
    CURRENT_DIR = Path(__file__).parent.resolve()
    ROBO_IMAGES = CURRENT_DIR / 'robo_images'
    REAL_ANNS = ROBO_IMAGES / 'yolo_annotations'
    BASELINE_OUTPUT = CURRENT_DIR / 'dataset_real_baseline'
    SYNTH_MOBI_IMGS = CURRENT_DIR / 'MoBI_outputs' / 'selected_synthetic_images'
    SYNTH_MOBI_ANNS = ROBO_IMAGES / 'yolo_annotations_mobi'

    setup_real_baseline(ROBO_IMAGES, REAL_ANNS, BASELINE_OUTPUT, seed=SEED)

    if Path(BASELINE_OUTPUT).exists():
        for synth_pct in [0.05, 0.08, 0.10, 0.20]:
            for aug_pct in [0.05, 0.08, 0.10, 0.20]:
                inject_synthetic_to_train(BASELINE_OUTPUT, SYNTH_MOBI_IMGS, SYNTH_MOBI_ANNS, synth_percentage=synth_pct, aug_percentage=aug_pct, seed=SEED)