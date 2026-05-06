import os
import cv2
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SAM2_REPO       = "./sam2"
SAM2_CHECKPOINT = "./sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"

_sam2_abs  = os.path.abspath(SAM2_REPO)
sys.path.insert(0, _sam2_abs)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

LADDER_CSV = "./MoBI_outputs/ladder_segmentations_real_size4_debug/ladder_instances.csv"
OUTPUT_DIR = "./MoBI_outputs/ladder_mask_postprocess_debug4_sam2_hierarchy"
LABELS_DIR = "./robo_images/yolo_annotations/labels"

TARGET_INSTANCE = "02e903d139804b5291b44f8aa5cf87b4_front_5_March_2026_12-58_ladder00"

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_yolo_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if int(parts[0].strip().lstrip('\ufeff')) != 0:
                continue
            coords = [float(p) for p in parts[1:]]
            if len(coords) == 4:
                cx, cy, w, h = coords
                x1 = int(max(0, (cx - w / 2) * img_w))
                y1 = int(max(0, (cy - h / 2) * img_h))
                x2 = int(min(img_w, (cx + w / 2) * img_w))
                y2 = int(min(img_h, (cy + h / 2) * img_h))
            else:
                xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
                ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
                x1 = int(max(0, min(xs)))
                y1 = int(max(0, min(ys)))
                x2 = int(min(img_w, max(xs)))
                y2 = int(min(img_h, max(ys)))
            boxes.append([x1, y1, x2, y2])
    return boxes

def main():
    device = get_device()
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LADDER_CSV, index_col=0)
    
    print("Loading SAM2 for Logit Threshold Testing...")
    model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(model)

    processed_count = 0

    for idx, row in df.iterrows():
        instance_id = row["instance_id"]
        
        if TARGET_INSTANCE not in instance_id:
            continue
            
        print(f"\nFound target instance: {instance_id}")

        processed_count += 1
        
        img_path = row["image_path"]
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"  -> ERROR: Could not read image {img_path}")
            continue
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_rgb.shape[:2]

        img_stem = Path(img_path).stem
        matches = list(Path(LABELS_DIR).glob(f"*{img_stem}.txt"))
        if not matches:
            print(f"  -> ERROR: Label file not found for {img_stem} in {LABELS_DIR}")
            continue
            
        label_path = str(matches[0])
        boxes = load_yolo_boxes(label_path, img_w, img_h)
        if not boxes: 
            print(f"  -> ERROR: No valid boxes loaded from {label_path}")
            continue
            
        print(f"  -> Running SAM2 inference...")
        box_xyxy = boxes[0] # assuming index 0 for target instance

        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(box_xyxy)[None, :],
            multimask_output=True,
            return_logits=True
        )
        
        # Squeeze out the batch dimension to safely get (C, H, W) and (C,) arrays
        scores = np.squeeze(scores)
        logits = np.squeeze(logits)
        
        best = np.argmax(scores)
        best_logits = logits[best] 
        
        # resize SAM2 (256x256) to original res (1920x1080)
        if best_logits.shape != (img_h, img_w):
            best_logits = cv2.resize(best_logits, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        mask_th_default = (best_logits > 0.0).astype(np.uint8)
        mask_th_1       = (best_logits > 1.0).astype(np.uint8)
        mask_th_2       = (best_logits > 2.0).astype(np.uint8)
        mask_th_3       = (best_logits > 3.0).astype(np.uint8)

        # Focus Crop
        pad = 20
        x1, y1, x2, y2 = box_xyxy
        cy1, cy2 = max(0, y1 - pad), min(img_h, y2 + pad)
        cx1, cx2 = max(0, x1 - pad), min(img_w, x2 + pad)
        
        crop_rgb = image_rgb[cy1:cy2, cx1:cx2].copy()

        def get_overlay(mask):
            overlay = crop_rgb.copy()
            crop_m = mask[cy1:cy2, cx1:cx2]
            if crop_m.size > 0 and crop_m.max() > 0:
                overlay[crop_m > 0] = (overlay[crop_m > 0] * 0.5 + np.array([0, 255, 100]) * 0.5).astype(np.uint8)
            return overlay

        # --- Visual Comparison ---
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f"{instance_id} - SAM2 Logit Thresholds", color="white")
        fig.patch.set_facecolor("#111")

        axes[0].imshow(crop_rgb)
        axes[0].set_title("Original Crop", color="white", fontsize=10)

        axes[1].imshow(get_overlay(mask_th_default))
        axes[1].set_title("Default (Logits > 0.0)", color="white", fontsize=10)

        axes[2].imshow(get_overlay(mask_th_1))
        axes[2].set_title("Strict (Logits > 1.0)", color="white", fontsize=10)
        
        axes[3].imshow(get_overlay(mask_th_2))
        axes[3].set_title("Very Strict (Logits > 2.0)", color="white", fontsize=10)

        axes[4].imshow(get_overlay(mask_th_3))
        axes[4].set_title("Extreme (Logits > 3.0)", color="white", fontsize=10)

        for ax in axes:
            ax.axis("off")
            ax.set_facecolor("#111")
        
        plt.tight_layout()
        out_file = out_dir / f"{instance_id}_sam2_logits.jpg"
        fig.savefig(str(out_file), facecolor="#111", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  -> Generated SAM2 Logits Debug: {out_file}")

    if processed_count == 0:
        print(f"\nWARNING: Target instance '{TARGET_INSTANCE}' was not found in the CSV!")
        print("Available instances are:")
        print(df["instance_id"].head().values)

if __name__ == "__main__":
    main()
