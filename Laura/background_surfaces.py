import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image as PILImage
from scipy.ndimage import label as scipy_label
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

BACKGROUNDS_DIR     = "./robo_images/no_ladder"
OUTPUT_DIR          = "./MoBI_outputs/background8_segf_filter_small_kernel"
DEPTH_ANYTHING_REPO = "./Depth-Anything-V2"
DEPTH_CHECKPOINT    = "./Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth"
SEGFORMER_MODEL     = "nvidia/segformer-b2-finetuned-ade-512-512"
LADDER_CSV          = "./MoBI_outputs/ladder_segmentations_real_size3/ladder_instances.csv" # change after each run -.-
PLACEMENTS_PER_IMAGE = 3
MIN_REGION_AREA_FRAC = 0.02
MAX_WALL_HEIGHT_FRAC = 0.75 # bottom N% of the image to be reachable from ground
GROUND_PROXIMITY_FRAC = 0.05 # within fraction of image height from bottom
PLACEMENTS_PER_IMAGE = 3 # to avoid too many options for Paint by Example


# ADE20K 150-class indices vertical structures
# wall, building, house, fence, railing, column/pillar, skyscraper, hovel/hut, tower
VALID_SURFACE_CLASSES = {0, 1, 25, 32, 38, 42, 48, 79, 84}
# ADE20K 150-class indices for vertical surfaces
# road, grass, sidewalk/pavement, earth/ground, path, dirt track, land/soil
GROUND_CLASSES = {3, 6, 9, 11, 13, 52, 91, 94}

# reference objects for ladder size
# person, door, windowpane, car, streetlight, pole, van, bicycle
SCALE_REFERENCE_CLASSES = {12: 1.7, 14: 2.1, 8: 1.5, 20: 1.5, 87: 4.0, 93: 0.9, 102: 2.2, 127: 1.2}

# typical ladder height in meters
LADDER_HEIGHT_M = 2.0

sys.path.insert(0, os.path.abspath(DEPTH_ANYTHING_REPO))
from depth_anything_v2.dpt import DepthAnythingV2

 
 
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
 
 
def load_depth_model(checkpoint, device):
    model = DepthAnythingV2(encoder="vitb", features=128, out_channels=[96, 192, 384, 768])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()
 
 
def load_segformer(model_name, device):
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model     = SegformerForSemanticSegmentation.from_pretrained(model_name)
    return processor, model.to(device).eval()
 
 
def estimate_depth(model, image_bgr):
    depth = model.infer_image(image_bgr)
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    return depth.astype(np.float32)
 
 
def segment_image(processor, model, image_rgb, device):
    inputs = processor(images=PILImage.fromarray(image_rgb), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    upsampled = torch.nn.functional.interpolate(
        outputs.logits,
        size=image_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    probs = torch.nn.functional.softmax(upsampled, dim=1)
    confidence_map = probs.max(dim=1).values.squeeze(0).cpu().numpy()
    seg_map = upsampled. argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16)
    return seg_map, confidence_map
 
 
def get_surface_mask(seg_map, valid_classes, img_h, max_height_frac):
    """ valid: intersection of vertical and horizontal surface """
    mask = np.zeros(seg_map.shape, dtype=np.uint8)
    for cls in valid_classes:
        mask[seg_map == cls] = 1
    # Zero out the upper portion — ladders can't reach there
    cutoff = int(img_h * (1.0 - max_height_frac))
    mask[:cutoff, :] = 0
    return mask
 
 
def get_ground_mask(seg_map, valid_classes):
    mask = np.zeros(seg_map.shape, dtype=np.uint8)
    for cls in valid_classes:
        mask[seg_map == cls] = 1
    return mask


def get_valid_wall_regions(surface_mask, img_area):
    labeled, n = scipy_label(surface_mask)
    regions = []
    for i in range(1, n+1):
        region = (labeled == i).astype(np.uint8)
        area = int(region.sum())
        if area < MIN_REGION_AREA_FRAC * img_area:
            continue
        ys, xs = np.where(region)
        regions.append((region, area, (float(xs.mean()), float(ys.mean()))))
    regions.sort(key=lambda r: r[1], reverse=True)
    return regions[:10]
 

def physical_length_to_pixels(physical_length_m, seg_map, depth_map, zone_depth_mean, confidence_map):
    estimates = []
    if zone_depth_mean < 1e-4:
        return None
        
    for cls, ref_h_m in SCALE_REFERENCE_CLASSES.items():
        cls_ys, cls_xs = np.where(seg_map == cls)
        if len(cls_ys) < 200:  # Strict pixel count
            continue
            
        high_conf = confidence_map[cls_ys, cls_xs] > 0.8
        cls_ys, cls_xs = cls_ys[high_conf], cls_xs[high_conf]
        if len(cls_ys) < 100:
            continue
            
        ref_disparity = float(np.median(depth_map[cls_ys, cls_xs]))
        
        # Avoid extreme depth differences
        if ref_disparity > 3.0 * zone_depth_mean or ref_disparity < 0.33 * zone_depth_mean:
            continue

        ref_px_height = float(cls_ys.max() - cls_ys.min())
        if ref_px_height < 30:
            continue
            
        # px_ladder = px_ref * (H_ladder / H_ref) * (disp_ladder / disp_ref)
        est = ref_px_height * (physical_length_m / ref_h_m) * (zone_depth_mean / ref_disparity)
        estimates.append(int(est))
        
    return int(np.median(estimates)) if estimates else None

 
def find_placement_zone(wall_region, ground_mask, depth_map, scaled_h, scaled_w, img_h, img_w, rng):
    """ box fits within the wall region + bottom is securely snapped to the oriented ground slope """
    wall_ys, wall_xs = np.where(wall_region > 0)
    if len(wall_xs) == 0:
        return None
 
    candidates = []
    for _ in range(500):
        idx = rng.integers(0, len(wall_xs))
        cx  = int(wall_xs[idx])
 
        x1 = max(0, cx - scaled_w // 2)
        x2 = min(img_w - 1, x1 + scaled_w)
 
        # Find the ground line explicitly across the width of the bounding box
        search_roi = ground_mask[:, x1:x2]
        ground_top_ys = np.argmax(search_roi > 0, axis=0)
        
        # Filter out columns where no ground was found (argmax returns 0 if all are 0)
        valid_cols = (search_roi[ground_top_ys, np.arange(x2-x1)] > 0)
        if valid_cols.sum() < (scaled_w * 0.5):  # Need ground under at least half the object
            continue
            
        # Compute ground slope using a line fit through the valid ground boundary points
        xs_valid = np.arange(x1, x2)[valid_cols]
        ys_valid = ground_top_ys[valid_cols]
        slope, intercept = np.polyfit(xs_valid, ys_valid, 1)
        
        # Calculate sloped y2 corner anchors (sunken 10px into the ground)
        y2_left_raw = int(slope * x1 + intercept) + 10
        y2_right_raw = int(slope * x2 + intercept) + 10
        
        y2_left = np.clip(y2_left_raw, 0, img_h - 1)
        y2_right = np.clip(y2_right_raw, 0, img_h - 1)
        
        # Use the highest point of the sloped bottom to define the top of the box
        y2_min = min(y2_left, y2_right)
        y1 = max(0, y2_min - scaled_h)
 
        if (x2 - x1) < scaled_w * 0.5 or (y2_min - y1) < scaled_h * 0.5:
            continue
 
        # Box must mostly overlap wall region
        # Use a simple bounding rect for the overlap check
        y2_max = max(y2_left, y2_right)
        if wall_region[y1:y2_max, x1:x2].mean() < 0.4:
            continue
 
        box_depth  = depth_map[y1:y2_max, x1:x2]
        depth_std  = float(box_depth.std())
        depth_mean = float(box_depth.mean())
        
        # Store left/right anchors instead of a flat y2
        candidates.append((depth_std, x1, y1, x2, y2_left, y2_right, depth_mean))
 
    if not candidates:
        return None
 
    candidates.sort(key=lambda c: c[0])
    _, x1, y1, x2, y2_l, y2_r, depth_mean = candidates[0]
    return x1, y1, x2, y2_l, y2_r, depth_mean
 
 
def colorize_seg_map(seg_map):
    h, w = seg_map.shape
    vis  = np.ones((h, w, 3), dtype=np.uint8) * 40
    for cls in VALID_SURFACE_CLASSES:
        vis[seg_map == cls] = [50, 200, 80]   # green = valid wall surface
    for cls in GROUND_CLASSES:
        vis[seg_map == cls] = [200, 160, 50]  # orange = ground
    return vis
 
 
def save_verification_row(image_bgr, depth_map, seg_map, placements, out_path, image_name, id2label):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(image_name, fontsize=10, color="#ccc")
    fig.patch.set_facecolor("#111")
 
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
 
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original", color="#aaa", fontsize=9)
 
    axes[1].imshow(depth_map, cmap="plasma")
    axes[1].set_title("Depth", color="#aaa", fontsize=9)
 
    axes[2].imshow(colorize_seg_map(seg_map))
    axes[2].set_title("Surfaces (green) / Ground (orange)", color="#aaa", fontsize=9)

    ref_vis = image_rgb.copy()
    for cls in SCALE_REFERENCE_CLASSES:
        cls_ys, cls_xs = np.where(seg_map == cls)
        if len(cls_ys) > 0:
            ref_vis[cls_ys, cls_xs] = [255, 100, 0]
            cy, cx = int(cls_ys.mean()), int(cls_xs.mean())
            axes[3].text(cx, cy, id2label.get(cls, str(cls)), color="white", fontsize=15) # name reference object
 
    axes[3].imshow(ref_vis)
    axes[3].set_title("Reference objects (orange)", color="#aaa", fontsize=9)

    axes[4].imshow(image_rgb)
    colors = ["red", "cyan", "yellow"]
    for i, p in enumerate(placements):
        x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
        y2_l, y2_r = p.get("y2_left", y2), p.get("y2_right", y2)
        color = colors[i % len(colors)]
        
        # Draw oriented polygon instead of flat rectangle
        poly = np.array([[x1, y1], [x2, y1], [x2, y2_r], [x1, y2_l]], dtype=np.int32)
        axes[4].add_patch(patches.Polygon(
            poly, closed=True,
            linewidth=2, edgecolor=color, facecolor="none"
        ))
        axes[4].text(x1, max(y1 - 4, 0), f"#{i}", color=color, fontsize=8)
        axes[4].set_title("Placement zones", color="#aaa", fontsize=9)
        axes[4].text(p["x1"], max(p["y1"] - 4, 0), f"#{i} {p.get('box_h','?')}px", color=color, fontsize=15)
    
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#111")
 
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
 
 
def save_intermediates(seg_map, surface_mask_pre, surface_mask_post, ground_mask, out_dir, image_name):
    seg_vis = colorize_seg_map(seg_map)
    cv2.imwrite(str(out_dir / f"{image_name}_seg.png"), cv2.cvtColor(colorize_seg_map(seg_map), cv2.COLOR_RGB2BGR))
    # cv2.imwrite(str(out_dir / f"{image_name}_surface_before_depth_filter.png"), surface_mask_pre  * 255)
    # cv2.imwrite(str(out_dir / f"{image_name}_surface_after_depth_filter.png"),  surface_mask_post * 255)
    # cv2.imwrite(str(out_dir / f"{image_name}_ground_mask.png"),                 ground_mask       * 255)
 
 
def main():
    device = get_device()
    print(f"Device: {device}")
 
    out_dir = Path(OUTPUT_DIR)
    verify_dir = out_dir / "verification"
    intermediates_dir = out_dir / "intermediates"
    mask_dir = out_dir / "PbE_masks"
    verify_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
 
    print("Loading Depth Anything V2...")
    depth_model = load_depth_model(DEPTH_CHECKPOINT, device)
 
    print("Loading SegFormer...")
    seg_processor, seg_model = load_segformer(SEGFORMER_MODEL, device)
    id2label = seg_model.config.id2label
 
    print("Loading ladder instances from script 1...")
    ladder_df = pd.read_csv(LADDER_CSV, index_col=0)
    ladder_df = ladder_df[ladder_df["physical_length_m"].notna()]
    print(f"  {len(ladder_df)} ladder instances loaded")
 
    image_paths = sorted(Path(BACKGROUNDS_DIR).glob("*.*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {BACKGROUNDS_DIR}")
    print(f"Found {len(image_paths)} background images\n")
 
    rng = np.random.default_rng(42)
    records = []
 
    for img_path in tqdm(image_paths, desc="Processing"):
        image_name = img_path.stem
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_bgr.shape[:2]
 
        depth_map = estimate_depth(depth_model, image_bgr)
        seg_map, confidence_map = segment_image(seg_processor, seg_model, image_rgb, device)
        print(f"classes segmentation{seg_map.max()}")
 
        surface_mask = get_surface_mask(seg_map, VALID_SURFACE_CLASSES, img_h, MAX_WALL_HEIGHT_FRAC)
        ground_mask  = get_ground_mask(seg_map, GROUND_CLASSES)
        surface_mask_pre = surface_mask.copy()

        ground_pixels = depth_map[ground_mask > 0] # exclude far away walls
        if len(ground_pixels) > 0:
            depth_threshold = float(np.percentile(ground_pixels, 5))
            surface_mask[depth_map < depth_threshold] = 0

        save_intermediates(seg_map, surface_mask_pre, surface_mask, ground_mask, intermediates_dir, image_name)
 
        kernel = np.ones((5, 5), np.uint8)
        surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_OPEN,  kernel)
        surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_CLOSE, kernel)
 
        wall_regions = get_valid_wall_regions(surface_mask, img_h * img_w)
        if not wall_regions:
            print(f"  [SKIP] No valid surfaces found in {image_name}")
            save_verification_row(image_bgr, depth_map, seg_map, [], out_dir / f"{image_name}_no_surface.png", image_name, id2label)
            continue

        combined_region_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for region_mask, _, _ in wall_regions:
            combined_region_mask = np.maximum(combined_region_mask, region_mask)
        region_mask_path = mask_dir / f"{image_name}_mask.png"
        cv2.imwrite(str(region_mask_path), combined_region_mask * 255)
 
        placements    = []
        placement_idx = 0
        shuffled_ladders = ladder_df.sample(frac=1, random_state=int(rng.integers(0, 9999))).reset_index(drop=True)

        for _, ladder in shuffled_ladders.iterrows():
            if placement_idx >= PLACEMENTS_PER_IMAGE:
                break
            physical_length_m = float(ladder["physical_length_m"])

            for region_mask, region_area, _ in wall_regions:    
                # Estimate depth at centre of this wall region
                wall_ys, wall_xs = np.where(region_mask > 0)
                zone_depth = float(depth_map[
                    int(np.clip(wall_ys.mean(), 0, img_w - 1)),
                    int(np.clip(wall_xs.mean(), 0, img_w - 1))
                ])

                box_h = physical_length_to_pixels(physical_length_m, seg_map, depth_map, zone_depth, confidence_map)
                if box_h is None:
                    continue
                box_w = int(box_h * float(ladder["mask_w"]) / max(float(ladder["mask_h"]), 1)) 

                result = find_placement_zone(
                    region_mask, ground_mask, depth_map,
                    box_h, box_w, img_h, img_w, rng
                )
                if result is None:
                    continue
    
                x1, y1, x2, y2_left, y2_right, depth_mean = result
                # Just store max y2 for simple crop boundaries in downstream scripts,
                # but keep the precise anchors if you want to apply shearing/warping later.
                y2 = max(y2_left, y2_right)
                placements.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "y2_left": y2_left, "y2_right": y2_right, "box_h" : box_h})
                records.append({
                    "background_image":  str(img_path),
                    "background_name":   image_name,
                    "placement_idx":     placement_idx,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "y2_left": y2_left, "y2_right": y2_right,
                    "zone_depth_mean":   round(depth_mean, 4),
                    "zone_w":            x2 - x1,
                    "zone_h":            y2 - y1,
                    "ladder_instance":   ladder["instance_id"],
                    "physical_length_m": round(physical_length_m, 3)
                })
                placement_idx += 1
                break
        if placements:
            save_verification_row(
                image_bgr, depth_map, seg_map,
                placements, verify_dir / f"{image_name}.jpg", image_name, id2label
            )
 
    if not records:
        print("No placement zones found.")
        return
 
    df = pd.DataFrame(records)
    df.to_csv(str(out_dir / "placements.csv"))
    print(f"\nFound {len(records)} placement zones across {len(image_paths)} images.")
    print(f"Verification : {verify_dir}")
    print(f"Intermediates: {intermediates_dir}")
    print(f"CSV          : {out_dir / 'placements_new.csv'}")
 
 
if __name__ == "__main__":
    main()