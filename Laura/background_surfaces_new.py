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

BACKGROUNDS_DIR      = "./robo_images/no_ladder"
OUTPUT_DIR           = "./MoBI_outputs/background10_new"
DEPTH_ANYTHING_REPO  = "./Depth-Anything-V2"
DEPTH_CHECKPOINT     = "./Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth"
SEGFORMER_MODEL      = "nvidia/segformer-b2-finetuned-ade-512-512"

PLACEMENTS_PER_IMAGE  = 3
MIN_REGION_AREA_FRAC  = 0.02
MAX_WALL_HEIGHT_FRAC  = 0.75   # only consider bottom N% of image (reachable from ground)

MIN_ZONE_WIDTH_FRAC = 0.08     # discard zones narrower than this

# ── ADE20K class sets ───────────────────────────────────────────────────────
# wall, building, house, fence, railing, column/pillar, skyscraper, hovel/hut, tower
VALID_SURFACE_CLASSES = {0, 1, 25, 32, 38, 42, 48, 79, 84}
# road, grass, sidewalk/pavement, earth/ground, path, dirt track, land/soil
GROUND_CLASSES = {3, 6, 9, 11, 13, 52, 91, 94}

sys.path.insert(0, os.path.abspath(DEPTH_ANYTHING_REPO))
from depth_anything_v2.dpt import DepthAnythingV2


# ── Model helpers ────────────────────────────────────────────────────────────

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
        outputs.logits, size=image_rgb.shape[:2],
        mode="bilinear", align_corners=False,
    )
    seg_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16)
    return seg_map


# ── Mask helpers ─────────────────────────────────────────────────────────────

def get_surface_mask(seg_map, valid_classes, img_h, max_height_frac):
    mask = np.zeros(seg_map.shape, dtype=np.uint8)
    for cls in valid_classes:
        mask[seg_map == cls] = 1
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
    for i in range(1, n + 1):
        region = (labeled == i).astype(np.uint8)
        area   = int(region.sum())
        if area < MIN_REGION_AREA_FRAC * img_area:
            continue
        ys, xs = np.where(region)
        regions.append((region, area, (float(xs.mean()), float(ys.mean()))))
    regions.sort(key=lambda r: r[1], reverse=True)
    return regions[:10]


# ── Zone finder — geometry only, no ladder scaling ───────────────────────────

def find_placement_zone(wall_region, ground_mask, depth_map, img_h, img_w, rng):

    min_zone_w = int(img_w * MIN_ZONE_WIDTH_FRAC)

    wall_ys, wall_xs = np.where(wall_region > 0)
    if len(wall_xs) == 0:
        return None

    candidates = []
    for _ in range(600):
        idx = rng.integers(0, len(wall_xs))
        cx  = int(wall_xs[idx])

        wall_xs_in_region = wall_xs  # full horizontal extent of this wall region
        x1 = max(0, int(wall_xs_in_region.min()))
        x2 = min(img_w - 1, int(wall_xs_in_region.max()))
        if (x2 - x1) < min_zone_w:
            continue

        # ── Build ground boundary inside this column slice ──────────────────
        search_roi    = ground_mask[:, x1:x2]
        ground_top_ys = np.argmax(search_roi > 0, axis=0)
        valid_cols    = (search_roi[ground_top_ys, np.arange(x2 - x1)] > 0)

        if valid_cols.sum() < (x2 - x1) * 0.4:
            continue

        xs_valid = np.arange(x1, x2)[valid_cols]
        ys_valid = ground_top_ys[valid_cols] + 8   # sink slightly into ground

        all_cols       = np.arange(x1, x2)
        ground_ys_full = np.clip(
            np.interp(all_cols, xs_valid, ys_valid).astype(int), 0, img_h - 1
        )

        y2_left  = int(ground_ys_full[0])
        y2_right = int(ground_ys_full[-1])
        y2_min   = int(ground_ys_full.min())
        y1 = max(0, int(wall_ys.min()))  # top of wall region, for visualisation only          # generous upward extent

        # Need enough wall coverage in the proposed box
        if wall_region[y1:y2_min, x1:x2].mean() < 0.30:
            continue

        box_depth  = depth_map[y1:y2_min, x1:x2]
        depth_std  = float(box_depth.std())
        depth_mean = float(box_depth.mean())

        candidates.append((
            depth_std, x1, y1, x2,
            y2_left, y2_right, depth_mean,
            all_cols, ground_ys_full
        ))

    if not candidates:
        return None

    # Prefer zones with low depth variance (flat wall, consistent distance)
    candidates.sort(key=lambda c: c[0])
    _, x1, y1, x2, y2_l, y2_r, depth_mean, ground_xs, ground_ys = candidates[0]
    return x1, y1, x2, y2_l, y2_r, depth_mean, ground_xs, ground_ys


# ── Visualisation helpers ────────────────────────────────────────────────────

def colorize_seg_map(seg_map):
    h, w = seg_map.shape
    vis  = np.ones((h, w, 3), dtype=np.uint8) * 40
    for cls in VALID_SURFACE_CLASSES:
        vis[seg_map == cls] = [50, 200, 80]
    for cls in GROUND_CLASSES:
        vis[seg_map == cls] = [200, 160, 50]
    return vis


def save_verification_row(image_bgr, depth_map, seg_map, placements, out_path, image_name):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(image_name, fontsize=10, color="#ccc")
    fig.patch.set_facecolor("#111")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original", color="#aaa", fontsize=9)

    axes[1].imshow(depth_map, cmap="plasma")
    axes[1].set_title("Depth", color="#aaa", fontsize=9)

    axes[2].imshow(colorize_seg_map(seg_map))
    axes[2].set_title("Wall (green) / Ground (orange)", color="#aaa", fontsize=9)

    axes[3].imshow(image_rgb)
    colors = ["red", "cyan", "yellow"]
    for i, p in enumerate(placements):
        c = colors[i % len(colors)]
        x1, y1, x2 = p["x1"], p["y1"], p["x2"]
        y2_l, y2_r = p["y2_left"], p["y2_right"]

        # Draw zone outline (trapezoid following ground slope)
        poly = np.array([[x1, y1], [x2, y1], [x2, y2_r], [x1, y2_l]], dtype=np.int32)
        axes[3].add_patch(patches.Polygon(
            poly, closed=True, linewidth=2, edgecolor=c, facecolor="none"
        ))
        # Draw ground boundary curve
        gxs, gys = p["ground_xs"], p["ground_ys"]
        axes[3].plot(gxs, gys, '-', color=c, lw=2)
        axes[3].text(x1, max(y1 - 6, 0), f"#{i}", color=c, fontsize=10)

    axes[3].set_title("Placement zones + ground boundary", color="#aaa", fontsize=9)

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#111")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Device: {device}")

    out_dir           = Path(OUTPUT_DIR)
    verify_dir        = out_dir / "verification"
    intermediates_dir = out_dir / "intermediates"
    mask_dir          = out_dir / "PbE_masks"
    for d in [verify_dir, intermediates_dir, mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("Loading Depth Anything V2...")
    depth_model = load_depth_model(DEPTH_CHECKPOINT, device)

    print("Loading SegFormer...")
    seg_processor, seg_model = load_segformer(SEGFORMER_MODEL, device)

    image_paths = sorted(Path(BACKGROUNDS_DIR).glob("*.*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {BACKGROUNDS_DIR}")
    print(f"Found {len(image_paths)} background images\n")

    rng     = np.random.default_rng(42)
    records = []

    for img_path in tqdm(image_paths, desc="Processing"):
        image_name = img_path.stem
        image_bgr  = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_bgr.shape[:2]

        depth_map = estimate_depth(depth_model, image_bgr)
        seg_map   = segment_image(seg_processor, seg_model, image_rgb, device)

        surface_mask = get_surface_mask(seg_map, VALID_SURFACE_CLASSES, img_h, MAX_WALL_HEIGHT_FRAC)
        ground_mask  = get_ground_mask(seg_map, GROUND_CLASSES)

        # Remove wall pixels that are farther away than the nearest ground (avoids distant walls)
        ground_pixels = depth_map[ground_mask > 0]
        if len(ground_pixels) > 0:
            depth_threshold = float(np.percentile(ground_pixels, 5))
            surface_mask[depth_map < depth_threshold] = 0

        kernel       = np.ones((5, 5), np.uint8)
        surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_OPEN,  kernel)
        surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_CLOSE, kernel)

        wall_regions = get_valid_wall_regions(surface_mask, img_h * img_w)
        if not wall_regions:
            print(f"  [SKIP] No valid surfaces in {image_name}")
            continue

        # Save combined PbE mask (union of all wall regions)
        combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for region_mask, _, _ in wall_regions:
            combined_mask = np.maximum(combined_mask, region_mask)
        cv2.imwrite(str(mask_dir / f"{image_name}_mask.png"), combined_mask * 255)

        # Save seg visualisation
        cv2.imwrite(
            str(intermediates_dir / f"{image_name}_seg.png"),
            cv2.cvtColor(colorize_seg_map(seg_map), cv2.COLOR_RGB2BGR)
        )

        placements    = []
        placement_idx = 0

        for region_mask, region_area, _ in wall_regions:
            if placement_idx >= PLACEMENTS_PER_IMAGE:
                break

            result = find_placement_zone(
                region_mask, ground_mask, depth_map,
                img_h, img_w, rng
            )
            if result is None:
                continue

            x1, y1, x2, y2_left, y2_right, depth_mean, ground_xs, ground_ys = result
            y2 = max(y2_left, y2_right)

            placements.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "y2_left": y2_left, "y2_right": y2_right,
                "ground_xs": ground_xs, "ground_ys": ground_ys,
            })
            records.append({
                "background_image":  str(img_path),
                "background_name":   image_name,
                "placement_idx":     placement_idx,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "y2_left":           y2_left,
                "y2_right":          y2_right,
                # Ground boundary encoded as "x,y;x,y;..." for easy re-parsing
                "ground_boundary":   ";".join(f"{x},{y}" for x, y in zip(ground_xs, ground_ys)),
                "zone_depth_mean":   round(depth_mean, 4),
                "zone_w":            x2 - x1,
                "zone_h":            y2 - y1,
            })
            placement_idx += 1

        if placements:
            save_verification_row(
                image_bgr, depth_map, seg_map,
                placements, verify_dir / f"{image_name}.jpg", image_name
            )

    if not records:
        print("No placement zones found.")
        return

    df = pd.DataFrame(records)
    out_csv = out_dir / "placements.csv"
    df.to_csv(str(out_csv), index=False)
    print(f"\nFound {len(records)} zones across {len(image_paths)} images.")
    print(f"CSV          : {out_csv}")
    print(f"Verification : {verify_dir}")


if __name__ == "__main__":
    main()