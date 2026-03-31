import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
import warnings
from PIL import Image as PILImage
import sys

warnings.filterwarnings("ignore")

IMAGES_DIR      = "./robo_images/ladder"
LABELS_DIR      = "./robo_images/yolo_annotations/labels"
OUTPUT_DIR      = "./MoBI_outputs/ladder_segmentations"
SAM2_REPO       = "./sam2"
SAM2_CHECKPOINT = "./sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEPTH_ANYTHING_REPO  = "./Depth-Anything-V2"
DEPTH_CHECKPOINT     = "./Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth"

_sam2_abs  = os.path.abspath(SAM2_REPO)
_depth_abs = os.path.abspath(DEPTH_ANYTHING_REPO)
 
# Add both repos to sys.path so their packages are importable
# without needing to chdir into them
sys.path.insert(0, _sam2_abs)
sys.path.insert(0, _depth_abs)
 
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from depth_anything_v2.dpt import DepthAnythingV2
 


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_yolo_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if int(parts[0]) != 0:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int(max(0, (cx - w / 2) * img_w))
            y1 = int(max(0, (cy - h / 2) * img_h))
            x2 = int(min(img_w, (cx + w / 2) * img_w))
            y2 = int(min(img_h, (cy + h / 2) * img_h))
            boxes.append([x1, y1, x2, y2])
    return boxes


def load_sam2(checkpoint, config, device):
    model = build_sam2(config, checkpoint, device=device)
    return SAM2ImagePredictor(model)


def get_mask_from_box(predictor, image_rgb, box_xyxy):
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(box_xyxy)[None, :],
        multimask_output=True,
    )
    best = np.argmax(scores)
    return masks[best].astype(np.uint8), float(scores[best])


def load_depth_model(checkpoint, device):
    model = DepthAnythingV2(encoder="vitb", features=128, out_channels=[96, 192, 384, 768])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device).eval()
    return model


def estimate_depth(model, image_rgb):
    # infer_image expects BGR numpy array
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    depth = model.infer_image(image_bgr)  # HxW raw relative depth
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    return depth.astype(np.float32)


def get_camera_intrinsics(img_w, img_h):
    fx = float(img_w)
    return np.array([
        [fx,  0, img_w / 2.0],
        [ 0, fx, img_h / 2.0],
        [ 0,  0, 1.0],
    ], dtype=np.float64)


def mask_to_3d_box(mask, depth_map, K):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None
 
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
 
    # PCA on mask pixels to find principal axis (long axis of ladder)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    center = pts.mean(axis=0)
    pts_centered = pts - center
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis  = eigvecs[:, 1]  # largest eigenvalue = ladder length direction
    short_axis = eigvecs[:, 0]  # smallest eigenvalue = ladder width direction
 
    # Project mask pixels onto principal axes to get extents
    proj_long  = pts_centered @ long_axis
    proj_short = pts_centered @ short_axis
    long_min,  long_max  = proj_long.min(),  proj_long.max()
    short_min, short_max = proj_short.min(), proj_short.max()
 
    # Sample depth along the long axis in bins to capture the lean gradient
    n_bins = 10
    bin_edges = np.linspace(long_min, long_max, n_bins + 1)
    bin_depths = []
    for i in range(n_bins):
        in_bin = (proj_long >= bin_edges[i]) & (proj_long < bin_edges[i + 1])
        if in_bin.sum() > 0:
            bin_px = pts[in_bin].astype(int)
            bin_depths.append(depth_map[bin_px[:, 1], bin_px[:, 0]].mean())
        else:
            bin_depths.append(np.nan)
    bin_depths = np.array(bin_depths)
    valid = ~np.isnan(bin_depths)
 
    # Ladder thickness from depth std across short axis — ladders are thin
    short_depth_std = float(depth_map[mask > 0].std())
    thickness = float(np.clip(short_depth_std * 0.5, 0.005, 0.05))
 
    # 4 corners of the oriented rectangle in image space
    corners_2d = np.array([
        long_axis * long_min  + short_axis * short_min,
        long_axis * long_min  + short_axis * short_max,
        long_axis * long_max  + short_axis * short_max,
        long_axis * long_max  + short_axis * short_min,
    ]) + center
 
    # Interpolate depth at each corner from the bin profile
    corner_proj = (corners_2d - center) @ long_axis
    mean_depth = float(np.nanmean(bin_depths))
    bin_centers = np.linspace(long_min, long_max, n_bins)
    bin_vals = np.where(valid, bin_depths, mean_depth)
    corner_depths = np.interp(corner_proj, bin_centers, bin_vals)
 
    # Back-project to 3D for front face and back face (thickness offset)
    corners_3d = []
    for face_offset in [0.0, thickness]:
        for i, (px, py) in enumerate(corners_2d):
            z = max(float(corner_depths[i]) + face_offset, 1e-4)
            X = (px - K[0, 2]) * z / K[0, 0]
            Y = (py - K[1, 2]) * z / K[1, 1]
            corners_3d.append([X, Y, z])
 
    return np.array(corners_3d, dtype=np.float32), [x1, y1, x2, y2]



def draw_3d_box(image_rgb, corners_3d, K):
    vis = image_rgb.copy()
    pts = []
    for X, Y, Z in corners_3d:
        if Z <= 0:
            pts.append(None)
            continue
        pts.append((int(K[0,0]*X/Z + K[0,2]), int(K[1,1]*Y/Z + K[1,2])))
    for i, j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        if pts[i] is not None and pts[j] is not None:
            cv2.line(vis, pts[i], pts[j], (0, 255, 0), 2)
    return vis


def save_verification_row(image_rgb, mask, depth_map, corners_3d, box_xyxy, K, instance_id, score, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{instance_id}  —  SAM2: {score:.3f}", fontsize=10, color="#ccc")
    fig.patch.set_facecolor("#111")

    x1, y1, x2, y2 = box_xyxy

    axes[0].imshow(image_rgb)
    axes[0].add_patch(patches.Rectangle(
        (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="yellow", facecolor="none"
    ))
    axes[0].set_title("Original + YOLO box", color="#aaa", fontsize=9)

    overlay = image_rgb.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 200, 100]) * 0.5).astype(np.uint8)
    axes[1].imshow(overlay)
    axes[1].set_title("SAM2 mask", color="#aaa", fontsize=9)

    axes[2].imshow(depth_map, cmap="plasma")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt = cnt.squeeze()
        if cnt.ndim == 2:
            axes[2].plot(cnt[:, 0], cnt[:, 1], "w-", linewidth=1.5)
    axes[2].set_title("Depth + mask contour", color="#aaa", fontsize=9)

    axes[3].imshow(draw_3d_box(image_rgb, corners_3d, K))
    axes[3].set_title("Projected 3D box", color="#aaa", fontsize=9)

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#111")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight", facecolor="#111")
    plt.close(fig)


def main():
    device = get_device()
    print(f"Device: {device}")

    out_dir     = Path(OUTPUT_DIR)
    corners_dir = out_dir / "corners_3d"
    masks_dir   = out_dir / "masks"
    verify_dir  = out_dir / "verification"
    for d in [corners_dir, masks_dir, verify_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("Loading SAM2...")
    sam_predictor = load_sam2(SAM2_CHECKPOINT, SAM2_CONFIG, device)
    print("Loading Depth Anything V2...")
    depth_model = load_depth_model(DEPTH_CHECKPOINT, device)

    image_paths = sorted(Path(IMAGES_DIR).glob("*.*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}")
    print(f"Found {len(image_paths)} images\n")

    records = []

    for img_path in tqdm(image_paths, desc="Processing"):
        image_name = img_path.stem
        label_path = Path(LABELS_DIR) / f"{image_name}.txt"

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_rgb.shape[:2]

        K     = get_camera_intrinsics(img_w, img_h)
        boxes = load_yolo_boxes(str(label_path), img_w, img_h)
        if not boxes:
            continue

        depth_map = estimate_depth(depth_model, image_rgb)

        for box_idx, box_xyxy in enumerate(boxes):
            instance_id = f"{image_name}_ladder{box_idx:02d}"

            mask, score   = get_mask_from_box(sam_predictor, image_rgb, box_xyxy)
            corners_3d, _ = mask_to_3d_box(mask, depth_map, K)
            if corners_3d is None:
                continue

            np.save(str(corners_dir / f"{instance_id}_corners3d.npy"), corners_3d)
            np.save(str(masks_dir   / f"{instance_id}_mask.npy"), mask)

            save_verification_row(
                image_rgb, mask, depth_map, corners_3d, box_xyxy, K,
                instance_id, score, verify_dir / f"{instance_id}.jpg"
            )

            ys, xs = np.where(mask > 0)
            records.append({
                "instance_id":      instance_id,
                "image_path":       str(img_path),
                "sam2_score":       round(score, 4),
                "mask_h":           int(ys.max() - ys.min()),
                "mask_w":           int(xs.max() - xs.min()),
                "mask_pixel_count": int((mask > 0).sum()),
                "depth_min":        round(float(depth_map[mask > 0].min()), 4),
                "depth_max":        round(float(depth_map[mask > 0].max()), 4),
                "depth_mean":       round(float(depth_map[mask > 0].mean()), 4),
                "corners3d_path":   str(corners_dir / f"{instance_id}_corners3d.npy"),
                "mask_path":        str(masks_dir   / f"{instance_id}_mask.npy"),
            })

    if not records:
        print("No instances processed.")
        return

    df = pd.DataFrame(records)
    df.to_csv(str(out_dir / "ladder_instances.csv"))
    print(f"\nProcessed {len(records)} instances.")
    print(f"Verification images : {verify_dir}")
    print(f"CSV                 : {out_dir / 'ladder_instances.csv'}")


if __name__ == "__main__":
    main()