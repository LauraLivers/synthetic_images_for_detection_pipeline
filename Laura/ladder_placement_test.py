print("STARTING", flush=True)
import subprocess, os, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if not os.path.exists("Depth-Anything-V2/depth_anything_v2_vitb.pth"):
    from huggingface_hub import hf_hub_download
    hf_hub_download("depth-anything/Depth-Anything-V2-Base", "depth_anything_v2_vitb.pth", local_dir="Depth-Anything-V2")
if not os.path.exists("sam2/sam2"):
    subprocess.run(["git", "clone", "https://github.com/facebookresearch/sam2", "sam2_code"], capture_output=True)
    os.rename("sam2_code/sam2", "sam2/sam2")

sys.path.insert(0, "Depth-Anything-V2")
sys.path.insert(0, "sam2")

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image as PILImage
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BACKGROUNDS_DIR      = "./robo_images/no_ladder"
LADDER_DIR           = "./robo_images/ladder"
PLACEMENTS_CSV       = "./MoBI_outputs/background10_new/placements.csv"
LADDER_INSTANCES_CSV = "./MoBI_outputs/ladder_segmentations_real_size5_FILTERED/ladder_instances.csv"
OUTPUT_DIR           = "./MoBI_outputs/batch_warp_v2"

DEPTH_ANYTHING_REPO  = "./Depth-Anything-V2"
DEPTH_CHECKPOINT     = "./Depth-Anything-V2/depth_anything_v2_vitb.pth"
SEGFORMER_MODEL      = "nvidia/segformer-b2-finetuned-ade-512-512"

# ── Physical ladder height (metres) ─────────────────────────────────────────
LADDER_HEIGHT_M = 2.0

# ── ADE20K reference objects
SCALE_REFERENCE_CLASSES = {
    12: 1.7,   # person
    14: 2.1,   # door
     8: 1.5,   # windowpane
    20: 1.5,   # car
    87: 4.0,   # streetlight
    93: 0.9,   # pole (short)
   102: 2.2,   # van
   127: 1.2,   # bicycle
}

REF_MIN_PIXELS     = 150
REF_MIN_CONFIDENCE = 0.75
REF_MAX_DEPTH_RATIO = 3.0

# Fallback: if no reference object found, ladder height = this fraction of image height
FALLBACK_HEIGHT_FRAC = 0.35

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
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
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
    probs      = torch.nn.functional.softmax(upsampled, dim=1)
    conf_map   = probs.max(dim=1).values.squeeze(0).cpu().numpy()
    seg_map    = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16)
    return seg_map, conf_map


# ── Scale estimation ─────────────────────────────────────────────────────────

def estimate_ladder_pixel_height(
    physical_height_m, seg_map, depth_map, conf_map,
    zone_depth_mean, img_h
):
    if zone_depth_mean < 1e-4:
        return None
    estimates = []
    for cls, ref_h_m in SCALE_REFERENCE_CLASSES.items():
        cls_ys, cls_xs = np.where(seg_map == cls)
        if len(cls_ys) < REF_MIN_PIXELS:
            continue
        high_conf = conf_map[cls_ys, cls_xs] > REF_MIN_CONFIDENCE
        cls_ys, cls_xs = cls_ys[high_conf], cls_xs[high_conf]
        if len(cls_ys) < REF_MIN_PIXELS // 2:
            print(f"    [REJECT cls={cls}] low confidence pixels: {len(cls_ys)}")
            continue
        ref_disparity = float(np.median(depth_map[cls_ys, cls_xs]))
        depth_ratio = max(ref_disparity, zone_depth_mean) / max(min(ref_disparity, zone_depth_mean), 1e-6)
        if depth_ratio > REF_MAX_DEPTH_RATIO:
            print(f"    [REJECT cls={cls}] depth_ratio={depth_ratio:.2f} ref={ref_disparity:.3f} zone={zone_depth_mean:.3f}")
            continue
        ref_px_height = float(cls_ys.max() - cls_ys.min())
        if ref_px_height < 20:
            print(f"    [REJECT cls={cls}] too short: {ref_px_height}px")
            continue
        est = ref_px_height * (physical_height_m / ref_h_m) * (zone_depth_mean / ref_disparity)
        if not np.isfinite(est):
            print(f"    [REJECT cls={cls}] NaN estimate")
            continue
        print(f"    [ACCEPT cls={cls}] est={est:.0f}px")
        estimates.append(int(round(est)))
    if not estimates:
        return None
    return int(np.median(estimates))


def get_pca_corners(mask_2d):
    """
    Return the 4 corners of the ladder mask via PCA, ordered:
      [top-left, top-right, bottom-right, bottom-left]
    """
    ys, xs = np.where(mask_2d)
    pts    = np.stack([xs, ys], axis=1).astype(np.float64)
    center = pts.mean(axis=0)
    pts_c  = pts - center

    _, eigvecs = np.linalg.eigh(np.cov(pts_c.T))
    long_axis  = eigvecs[:, 1]
    short_axis = eigvecs[:, 0]

    proj_long  = pts_c @ long_axis
    proj_short = pts_c @ short_axis

    idx_tl = np.argmin(proj_long + proj_short)
    idx_tr = np.argmin(proj_long - proj_short)
    idx_br = np.argmax(proj_long + proj_short)
    idx_bl = np.argmax(proj_long - proj_short)

    return np.array([
        pts[idx_tl], pts[idx_tr], pts[idx_br], pts[idx_bl]
    ], dtype=np.float32)


def bottom_edge_corners(corners):
    sorted_by_y = corners[corners[:, 1].argsort()]
    bl, br      = sorted_by_y[2], sorted_by_y[3]
    if bl[0] > br[0]:
        bl, br = br, bl
    return bl, br


def warp_ladder_to_ground(
    ladder_rgb, ladder_mask,
    src_corners,
    ground_xs, ground_ys,
    x1, x2, canvas_W, canvas_H
):
    src_bl, src_br = bottom_edge_corners(src_corners)

    dst_bl = np.array([x1,  ground_ys[0]],  dtype=np.float64)
    dst_br = np.array([x2,  ground_ys[-1]], dtype=np.float64)

    src_vec   = src_br - src_bl
    dst_vec   = dst_br - dst_bl
    src_angle = np.arctan2(src_vec[1], src_vec[0])
    dst_angle = np.arctan2(dst_vec[1], dst_vec[0])
    angle     = dst_angle - src_angle
    #scale     = np.linalg.norm(dst_vec) / (np.linalg.norm(src_vec) + 1e-6)
    scale = 1.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = scale * np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    dst_corners = ((src_corners - src_bl) @ R.T) + dst_bl
    src_pts     = src_corners.astype(np.float32)
    dst_pts     = dst_corners.astype(np.float32)

    M, _ = cv2.findHomography(src_pts, dst_pts)
    if M is None:
        raise RuntimeError("findHomography failed — degenerate corners?")

    warp_kwargs = dict(
        dsize=(canvas_W, canvas_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_rgb  = cv2.warpPerspective(ladder_rgb,              M, **warp_kwargs)
    warped_mask = cv2.warpPerspective(ladder_mask.astype(np.uint8), M, **warp_kwargs).astype(bool)

    return warped_rgb, warped_mask


def composite(background_rgb, ladder_rgb, ladder_mask):
    out = background_rgb.copy()
    out[ladder_mask] = ladder_rgb[ladder_mask]
    return out


def parse_ground_boundary(boundary_str):
    pairs  = [p.split(",") for p in boundary_str.split(";")]
    xs     = np.array([int(p[0]) for p in pairs])
    ys     = np.array([int(p[1]) for p in pairs])
    return xs, ys


def sample_ground_boundary_at(ground_xs, ground_ys, x1, x2):
    # Restrict to the zone columns
    mask      = (ground_xs >= x1) & (ground_xs <= x2)
    xs_zone   = ground_xs[mask]
    ys_zone   = ground_ys[mask]

    if len(xs_zone) < 2:
        # Flat fallback
        y_mid = int(np.interp([(x1 + x2) / 2], ground_xs, ground_ys)[0])
        xs_zone = np.array([x1, x2])
        ys_zone = np.array([y_mid, y_mid])

    y_at_x1 = int(np.interp(x1, xs_zone, ys_zone))
    y_at_x2 = int(np.interp(x2, xs_zone, ys_zone))
    return xs_zone, ys_zone, y_at_x1, y_at_x2


def main():
    import random
    random.seed(0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    print("Loading Depth Anything V2...")
    depth_model = load_depth_model(DEPTH_CHECKPOINT, device)

    print("Loading SegFormer...")
    seg_processor, seg_model = load_segformer(SEGFORMER_MODEL, device)

    df_placements = pd.read_csv(PLACEMENTS_CSV)
    df_ladders    = pd.read_csv(LADDER_INSTANCES_CSV)

    ladder_images = sorted(Path(LADDER_DIR).glob("*.jpg"))
    if not ladder_images:
        raise FileNotFoundError(f"No ladder images in {LADDER_DIR}")

    colors = ["red", "cyan", "yellow"]

    # Cache depth+seg per background to avoid re-running models
    bg_cache = {}

    for ladder_path in ladder_images:
        ladder_bgr = cv2.imread(str(ladder_path))
        ladder_rgb = cv2.cvtColor(ladder_bgr, cv2.COLOR_BGR2RGB)

        ladder_name = ladder_path.stem
        row_matches = df_ladders[df_ladders["image_path"].str.endswith(ladder_path.name)]
        if row_matches.empty:
            print(f"  [SKIP] No ladder instance found for {ladder_path.name}")
            continue
        ladder_row  = row_matches.iloc[0]
        ladder_mask_full = np.load(ladder_row["mask_path"]).astype(bool)

        ys, xs = np.where(ladder_mask_full)
        ty1, ty2 = ys.min(), ys.max() + 1
        tx1, tx2 = xs.min(), xs.max() + 1
        tight_rgb  = ladder_rgb[ty1:ty2, tx1:tx2]
        tight_mask = ladder_mask_full[ty1:ty2, tx1:tx2]
        src_corners = get_pca_corners(tight_mask)

        ladder_physical_h = float(ladder_row.get("physical_length_m", LADDER_HEIGHT_M))

        bg_names = df_placements["background_name"].unique().tolist()
        chosen   = random.sample(bg_names, min(3, len(bg_names)))

        for bg_name in chosen:
            bg_placements = df_placements[df_placements["background_name"] == bg_name]
            if bg_placements.empty:
                continue

            bg_row     = bg_placements.iloc[0]
            bg_path    = bg_row["background_image"]
            bg_bgr     = cv2.imread(bg_path)
            if bg_bgr is None:
                print(f"  [SKIP] Cannot read {bg_path}")
                continue
            bg_rgb     = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
            H, W       = bg_rgb.shape[:2]

            if bg_name not in bg_cache:
                print(f"  Running models on background: {bg_name}")
                depth_map = estimate_depth(depth_model, bg_bgr)
                seg_map, conf_map = segment_image(seg_processor, seg_model, bg_rgb, device)
                bg_cache[bg_name] = (depth_map, seg_map, conf_map)
            depth_map, seg_map, conf_map = bg_cache[bg_name]

            for pidx, (_, placement) in enumerate(bg_placements.iterrows()):
                x1   = int(placement["x1"])
                x2   = int(placement["x2"])
                y1   = int(placement["y1"])

                gxs_full, gys_full = parse_ground_boundary(placement["ground_boundary"])
                gxs, gys, y_at_x1, y_at_x2 = sample_ground_boundary_at(gxs_full, gys_full, x1, x2)

                #zone_depth_mean = float(placement["zone_depth_mean"])
                zone_depth_mean = float(np.median(depth_map[gys, gxs]))
                px_h = estimate_ladder_pixel_height(
                    ladder_physical_h, seg_map, depth_map, conf_map,
                    zone_depth_mean, H
                )
                if px_h is None:
                    px_h = tight_rgb.shape[0]
                    print(f"    [NO REF] Using original ladder height: {px_h}px")

                px_h = int(np.clip(px_h, int(H * 0.10), int(H * 0.85)))

                orig_h, orig_w = tight_rgb.shape[:2]
                scale_factor   = px_h / max(orig_h, 1)
                new_h          = px_h
                new_w          = max(int(orig_w * scale_factor), 1)

                scaled_rgb  = cv2.resize(tight_rgb,  (new_w, new_h), interpolation=cv2.INTER_AREA)
                scaled_mask = cv2.resize(
                    tight_mask.astype(np.uint8), (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                scaled_corners = get_pca_corners(scaled_mask)

                try:
                    warped_rgb, warped_mask = warp_ladder_to_ground(
                        scaled_rgb, scaled_mask,
                        scaled_corners,
                        gxs, gys,
                        x1, x2, W, H
                    )
                except RuntimeError as e:
                    print(f"    [SKIP placement {pidx}] {e}")
                    continue

                result = composite(bg_rgb, warped_rgb, warped_mask)

                fig, axes = plt.subplots(1, 3, figsize=(24, 8))
                fig.patch.set_facecolor("#111")

                # Panel 0: background with all zones + ground boundary curves
                axes[0].imshow(bg_rgb)
                for i, (_, row) in enumerate(bg_placements.iterrows()):
                    c   = colors[i % len(colors)]
                    rx1, rx2 = int(row["x1"]), int(row["x2"])
                    ry1      = int(row["y1"])
                    ry2_l    = int(row["y2_left"])
                    ry2_r    = int(row["y2_right"])
                    gxb, gyb = parse_ground_boundary(row["ground_boundary"])
                    axes[0].plot(gxb, gyb, '-', color=c, lw=2, label=f"#{i} ground")
                    poly = np.array([[rx1, ry1], [rx2, ry1], [rx2, ry2_r], [rx1, ry2_l]])
                    axes[0].add_patch(plt.Polygon(poly, closed=True, lw=1.5, edgecolor=c, facecolor="none"))
                axes[0].set_title("Placement zones", color="white")
                axes[0].legend(fontsize=7)

                # Panel 1: scaled source ladder with corners
                _, src_bl = bottom_edge_corners(scaled_corners)[0], bottom_edge_corners(scaled_corners)[1]
                src_bl_pt, src_br_pt = bottom_edge_corners(scaled_corners)
                axes[1].imshow(scaled_rgb)
                axes[1].scatter(scaled_corners[:, 0], scaled_corners[:, 1],
                                c='cyan', s=60, marker='x', label="PCA corners")
                axes[1].plot(
                    [src_bl_pt[0], src_br_pt[0]],
                    [src_bl_pt[1], src_br_pt[1]],
                    'r-', lw=3, label="Bottom edge"
                )
                axes[1].set_title(f"Scaled ladder ({new_h}px tall)", color="white")
                axes[1].legend(fontsize=7)

                # Panel 2: composite result
                axes[2].imshow(result)
                # Overlay ground boundary used for this placement
                axes[2].plot(gxs, gys, 'g-', lw=2, label="Ground boundary")
                axes[2].set_title(f"Placement #{pidx} — {px_h}px, {ladder_physical_h:.1f}m", color="white")
                axes[2].legend(fontsize=7)

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()
                out_name = f"{bg_name}__{ladder_name}__p{pidx}.jpg"
                plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=150)
                plt.close()
                print(f"  Saved {out_name}  (px_h={px_h})")


if __name__ == "__main__":
    main()
