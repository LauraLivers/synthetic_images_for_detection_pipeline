"""
Inference script: insert a reference ladder into a background image
using the trained Paint-by-Example model.

Usage:
    uv run python infer.py
"""
import sys
import os
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
import torchvision.transforms as T
import torchvision
import inspect
import random
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── paths ──────────────────────────────────────────────────────────────────
MOBI_ROOT      = os.path.abspath("../MobI")
MY_FOLDER      = os.path.abspath(".")
TAMING_ROOT    = os.path.abspath("../taming-transformers")

DEPTH_ROOT     = os.path.abspath("Depth-Anything-V2")
SAM2_ROOT      = os.path.abspath("sam2")
SEG_ROOT       = os.path.abspath("SegFormer")

for p in [MOBI_ROOT, MY_FOLDER, TAMING_ROOT, DEPTH_ROOT, SAM2_ROOT, SEG_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ladder_placement_test import (
    load_depth_model, load_segformer, estimate_depth, segment_image,
    estimate_ladder_pixel_height, parse_ground_boundary, sample_ground_boundary_at,
    get_pca_corners, bottom_edge_corners, LADDER_HEIGHT_M, DEPTH_CHECKPOINT, SEGFORMER_MODEL
)

BACKGROUNDS_DIR = "robo_images/no_ladder"
LADDER_DIR      = "robo_images/ladder"
PLACEMENTS_CSV = "MoBI_outputs/background10_new/placements.csv"
CHECKPOINT     = "../MoBI_outputs/mobi_inpainting2/2026-05-31T11-03-19_ladder_dataset_mobi/checkpoints/epoch=000001.ckpt"
CONFIG_PATH    = "ladder_dataset_mobi.yaml"
OUTPUT_DIR     = "MoBI_outputs/inpaint_results_lambda1.2"
LADDER_INSTANCES_CSV = "MoBI_outputs/ladder_segmentations_real_size5_FILTERED/ladder_instances.csv"

DDIM_STEPS     = 50
IMAGE_SIZE     = 512

# helpers 
def get_tensor(normalize=True):
    transforms = [torchvision.transforms.ToTensor()]
    if normalize:
        transforms.append(torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)))
    return torchvision.transforms.Compose(transforms)

def get_tensor_clip():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711))
    ])

def load_placement(csv_path, background_img):
    df = pd.read_csv(csv_path, index_col=0)
    name = os.path.splitext(os.path.basename(background_img))[0]
    rows = df[df["background_name"] == name]
    if len(rows) == 0:
        raise ValueError(f"No placement found for {name}")
    return rows.iloc[0]

def get_edge_length(corners):
    edges = []
    for i in range(4):
        a = corners[i]
        b = corners[(i+1) % 4]
        edges.append((np.linalg.norm(b - a), i, (i + 1) % 4))
    return edges

def get_pca_corners(mask_tight):
    ys_t, xs_t = np.where(mask_tight)
    pts = np.stack([xs_t, ys_t], axis=1).astype(np.float64)
    center = pts.mean()
    pts_c = pts - center
    _, eigvecs = np.linalg.eigh(np.cov(pts_c.T))
    long_axis = eigvecs[:, 1]
    short_axis = eigvecs[:, 0]
    proj_long = pts_c @ long_axis
    proj_short = pts_c @ short_axis
    idx_tl = np.argmin(proj_long + proj_short)
    idx_tr = np.argmin(proj_long - proj_short)
    idx_br = np.argmax(proj_long + proj_short)
    idx_bl = np.argmax(proj_long - proj_short)

    return np.array([
        pts[idx_tl], pts[idx_tr], pts[idx_br], pts[idx_bl]
    ], dtype=np.float32)

def warp_ladder_local(scaled_rgb, scaled_mask, src_corners, x1, x2, y_at_x1, y_at_x2, crop_x1, crop_y1, crop_W, crop_H):
    """
    Uses the scale=1.0 and PCA bottom edge rotation from ladder_placement_test
    but targets the local crop coordinate space for the inpainting pipeline.
    """
    src_bl, src_br = bottom_edge_corners(src_corners)

    dst_bl = np.array([x1 - crop_x1, y_at_x1 - crop_y1], dtype=np.float64)
    dst_br = np.array([x2 - crop_x1, y_at_x2 - crop_y1], dtype=np.float64)

    src_vec   = src_br - src_bl
    dst_vec   = dst_br - dst_bl
    src_angle = np.arctan2(src_vec[1], src_vec[0])
    dst_angle = np.arctan2(dst_vec[1], dst_vec[0])
    angle     = dst_angle - src_angle
    scale     = 1.0 # Force scale 1.0 so image is not stretched exactly as designed

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = scale * np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    dst_corners = ((src_corners - src_bl) @ R.T) + dst_bl
    src_pts     = src_corners.astype(np.float32)
    dst_pts     = dst_corners.astype(np.float32)

    M, _ = cv2.findHomography(src_pts, dst_pts)
    if M is None:
        raise RuntimeError("no Homography found")
    
    warp_kwargs = dict(
        dsize=(crop_W, crop_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_rgb  = cv2.warpPerspective(scaled_rgb, M, **warp_kwargs)
    warped_mask = cv2.warpPerspective(scaled_mask.astype(np.uint8), M, **warp_kwargs).astype(bool)
    
    return warped_rgb, warped_mask


def main():
       
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Device: {device}")

    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(CONFIG_PATH)

    print("Loading Depth Anything V2 & Segformer...")
    depth_model = load_depth_model(DEPTH_CHECKPOINT, device)
    seg_processor, seg_model = load_segformer(SEGFORMER_MODEL, device)

    print("Loading model...")
    model = instantiate_from_config(config.model)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device).eval()
    print("Model loaded.")

    df_placements = pd.read_csv(PLACEMENTS_CSV)
    instances = pd.read_csv(LADDER_INSTANCES_CSV)
    
    ladder_images = sorted(list(Path(LADDER_DIR).glob("*.jpg")))
    bg_names = df_placements["background_name"].unique().tolist()
    bg_cache = {}

    for ladder_path in ladder_images:
        ladder_name = ladder_path.stem
        row_matches = instances[instances["image_path"].str.endswith(ladder_path.name)]
        if row_matches.empty:
            print(f"  [SKIP] No ladder instance found for {ladder_path.name}")
            continue
        ref_row = row_matches.iloc[0]
        REFERENCE_IMG = str(ladder_path)

        chosen_bgs = random.sample(bg_names, min(3, len(bg_names)))

        for bg_name in chosen_bgs:
            bg_placements = df_placements[df_placements["background_name"] == bg_name]
            if bg_placements.empty:
                continue
            
            placement = bg_placements.iloc[0]
            BACKGROUND_IMG = placement.get("background_image", os.path.join(BACKGROUNDS_DIR, f"{bg_name}.jpg"))
            
            print(f"Processing: {ladder_name} on {bg_name}")

            bg_bgr = cv2.imread(BACKGROUND_IMG)
            if bg_bgr is None:
                continue
            bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
            H, W = bg_rgb.shape[:2]

            if bg_name not in bg_cache:
                print(f"  Running depth/seg on: {bg_name}")
                bg_cache[bg_name] = (
                    estimate_depth(depth_model, bg_bgr),
                    *segment_image(seg_processor, seg_model, bg_rgb, device)
                )
            depth_map, seg_map, conf_map = bg_cache[bg_name]

            x1, y1, x2 = int(placement.x1), int(placement.y1), int(placement.x2)
            
            gxs_full, gys_full = parse_ground_boundary(placement.get("ground_boundary", ""))
            gxs, gys, y_at_x1, y_at_x2 = sample_ground_boundary_at(gxs_full, gys_full, x1, x2)

            ladder_physical_h = float(ref_row.get("physical_length_m", LADDER_HEIGHT_M))
            zone_depth_mean = float(np.median(depth_map[gys, gxs])) if len(gxs) > 0 else 0.0

            ref_bgr = cv2.imread(REFERENCE_IMG)
            ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
            ref_H, ref_W = ref_rgb.shape[:2]

            ref_mask = np.load(ref_row["mask_path"]).astype(bool)
            ys, xs   = np.where(ref_mask)
            pad      = 10

            ref_crop   = ref_rgb[max(0, ys.min()-pad):ys.max()+pad, max(0, xs.min()-pad):xs.max()+pad]
            ref_pil    = Image.fromarray(ref_crop).resize((224, 224))
            ref_tensor = get_tensor_clip()(ref_pil).unsqueeze(0).to(device)

            tight_y1, tight_y2 = ys.min(), ys.max() + 1
            tight_x1, tight_x2 = xs.min(), xs.max() + 1
            tight_rgb  = ref_rgb[tight_y1:tight_y2, tight_x1:tight_x2]
            mask_tight = ref_mask[tight_y1:tight_y2, tight_x1:tight_x2]

            # Use imported scale logic to resize the tight mask before moving into the local crop
            px_h = estimate_ladder_pixel_height(
                ladder_physical_h, seg_map, depth_map, conf_map, zone_depth_mean, H
            )
            if px_h is None:
                px_h = tight_rgb.shape[0]
            px_h = int(np.clip(px_h, int(H * 0.10), int(H * 0.85)))

            orig_h, orig_w = tight_rgb.shape[:2]
            scale_factor   = px_h / max(orig_h, 1)
            new_h, new_w   = px_h, max(int(orig_w * scale_factor), 1)

            scaled_rgb  = cv2.resize(tight_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scaled_mask = cv2.resize(mask_tight.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            scaled_corners = get_pca_corners(scaled_mask)

            # Local bounding box using ground boundaries
            cx, cy    = (x1 + x2) // 2, (y1 + max(y_at_x1, y_at_x2)) // 2
            box_size  = max(x2 - x1, max(y_at_x1, y_at_x2) - y1)
            half_size = box_size // 2

            crop_y1, crop_y2 = max(0, cy - half_size), min(H, cy + half_size)
            crop_x1, crop_x2 = max(0, cx - half_size), min(W, cx + half_size)

            bg_crop        = bg_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_H, crop_W = bg_crop.shape[:2]

            try:
                ladder_ghost, ref_mask_placed = warp_ladder_local(
                    scaled_rgb, scaled_mask, scaled_corners, 
                    x1, x2, y_at_x1, y_at_x2, 
                    crop_x1, crop_y1, crop_W, crop_H
                )
            except RuntimeError as e:
                print(f"  [SKIP] Homography failed: {e}")
                continue

            bg_pil      = Image.fromarray(bg_crop).resize((IMAGE_SIZE, IMAGE_SIZE))
            mask_pil    = Image.fromarray(ref_mask_placed.astype(np.uint8) * 255).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            bg_tensor   = get_tensor()(bg_pil).unsqueeze(0).to(device)
            mask_tensor = T.ToTensor()(mask_pil).unsqueeze(0).to(device)
            mask_tensor = (mask_tensor > 0.5).float()

            mask_patch = ref_mask_placed.astype(bool)
            ghost_bg   = bg_crop.copy()
            ghost_bg[mask_patch] = ladder_ghost[mask_patch]

            ghost_pil    = Image.fromarray(ghost_bg).resize((IMAGE_SIZE, IMAGE_SIZE))
            ghost_tensor = get_tensor()(ghost_pil).unsqueeze(0).to(device)

            inpaint_tensor = bg_tensor * (1 - mask_tensor) + ghost_tensor * mask_tensor

            # bbox coords from corners3d
            corners3d = np.load(ref_row["corners3d_path"])
            corners2d = np.zeros((8, 3), dtype=np.float64)
            for i, (X, Y, Z) in enumerate(corners3d):
                Z = max(Z, 0.1)
                corners2d[i] = [ref_W * 0.8 * X / Z + ref_W / 2.0,
                                ref_H * 0.8 * Y / Z + ref_H / 2.0,
                                Z]
            corners2d[:, 0] = np.clip(corners2d[:, 0] / ref_W, 0, 1)
            corners2d[:, 1] = np.clip(corners2d[:, 1] / ref_H, 0, 1)
            corners2d[:, 2] = np.clip(corners2d[:, 2] / 25.0 - 1.0, -1, 1)
            bbox_coords = torch.tensor(corners2d, dtype=torch.float32).unsqueeze(0).to(device)

            # build batch
            batch = {
                "image": {
                    "GT":            bg_tensor,
                    "inpaint_image": inpaint_tensor,
                    "inpaint_mask":  mask_tensor,
                    "cond": {
                        "ref_image": ref_tensor,
                        "ref_bbox":  bbox_coords,
                    },
                },
                "lidar": {},
            }

            # run DDIM sampling
            from ldm.models.diffusion.ddim import DDIMSampler
            sampler = DDIMSampler(model)

            with torch.no_grad():
                out = model.get_input(batch, "inpaint", force_c_encode=True)
                shape = (model.channels, model.image_size, model.image_size)
                uc = model.learnable_vector.repeat(1, 1, 1)
                samples, _ = sampler.sample(
                    S=DDIM_STEPS,
                    conditioning=out["cond"],
                    batch_size=1,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=5.0,
                    unconditional_conditioning=uc,
                    eta=1.0,
                    x_T=None,
                    test_model_kwargs={
                        "inpaint_image": out["z"][:, 4:8],
                        "inpaint_mask":  out["z"][:, [8]],
                    }
                )
                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0, 1)

            # paste result back into original background at placement zone
            result_np   = (x_samples[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
            result_crop = cv2.resize(result_np, (crop_W, crop_H))

            output_bg  = bg_rgb.copy()
            placed_mask_bool = ref_mask_placed.astype(bool)

            target_roi = output_bg[crop_y1:crop_y2, crop_x1:crop_x2]
            target_roi[placed_mask_bool] = result_crop[placed_mask_bool]
            output_bg[crop_y1:crop_y2, crop_x1:crop_x2] = target_roi

            output_bgr = cv2.cvtColor(output_bg, cv2.COLOR_RGB2BGR)
            OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{bg_name}__{ladder_name}.png")
            cv2.imwrite(OUTPUT_PATH, output_bgr)
            print(f"Final image saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
