"""
Inference script: insert a reference ladder into a background image
using the trained Paint-by-Example model.

Usage:
    uv run python infer.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
import torchvision.transforms as T
import torchvision
import inspect


os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── paths ──────────────────────────────────────────────────────────────────
MOBI_ROOT      = os.path.abspath("../MobI")
MY_FOLDER      = os.path.abspath(".")
TAMING_ROOT    = os.path.abspath("../taming-transformers")

for p in [MOBI_ROOT, MY_FOLDER, TAMING_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

BACKGROUND_IMG = "robo_images/no_ladder/005cc92de4794b4d9259d80c5812d72d_right_5_March_2026_13-00.jpg"
REFERENCE_IMG  = "robo_images/ladder/02e903d139804b5291b44f8aa5cf87b4_front_5_March_2026_12-58.jpg"
PLACEMENTS_CSV = "MoBI_outputs/background6_segf_filter/placements.csv"
CHECKPOINT     = "../MoBI_outputs/mobi_inpaining1/2026-04-14T02-17-13_ladder_dataset_mobi/checkpoints/epoch=000005.ckpt"
CONFIG_PATH    = "ladder_dataset_mobi.yaml"
OUTPUT_PATH    = "MoBI_outputs/inpainted_result.png"
LADDER_INSTANCES_CSV = "MoBI_outputs/ladder_segmentations_real_size3/ladder_instances.csv"

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

# def build_mask(x1, y1, x2, y2, H, W):
#     mask = np.zeros((H, W), dtype=np.float32)
#     mask[y1:y2, x1:x2] = 1.0
#     return mask

# def warp_to_slope(source_img, src_pca_pts, dst_x1, dst_y1, dst_x2, dst_y2_l, dst_y2_r, crop_W, crop_H, offset_px=(0, 0)):
#     """
#     Warp using the true oriented PCA corners of the ladder. 
#     The bottom edge of the PCA box aligns with the sloped ground line.
#     """
#     # Sort PCA points by Y to find the "bottom" edge of the ladder
#     sorted_by_y = src_pca_pts[src_pca_pts[:, 1].argsort()]
#     top_pts = sorted_by_y[:2]
#     bottom_pts = sorted_by_y[2:]
    
#     # Sort X to get Top-Left, Top-Right, Bottom-Left, Bottom-Right
#     tl = top_pts[top_pts[:, 0].argsort()][0]
#     tr = top_pts[top_pts[:, 0].argsort()][1]
#     bl = bottom_pts[bottom_pts[:, 0].argsort()][0]
#     br = bottom_pts[bottom_pts[:, 0].argsort()][1]
    
#     src_pts = np.array([tl, tr, bl, br], dtype=np.float32)
    
#     # The destination points: bottom edge locks to the ground slope (y2_l to y2_r)
#     # Applying a slight parallel offset if needed (e.g. sinking slightly into the ground)
#     ox, oy = offset_px
#     dst_pts = np.array([
#         [dst_x1 + ox, dst_y1 + oy],
#         [dst_x2 + ox, dst_y1 + oy],
#         [dst_x1 + ox, dst_y2_l + oy],
#         [dst_x2 + ox, dst_y2_r + oy]
#     ], dtype=np.float32)

#     M, _ = cv2.findHomography(src_pts, dst_pts)
#     if M is None:  # Fallback to affine if homography fails
#         M = cv2.getAffineTransform(src_pts[:3].astype(np.float32), dst_pts[:3].astype(np.float32))
#         warped = cv2.warpAffine(source_img, M, (crop_W, crop_H), flags=cv2.INTER_LINEAR)
#     else:
#         warped = cv2.warpPerspective(source_img, M, (crop_W, crop_H), flags=cv2.INTER_LINEAR)
    
#     return warped
def warp_to_slope(source_img, src_corners, dst_x1, dst_y1, dst_x2, dst_y2_l, dst_y2_r, canvas_W, canvas_H):
    edges = sorted(get_edge_length(src_corners), key=lambda e: e[0])
    _, ia, ib = edges[1]
    bl_src = src_corners[ia].astype(np.float64)
    br_src = src_corners[ib].astype(np.float64)
    if bl_src[0] > br_src[0]:
        bl_src, br_src = br_src, bl_src
    bl_dst = np.array([dst_x1, dst_y2_l], dtype=np.float64)
    br_dst = np.array([dst_x2, dst_y2_r], dtype=np.float64)

    src_vec = br_src - bl_src
    dst_vec = br_dst - bl_dst
    src_angle = np.arctan2(src_vec[1], src_vec[0])
    dst_angle = np.arctan2(dst_vec[1], dst_vec[0])
    angle = dst_angle - src_angle
    scale = np.linalg.norm(dst_vec) / np.linalg.norm(src_vec)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = scale * np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    dst_corners = ((src_corners - bl_src) @ R.T) + bl_dst

    src_pts = src_corners.astype(np.float32)
    dst_pts = dst_corners.astype(np.float32)
    M, _ = cv2.findHomography(src_pts, dst_pts)

    if M is None:
        raise RuntimeError("no Homography found")
    
    return cv2.warpPerspective(source_img, M, (canvas_W, canvas_H),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(CONFIG_PATH)

    print("Loading model...")
    model = instantiate_from_config(config.model)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device).eval()
    print("Model loaded.")

    # background image
    bg_bgr = cv2.imread(BACKGROUND_IMG)
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    H, W = bg_rgb.shape[:2]

    # placement zone
    placement = load_placement(PLACEMENTS_CSV, BACKGROUND_IMG)
    x1, y1, x2 = int(placement.x1), int(placement.y1), int(placement.x2)
    y2_l, y2_r = int(placement.y2_left), int(placement.y2_right)
    y2 = max(y2_l, y2_r)
    print(f"Placement zone: X:({x1}->{x2}), Y_Top:{y1}, Y_Bottom_Slope:({y2_l}->{y2_r})")

    # reference ladder image, mask, crop
    ref_bgr = cv2.imread(REFERENCE_IMG)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    ref_H, ref_W = ref_rgb.shape[:2]

    instances = pd.read_csv(LADDER_INSTANCES_CSV, index_col=0)
    ref_row   = instances[instances["image_path"] == REFERENCE_IMG].iloc[0]

    ref_mask = np.load(ref_row["mask_path"]).astype(bool)
    ys, xs   = np.where(ref_mask)
    pad      = 10

    # cropped reference image (Paint-by-Example CLIP embeddings break if given pure black backgrounds)
    ref_crop   = ref_rgb[max(0, ys.min()-pad):ys.max()+pad, max(0, xs.min()-pad):xs.max()+pad]
    ref_pil    = Image.fromarray(ref_crop).resize((224, 224))
    ref_tensor = get_tensor_clip()(ref_pil).unsqueeze(0).to(device)

    # tight crop for warping
    tight_y1, tight_y2 = ys.min(), ys.max() + 1
    tight_x1, tight_x2 = xs.min(), xs.max() + 1
    tight_rgb  = ref_rgb[tight_y1:tight_y2, tight_x1:tight_x2]
    mask_tight = ref_mask[tight_y1:tight_y2, tight_x1:tight_x2]

    # PCA corners from mask
    corners_in_tight = get_pca_corners(mask_tight)

    # local crop around the placement zone
    cx, cy    = (x1 + x2) // 2, (y1 + y2) // 2
    box_size  = max(x2 - x1, y2 - y1)
    half_size = box_size // 2

    crop_y1, crop_y2 = max(0, cy - half_size), min(H, cy + half_size)
    crop_x1, crop_x2 = max(0, cx - half_size), min(W, cx + half_size)

    # local coordinates of the placement zone within the crop
    loc_x1   = x1   - crop_x1
    loc_x2   = x2   - crop_x1
    loc_y1   = y1   - crop_y1
    loc_y2_l = y2_l - crop_y1
    loc_y2_r = y2_r - crop_y1

    # define crop
    bg_crop        = bg_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_H, crop_W = bg_crop.shape[:2]

    # warp mask and image
    ref_mask_placed = warp_to_slope(
        mask_tight.astype(np.uint8), corners_in_tight,
        loc_x1, loc_y1, loc_x2, loc_y2_l, loc_y2_r,
        crop_W, crop_H
    ).astype(bool)

    ladder_ghost = warp_to_slope(
        tight_rgb, corners_in_tight,
        loc_x1, loc_y1, loc_x2, loc_y2_l, loc_y2_r,
        crop_W, crop_H
    )

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
    cv2.imwrite(OUTPUT_PATH, output_bgr)
    print(f"Final image saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()