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

def build_mask(x1, y1, x2, y2, H, W):
    mask = np.zeros((H, W), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask

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

    # load background image
    bg_bgr = cv2.imread(BACKGROUND_IMG)
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    H, W = bg_rgb.shape[:2]

    # load placement zone
    placement = load_placement(PLACEMENTS_CSV, BACKGROUND_IMG)
    x1, y1, x2, y2 = int(placement.x1), int(placement.y1), int(placement.x2), int(placement.y2)
    print(f"Placement zone: ({x1},{y1}) -> ({x2},{y2})")
    
    # --- VISUAL DEBUGGING: PLACEMENT ZONE ---
    os.makedirs("MoBI_outputs/debug", exist_ok=True)
    bg_debug = bg_bgr.copy()
    cv2.rectangle(bg_debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite("MoBI_outputs/debug/debug_00a_placement_on_bg.png", bg_debug)
    # ----------------------------------------

    # load reference ladder image, mask, crop
    ref_bgr = cv2.imread(REFERENCE_IMG)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    ref_H, ref_W = ref_rgb.shape[:2]
    
    # --- VISUAL DEBUGGING: REFERENCE LADDER CROP ---
    cv2.imwrite("MoBI_outputs/debug/debug_00b_reference_raw.png", ref_bgr)
    # -----------------------------------------------

    instances = pd.read_csv(LADDER_INSTANCES_CSV, index_col=0)
    ref_row  = instances[instances["image_path"] == REFERENCE_IMG].iloc[0]
    
    ref_mask = np.load(ref_row["mask_path"]).astype(bool)
    ys, xs   = np.where(ref_mask)
    pad      = 10
    ref_crop = ref_rgb[max(0, ys.min()-pad):ys.max()+pad, max(0, xs.min()-pad):xs.max()+pad]
    
    # --- VISUAL DEBUGGING: TIGHT MASK VS FULL IMAGE FORMAT ---
    # This will prove if resizing the full image into the placement box 
    # compresses the ladder because of excessive empty padding in the reference image.
    mask_bbox_vis = ref_bgr.copy()
    tight_x1, tight_y1 = xs.min(), ys.min()
    tight_x2, tight_y2 = xs.max(), ys.max()
    # Draw tight bounding box around just the ladder in MAGENTA
    cv2.rectangle(mask_bbox_vis, (tight_x1, tight_y1), (tight_x2, tight_y2), (255, 0, 255), 4)
    # Draw border representing the full reference image size in YELLOW
    cv2.rectangle(mask_bbox_vis, (0, 0), (ref_W-1, ref_H-1), (0, 255, 255), 6)
    
    # Calculate and document the exact shrink percentage
    ladder_w = tight_x2 - tight_x1
    ladder_h = tight_y2 - tight_y1
    w_ratio = ladder_w / ref_W * 100
    h_ratio = ladder_h / ref_H * 100
    cv2.putText(mask_bbox_vis, f"Ladder is {w_ratio:.1f}% W, {h_ratio:.1f}% H of full image", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    print(f"\n[DEBUG CONCLUSION] Ladder object is {w_ratio:.1f}% W x {h_ratio:.1f}% H of the full reference image.")
    print("[DEBUG CONCLUSION] Mapping the FULL image into the placement box shrinks the ladder by this exact margin.\n")

    cv2.imwrite("MoBI_outputs/debug/debug_00e_mask_tight_vs_full.png", mask_bbox_vis)
    # ---------------------------------------------------------

    # --- VISUAL DEBUGGING: SCALE CLASH PROOF ---
    # Paste the RAW, unscaled reference crop onto the background to visually compare
    # the original pixel size vs the tiny placement bounding box.
    scale_clash_vis = bg_bgr.copy()
    c_h, c_w = ref_crop.shape[:2]
    # Try to safely paste it near the placement zone for comparison
    paste_y2 = min(H, y1 + c_h)
    paste_x2 = min(W, x1 + c_w)
    actual_h = paste_y2 - y1
    actual_w = paste_x2 - x1
    
    if actual_h > 0 and actual_w > 0:
        scale_clash_vis[y1:paste_y2, x1:paste_x2] = cv2.cvtColor(ref_crop[:actual_h, :actual_w], cv2.COLOR_RGB2BGR)
        # Draw the target placement zone in RED (what it shrinks down to)
        cv2.rectangle(scale_clash_vis, (x1, y1), (x2, y2), (0, 0, 255), 4)
        # Draw the original ladder's actual footprint in BLUE (what it starts as)
        cv2.rectangle(scale_clash_vis, (x1, y1), (paste_x2, paste_y2), (255, 0, 0), 4)
    cv2.imwrite("MoBI_outputs/debug/debug_00d_scale_clash_comparison.png", scale_clash_vis)
    # -------------------------------------------

    ref_pil  = Image.fromarray(ref_crop).resize((224, 224))
    ref_tensor = get_tensor_clip()(ref_pil).unsqueeze(0).to(device)

    # build inpaint inputs from reference image and its segmentation mask
    ref_mask_placed = np.zeros((H, W), dtype=np.uint8)
    ref_mask_placed[y1:y2, x1:x2] = cv2.resize(ref_mask.astype(np.uint8), (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
    bg_pil      = Image.fromarray(bg_rgb).resize((IMAGE_SIZE, IMAGE_SIZE))
    mask_pil    = Image.fromarray(ref_mask_placed * 255).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    bg_tensor   = get_tensor()(bg_pil).unsqueeze(0).to(device)
    mask_tensor = T.ToTensor()(mask_pil).unsqueeze(0).to(device)
    mask_tensor = (mask_tensor > 0.5).float()
    inpaint_tensor = bg_tensor * (1 - mask_tensor)

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

    # --- VISUAL DEBUGGING: 3D TO 2D CORNERS ON REFERENCE ---
    ref_bbox_debug = ref_bgr.copy()
    for (px, py, pz) in corners2d:
        cx, cy = int(px * ref_W), int(py * ref_H)
        cv2.circle(ref_bbox_debug, (cx, cy), 5, (0, 0, 255), -1)
    cv2.imwrite("MoBI_outputs/debug/debug_00c_projected_corners.png", ref_bbox_debug)
    # -------------------------------------------------------

    # --- DEBUGGING HIGHLIGHT START ---
    print("[DEBUG] OVERWRITE WARNING: bg_tensor, mask_tensor, and inpaint_tensor are being reassigned here")
    print("[DEBUG] They previously held the empty background image but are now being replaced by the reference ladder image.")
    # --- DEBUGGING HIGHLIGHT END ---

    ref_full_pil = Image.fromarray(ref_rgb).resize((IMAGE_SIZE, IMAGE_SIZE))
    ref_full_tensor = get_tensor()(ref_full_pil).unsqueeze(0).to(device)
    ref_mask_pil = Image.fromarray((ref_mask.astype(np.uint8) * 255)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    bg_tensor    = get_tensor()(ref_full_pil).unsqueeze(0).to(device)
    mask_tensor  = T.ToTensor()(ref_mask_pil).unsqueeze(0).to(device)
    mask_tensor  = (mask_tensor > 0.5).float()
    inpaint_tensor = bg_tensor * (1 - mask_tensor)

    # --- DEBUGGING OUTPUTS START ---
    import torchvision.utils as vutils
    os.makedirs("MoBI_outputs/debug", exist_ok=True)
    
    # Save the tensors to visually inspect what the model is actually receiving
    def denorm(t):
        return torch.clamp((t + 1.0) / 2.0, 0, 1)
        
    vutils.save_image(denorm(bg_tensor), "MoBI_outputs/debug/debug_01_bg_tensor.png")
    vutils.save_image(mask_tensor, "MoBI_outputs/debug/debug_02_mask_tensor.png")
    vutils.save_image(denorm(inpaint_tensor), "MoBI_outputs/debug/debug_03_inpaint_tensor.png")
    
    # Also save the original placement-based background and mask created earlier
    # to compare against what actually ended up in inpaint_tensor
    bg_pil.save("MoBI_outputs/debug/debug_04_original_bg_pil.png")
    mask_pil.save("MoBI_outputs/debug/debug_05_original_mask_pil.png")
    # --- DEBUGGING OUTPUTS END ---

    # build batch
    batch = {
        "image": {
            "GT":            ref_full_tensor,
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
        z = out["z"]
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
    result_full = np.array(Image.fromarray(result_np).resize((ref_W, ref_H)))
    
    # --- VISUAL DEBUGGING: WHAT IS BEING PASTED ---
    cv2.imwrite("MoBI_outputs/debug/debug_06_raw_model_output.png", cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite("MoBI_outputs/debug/debug_07_upscaled_model_output.png", cv2.cvtColor(result_full, cv2.COLOR_RGB2BGR))
    # ----------------------------------------------
    
    output_bg   = bg_rgb.copy()
    ladder_region = result_full[ref_mask]
    ys_place = np.linspace(y1, y2-1, ladder_region.shape[0]).astype(int)

    # --- DEBUGGING: ACTUALLY PASTE THE PIXELS ---
    # The original code calculated the region but never mapped it onto output_bg
    # We will resize the 2D output image to the bounding box to see it in context.
    
    placed_result = cv2.resize(result_full, (x2 - x1, y2 - y1))
    placed_mask = cv2.resize(ref_mask.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # Only paste where the newly resized mask is True
    placement_zone = output_bg[y1:y2, x1:x2]
    placement_zone[placed_mask] = placed_result[placed_mask]
    output_bg[y1:y2, x1:x2] = placement_zone
    # --------------------------------------------

    # --- DEBUGGING OUTPUT FIX START ---
    # The result was being calculated but never saved!
    output_bgr = cv2.cvtColor(output_bg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_PATH, output_bgr)
    print(f"[DEBUG] Final image saved to {OUTPUT_PATH}")
    # --- DEBUGGING OUTPUT FIX END ---

if __name__ == "__main__":
    main()