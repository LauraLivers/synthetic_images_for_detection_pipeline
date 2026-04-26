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

    # load reference ladder image, mask, crop
    ref_bgr = cv2.imread(REFERENCE_IMG)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    ref_H, ref_W = ref_rgb.shape[:2]

    instances = pd.read_csv(LADDER_INSTANCES_CSV, index_col=0)
    ref_row  = instances[instances["image_path"] == REFERENCE_IMG].iloc[0]
    
    ref_mask = np.load(ref_row["mask_path"]).astype(bool)
    ys, xs   = np.where(ref_mask)
    pad      = 10
    
    # Use the natural cropped reference image (Paint-by-Example CLIP embeddings break if given pure black backgrounds)
    ref_crop = ref_rgb[max(0, ys.min()-pad):ys.max()+pad, max(0, xs.min()-pad):xs.max()+pad]
    
    ref_pil  = Image.fromarray(ref_crop).resize((224, 224))
    ref_tensor = get_tensor_clip()(ref_pil).unsqueeze(0).to(device)

    # get tight mask to fix scaling issue 
    tight_y1, tight_y2 = ys.min(), ys.max() + 1
    tight_x1, tight_x2 = xs.min(), xs.max() + 1
    ref_mask_tight = ref_mask[tight_y1:tight_y2, tight_x1:tight_x2]

    # build inpaint inputs using background and tight mask
    ref_mask_placed = np.zeros((H, W), dtype=np.uint8)
    ref_mask_placed[y1:y2, x1:x2] = cv2.resize(ref_mask_tight.astype(np.uint8), (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
    
    bg_pil      = Image.fromarray(bg_rgb).resize((IMAGE_SIZE, IMAGE_SIZE))
    mask_pil    = Image.fromarray(ref_mask_placed * 255).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    bg_tensor   = get_tensor()(bg_pil).unsqueeze(0).to(device)
    mask_tensor = T.ToTensor()(mask_pil).unsqueeze(0).to(device)
    
    # Restore standard mask polarity: 1.0 = "missing area to generate", 0.0 = "known context to keep"
    mask_tensor = (mask_tensor > 0.5).float()
    
    # [DEBUG HYPOTHESIS TEST: GHOSTING]
    # Instead of a pure grey hole, we paste the actual resized reference ladder into the mask hole, 
    # but faded/noisy. If it generates a perfect ladder now, it proves the model 
    # learned to "enhance/reconstruct" existing pixels instead of generating from scratch.
    ladder_ghost = cv2.resize(ref_crop, (x2-x1, y2-y1))
    ghost_bg = bg_rgb.copy()
    ghost_bg[y1:y2, x1:x2][ref_mask_tight.astype(bool)] = ladder_ghost[ref_mask_tight.astype(bool)]
    ghost_pil = Image.fromarray(ghost_bg).resize((IMAGE_SIZE, IMAGE_SIZE))
    ghost_tensor = get_tensor()(ghost_pil).unsqueeze(0).to(device)
    
    # Use the ghosted background specifically inside the hole
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

    # build batch (removed the buggy overwrite block)
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
    result_full = cv2.resize(result_np, (W, H))
    
    output_bg = bg_rgb.copy()
    placed_mask_bool = ref_mask_placed.astype(bool)
    output_bg[placed_mask_bool] = result_full[placed_mask_bool]

    output_bgr = cv2.cvtColor(output_bg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_PATH, output_bgr)
    print(f"Final image saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()