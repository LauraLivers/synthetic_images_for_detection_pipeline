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

BACKGROUND_IMG = "robo_images/ladder/1b81499856df49858d2f837de745cda0_front_5_March_2026_13-02.jpg"
REFERENCE_IMG  = "robo_images/ladder/02e903d139804b5291b44f8aa5cf87b4_front_5_March_2026_12-58.jpg"
PLACEMENTS_CSV = "MoBI_outputs/background6_segf_filter/placements.csv"
CHECKPOINT     = "../MoBI_outputs/mobi_inpaining1/2026-04-14T02-17-13_ladder_dataset_mobi/checkpoints/epoch=000005.ckpt"
CONFIG_PATH    = "ladder_dataset_mobi.yaml"
OUTPUT_PATH    = "MoBI_outputs/inpainted_result.png"

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

# main 
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load config and model
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

    # build mask
    mask_np = build_mask(x1, y1, x2, y2, H, W)
    

    # crop and resize background to model input size
    bg_pil = Image.fromarray(bg_rgb).resize((IMAGE_SIZE, IMAGE_SIZE))
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

    bg_tensor   = get_tensor()(bg_pil).unsqueeze(0).to(device)       # (1,3,512,512)
    mask_tensor = T.ToTensor()(mask_pil).unsqueeze(0).to(device)     # (1,1,512,512)
    mask_tensor = (mask_tensor > 0.5).float()
    inpaint_tensor = bg_tensor * (1 - mask_tensor)                    # masked background

    # load reference ladder image
    ref_bgr = cv2.imread(REFERENCE_IMG)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    ref_pil = Image.fromarray(ref_rgb).resize((224, 224))
    ref_tensor = get_tensor_clip()(ref_pil).unsqueeze(0).to(device)
    

    # dummy bbox coords (normalised 8x3)
    bbox_coords = torch.zeros(1, 8, 3).to(device)
    ref_full_pil = Image.fromarray(ref_rgb).resize((IMAGE_SIZE, IMAGE_SIZE))
    ref_full_tensor = get_tensor()(ref_full_pil).unsqueeze(0).to(device)
    
    # build batch
    batch = {
        "image": {
            "GT":           ref_full_tensor,
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
        c = out["cond"]
        cond = {"c_crossattn": [c]}
        print("c shape:", c.shape)
        print("c min/max:", c.min().item(), c.max().item())
        print("ref_tensor min/max:", ref_tensor.min().item(), ref_tensor.max().item())
        # Does c match encoding of ref_tensor or bg_tensor?
        c_from_ref, _ = model.process_conditioning({"ref_image": ref_tensor}, force_c_encode=True)
        c_from_bg, _ = model.process_conditioning({"ref_image": bg_tensor[:, :, :224, :224]}, force_c_encode=True)
        print("c equals ref encoding:", torch.allclose(c, c_from_ref))
        print("c equals bg encoding:", torch.allclose(c, c_from_bg))
    
        shape = (model.channels, model.image_size, model.image_size)
        samples, _ = sampler.sample(
            DDIM_STEPS, 1, shape,
            conditioning=cond,
            verbose=False,
            eta=1.0,
            x_T=None,
            rest=out["z"][:, 4:, :, :]
        )
        print("z shape:", z.shape)
        print("z min/max:", z.min().item(), z.max().item())
        print("samples shape:", samples.shape)
        print("samples min/max:", samples.min().item(), samples.max().item())
        print("samples equal to z_inpaint:", torch.allclose(samples, out["z"][:,4:8], atol=1e-2))
        x_samples = model.decode_first_stage(samples[0] if isinstance(samples, list) else samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0, 1)

    # paste result back into original background at placement zone
    result_np = (x_samples[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    result_pil = Image.fromarray(result_np).resize((x2-x1, y2-y1))

    output_bg = bg_rgb.copy()
    output_bg[y1:y2, x1:x2] = np.array(result_pil)
    Image.fromarray(result_np).save("MoBI_outputs/full_generated_512.png")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    Image.fromarray(output_bg).save(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()