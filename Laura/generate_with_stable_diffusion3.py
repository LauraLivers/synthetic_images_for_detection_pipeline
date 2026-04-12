"""
uv run generate_sd3.py --input_dir [root_folder_hard_samples]/ --output_dir DiffCL_images/[new_output_dir]

Key differences from generate.py:
- Model: stabilityai/stable-diffusion-3.5-large (MMDiT transformer, flow matching)
- dtype: bfloat16 (SD3.5 default, float32 fallback for MPS)
- Image resizing: multiples of 64 instead of multiples of 8
- No original_size / target_size micro-conditioning (SDXL-specific, not used in SD3)
- num_inference_steps: 28 (SD3.5 default, fewer steps needed vs SDXL)
- base_prompt: class-label anchor fed to CLIP encoders only, decoupling object
  identity from scene conditioning (same trick as original, ported to SD3.5)
- guidance_scale_for_img_guid: same coupling logic as original, kept intact
"""

import os
import csv
import argparse
import random
from pathlib import Path
from string import Template
from dotenv import load_dotenv
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

from model import StableDiffusion3Img2ImgPipelineWithBasePrompt

LOCATION_PROMPTS = {
    "concrete": "a 8k real photo: aluminium ladder on the floor leaning against a concrete wall",
    "park":     "a 8k real photo: a ladder in a park",
    "street":   "a 8k real photo: a ladder in urban setting",
}

# Class Anchor: fed only to CLIP-L and CLIP-G to stabilize object-class pooled conditions
BASE_PROMPT_TEMPLATE = Template("a photo of a $classname")
CLASS_NAME = "ladder"
BASE_PROMPT = BASE_PROMPT_TEMPLATE.substitute(classname=CLASS_NAME)

IMG_GUIDANCES = [0.4, 0.5, 0.6, 0.7] # FINETUNE
RANDOM_SEEDS = [30, 40] # FINETUNE?

SD_MODEL = 'stabilityai/stable-diffusion-3.5-medium'
CLIP_MODEL = 'openai/clip-vit-base-patch32'
CLIP_THRESHOLD = 0.7

load_dotenv()

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_sd_pipeline(model_id: str, device: str):
    print(f"Loading SD3.5 pipeline: {model_id}")
    if device == "mps":
        # MPS does not support bfloat16 reliably
        pipe = StableDiffusion3Img2ImgPipelineWithBasePrompt.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        ).to(device)
    elif device == "cpu":
        pipe = StableDiffusion3Img2ImgPipelineWithBasePrompt.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        ).to(device)
    else:
        # cuda — bfloat16 is the recommended dtype for SD3.5
        pipe = StableDiffusion3Img2ImgPipelineWithBasePrompt.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device)
        # Remove if you have 24GB+ VRAM
        pipe.enable_model_cpu_offload()
    return pipe

def load_clip_model(model_id: str, device: str):
    print(f"Loading CLIP model: {model_id}")
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    return model, processor, tokenizer

def guidance_scale_for_img_guid(img_guid: float) -> float:
    """ higher image guidance (less noise) → lower text guidance scale, to avoid over-processing."""
    strength = 1 - img_guid
    return round(12.0 - (strength * 6.0), 1)

def generate_variant(pipe, image: Image.Image, prompt: str, base_prompt: str, img_guid: float,
                     text_guid: float, seed: int, output_path: Path) -> Image.Image:
    """
    Generate one variant and save it. Returns the PIL image.
 
    SD3.5 requires image dimensions to be multiples of 64.
    Resizes to closest multiple of 64 preserving approximate aspect ratio,
    then resizes output back to original dimensions.
    """
    original_width, original_height = image.size
 
    target_width  = (original_width  // 64) * 64
    target_height = (original_height // 64) * 64

    target_width  = max(target_width,  64)
    target_height = max(target_height, 64)
 
    image_resized = image.resize((target_width, target_height), Image.LANCZOS)
 
    set_seed(seed)
    results = pipe(
        prompt=prompt,
        base_prompt=base_prompt,
        image=image_resized,
        strength=1 - img_guid,          # same semantics as original
        guidance_scale=text_guid,
        num_inference_steps=28,          # SD3.5 default, fewer steps than SDXL
        # No original_size / target_size: those are SDXL micro-conditioning args,
    )
 
    generated = results.images[0]
    generated = generated.resize((original_width, original_height), Image.LANCZOS)
    generated.save(output_path)
    return generated

def get_image_embedding(processor, model, image: Image.Image, device: str) -> np.ndarray:
    pixel_values = processor(text=None, images=image, return_tensors="pt", do_rescale=True)[
        "pixel_values"
    ].to(device)
    output = model.get_image_features(pixel_values)
    if hasattr(output, "pooler_output"):
        embedding = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        embedding = output.last_hidden_state[:, 0, :]
    else:
        embedding = output
    return embedding.cpu().detach().numpy()

def get_text_embedding(tokenizer, model, text: str, device: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output = model.get_text_features(**inputs)
    if hasattr(output, "pooler_output"):
        embedding = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        embedding = output.last_hidden_state[:, 0, :]
    else:
        embedding = output
    return embedding.cpu().detach().numpy()

def compute_clip_scores(
    clip_model, processor, tokenizer,
    original_image: Image.Image,
    generated_image: Image.Image,
    prompt: str,
    device: str,
) -> tuple[float, float]:
    """Returns (text_similarity, image_similarity) against the original."""
    orig_emb = get_image_embedding(processor, clip_model, original_image, device)
    gen_emb  = get_image_embedding(processor, clip_model, generated_image, device)
    text_emb = get_text_embedding(tokenizer, clip_model, prompt, device)
 
    text_sim = float(cosine_similarity(gen_emb, text_emb)[0][0])
    img_sim  = float(cosine_similarity(orig_emb, gen_emb)[0][0])
    return text_sim, img_sim

def process_location(
    location: str,
    image_paths: list[Path],
    prompt: str,
    base_prompt: str,
    pipe,
    clip_model,
    processor,
    tokenizer,
    output_dir: Path,
    device: str,
) -> list[dict]:
    rows = []
    location_out = output_dir / location
    location_out.mkdir(parents=True, exist_ok=True)
 
    for img_path in image_paths:
        print(f"\n  [{location}] Processing: {img_path.name}")
        original = Image.open(img_path).convert("RGB")
 
        for seed in RANDOM_SEEDS:
            for img_guid in IMG_GUIDANCES:
                combined_seed = (hash(img_path.name) ^ seed) % (2**16)
 
                out_filename = f"{img_path.stem}_guid{int(img_guid*100)}_seed{seed}.jpg"
                out_path = location_out / out_filename
 
                if out_path.exists():
                    print(f"Skipping (exists): {out_filename}")
                    generated = Image.open(out_path).convert("RGB")
                else:
                    text_guid = guidance_scale_for_img_guid(img_guid)
                    print(
                        f"Generating: {out_filename}  "
                        f"(img_guid={img_guid}, text_guid={text_guid}, seed={seed})"
                    )
                    generated = generate_variant(
                        pipe=pipe,
                        image=original,
                        prompt=prompt,
                        base_prompt=base_prompt,
                        img_guid=img_guid,
                        text_guid=text_guid,
                        seed=combined_seed,
                        output_path=out_path,
                    )
 
                # CLIP scoring — resize for efficiency, same as original
                gen_resized  = generated.resize((480, 270))
                orig_resized = original.resize((480, 270))
                text_sim, img_sim = compute_clip_scores(
                    clip_model, processor, tokenizer,
                    orig_resized, gen_resized, prompt, device,
                )
 
                rows.append({
                    "location":       location,
                    "source_image":   str(img_path),
                    "generated_path": str(out_path),
                    "prompt":         prompt,
                    "base_prompt":    base_prompt,
                    "img_guidance":   img_guid,
                    "seed":           seed,
                    "text_sim":       round(text_sim, 4),
                    "img_sim":        round(img_sim, 4),
                    "passes_filter":  img_sim >= CLIP_THRESHOLD,
                })
 
    return rows

def main():
    print("main() called", flush=True)
    parser = argparse.ArgumentParser(
        description="DiffCL hard-sample augmentation for ladder detection — SD3.5 version"
    )
    parser.add_argument("--input_dir",  type=Path, required=True,
                        help="Root folder with subfolders per location")
    parser.add_argument("--output_dir", type=Path, default=Path("generated_sd3"),
                        help="Where to save generated images and CSV log")
    parser.add_argument("--clip_threshold", type=float, default=CLIP_THRESHOLD,
                        help="Minimum image-image CLIP similarity to mark as passing filter")
    args = parser.parse_args()
 
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
 
    args.output_dir.mkdir(parents=True, exist_ok=True)
 
    found_locations = {
        d.name: sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
        for d in sorted(args.input_dir.iterdir())
        if d.is_dir()
    }
 
    if not found_locations:
        raise ValueError(f"No subfolders found in {args.input_dir}")
 
    for loc in found_locations:
        if loc not in LOCATION_PROMPTS:
            print(
                f"WARNING: No prompt defined for location '{loc}'. "
                f"Add it to LOCATION_PROMPTS in generate_sd3.py. Skipping."
            )
 
    # Load models once
    pipe = load_sd_pipeline(SD_MODEL, device)
    clip_model, processor, tokenizer = load_clip_model(CLIP_MODEL, device)
 
    all_rows = []
 
    for location, image_paths in found_locations.items():
        if location not in LOCATION_PROMPTS:
            continue
        prompt = LOCATION_PROMPTS[location]
        print(
            f"Location: {location}  |  {len(image_paths)} images  |  "
            f"Prompt: '{prompt}'  |  Base: '{BASE_PROMPT}'"
        )
 
        rows = process_location(
            location=location,
            image_paths=image_paths,
            prompt=prompt,
            base_prompt=BASE_PROMPT,
            pipe=pipe,
            clip_model=clip_model,
            processor=processor,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            device=device,
        )
        all_rows.extend(rows)
 
    csv_path = args.output_dir / "clip_scores.csv"
    fieldnames = [
        "location", "source_image", "generated_path", "prompt", "base_prompt",
        "img_guidance", "seed", "text_sim", "img_sim", "passes_filter",
    ]
 
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
 
    total   = len(all_rows)
    passing = sum(1 for r in all_rows if r["passes_filter"])
    print(
        f"Done. Generated {total} images, {passing} pass CLIP filter "
        f"(threshold={args.clip_threshold})"
    )
    print(f"CSV log saved to: {csv_path}")
 
 
if __name__ == "__main__":
    main()