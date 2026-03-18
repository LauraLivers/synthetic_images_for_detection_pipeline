"""
uv run generaty.py --input_dir [root_folder_hard_samples]/ --output_dir DiffCL_images/[new_output_dir] 
"""
print("Script started", flush=True)
import os
import csv
import argparse
import random
from pathlib import Path
 
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
 
from model import StableDiffusionXLImg2ImgPipeline


LOCATION_PROMPTS = {
    "concrete" : "a photo of a ladder leaning against a concrete wall surrounded with office building in the background",
    "park" : "a photo of a ladder in a park, partially hidden by vegetation",
    "street" : "a phot of a ladder leaning against a fence with a busy street behind, partially occluded"
}

IMG_GUIDANCES = [0.8, 0.85, 0.9, 0.95] # exclude smaller values to preserve geometry of original
TEXT_GUIDANCE = 10
RANDOM_SEEDS = [10, 20, 30, 40]

SD_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
CLIP_MODEL = "openai/clip-vit-base-patch32"

CLIP_THRESHOLD = 0.50 # the higher the less permissive is the filter

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # works for MPS
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_sd_pipeline(model_id: str, device: str):
    print(f"Loading SD pipeline: {model_id}")
    if device == "cpu":
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id, use_safetensors=True
        ).to(device)
    elif device == "mps":
        # MPS doesn't support float16 variants reliably
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32, use_safetensors=True
        ).to(device)
    else:
        # cuda
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(device)
    return pipe

def load_clip_model(model_id: str, device: str):
    print(f"Loading CLIP model: {model_id}")
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    return model, processor, tokenizer

def generate_variant(pipe, image: Image.Image, prompt: str,
    img_guid: float, # how much to preserve the original
    text_guid: float, seed: int, output_path: Path) -> Image.Image:
    """Generates one variant and saves it. Returns the PIL image."""
    set_seed(seed)
    # strength = 1 - img_guid, so high img_guid = low strength = stays close to original
    results = pipe(prompt=prompt, image=image, strength=1 - img_guid, guidance_scale=text_guid)
    generated = results.images[0]
    generated.save(output_path)
    return generated

def get_image_embedding(processor, model, image: Image.Image, device: str) -> np.ndarray:
    pixel_values = processor(text=None, images=image, return_tensors="pt", do_rescale=True)[
        "pixel_values"
    ].to(device)
    output = model.get_image_features(pixel_values)
    # get_image_features can return a tensor or a BaseModelOutputWithPooling
    if hasattr(output, "pooler_output"):
        embedding = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        embedding = output.last_hidden_state[:, 0, :]
    else:
        embedding = output  # already a tensor
    return embedding.cpu().detach().numpy()

def get_text_embedding(tokenizer, model, text: str, device: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output = model.get_text_features(**inputs)
    if hasattr(output, "pooler_output"):
        embedding = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        embedding = output.last_hidden_state[:, 0, :]
    else:
        embedding = output  # already a tensor
    return embedding.cpu().detach().numpy()

def compute_clip_scores(clip_model, processor, tokenizer, original_image: Image.Image,
    generated_image: Image.Image, prompt: str, device: str) -> tuple[float, float]:
    """Returns (text_similarity, image_similarity) against the original."""
    orig_emb  = get_image_embedding(processor, clip_model, original_image, device)
    gen_emb   = get_image_embedding(processor, clip_model, generated_image, device)
    text_emb  = get_text_embedding(tokenizer, clip_model, prompt, device)
 
    text_sim = float(cosine_similarity(gen_emb, text_emb)[0][0])
    img_sim  = float(cosine_similarity(orig_emb, gen_emb)[0][0])
    return text_sim, img_sim

### Pipeline
def process_location(location: str, image_paths: list[Path], prompt: str,
    pipe, clip_model, processor, tokenizer, output_dir: Path, device: str,) -> list[dict]:
    """
    For each image in a location group, generate all guidance seed variants,
    score them, and return rows for the CSV log.
    """
    rows = []
    location_out = output_dir / location
    location_out.mkdir(parents=True, exist_ok=True)
 
    for img_path in image_paths:
        print(f"\n  [{location}] Processing: {img_path.name}")
        original = Image.open(img_path).convert("RGB") # unnecessary?
        original_resized = original.resize((768, 512)) # Resize to a consistent size — SDXL works best at multiples of 8
 
        for seed in RANDOM_SEEDS:
            for img_guid in IMG_GUIDANCES:
                combined_seed = (hash(img_path.name) ^ seed) % (2**32) # Unique seed per (image, seed, guidance) combination
 
                out_filename = f"{img_path.stem}_guid{int(img_guid*100)}_seed{seed}.jpg"
                out_path = location_out / out_filename
 
                # Skip if already generated (allows resuming interrupted runs)
                if out_path.exists():
                    print(f"Skipping (exists): {out_filename}")
                    generated = Image.open(out_path).convert("RGB")
                else:
                    print(f"Generating: {out_filename}  (img_guid={img_guid}, seed={seed})")
                    generated = generate_variant(
                        pipe=pipe,
                        image=original_resized,
                        prompt=prompt,
                        img_guid=img_guid,
                        text_guid=TEXT_GUIDANCE,
                        seed=combined_seed,
                        output_path=out_path,
                    )
 
                # CLIP scoring
                gen_resized = generated.resize((480, 270))
                orig_resized_clip = original_resized.resize((480, 270))
                text_sim, img_sim = compute_clip_scores(
                    clip_model, processor, tokenizer,
                    orig_resized_clip, gen_resized, prompt, device
                )
 
                rows.append({
                    "location":       location,
                    "source_image":   str(img_path),
                    "generated_path": str(out_path),
                    "prompt":         prompt,
                    "img_guidance":   img_guid,
                    "seed":           seed,
                    "text_sim":       round(text_sim, 4),
                    "img_sim":        round(img_sim, 4),
                    "passes_filter":  img_sim >= CLIP_THRESHOLD,
                })
 
    return rows

### MAIN
def main():
    print("main() called", flush=True)
    parser = argparse.ArgumentParser(description="DisCL hard-sample augmentation for ladder detection")
    parser.add_argument("--input_dir",  type=Path, required=True,
                        help="Root folder with subfolders per location (e.g. hard_samples/concrete/)")
    parser.add_argument("--output_dir", type=Path, default=Path("generated"),
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
            print(f"WARNING: No prompt defined for location '{loc}'. "
                  f"Add it to LOCATION_PROMPTS in generate.py. Skipping.")
 
    # Load models once (outside the loop)
    pipe = load_sd_pipeline(SD_MODEL, device)
    clip_model, processor, tokenizer = load_clip_model(CLIP_MODEL, device)
 
    all_rows = []
 
    for location, image_paths in found_locations.items():
        if location not in LOCATION_PROMPTS:
            continue
        prompt = LOCATION_PROMPTS[location]
        print(f"Location: {location}  |  {len(image_paths)} images  |  Prompt: '{prompt}'")
 
        rows = process_location(
            location=location,
            image_paths=image_paths,
            prompt=prompt,
            pipe=pipe,
            clip_model=clip_model,
            processor=processor,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            device=device,
        )
        all_rows.extend(rows)
 
    csv_path = args.output_dir / "clip_scores.csv"
    fieldnames = ["location", "source_image", "generated_path", "prompt",
                  "img_guidance", "seed", "text_sim", "img_sim", "passes_filter"]
 
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
 
    total     = len(all_rows)
    passing   = sum(1 for r in all_rows if r["passes_filter"])
    print(f"Done. Generated {total} images, {passing} pass CLIP filter (threshold={args.clip_threshold})")
    print(f"CSV log saved to: {csv_path}")
 
 
if __name__ == "__main__":
    main()