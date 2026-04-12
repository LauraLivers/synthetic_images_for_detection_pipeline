"""
debug_depth_overlay.py
======================
Step-by-step diagnostic for DepthAnythingV2 output.

For every image it produces ONE debug image with four panels written directly
on the frame — no CSVs, no guessing:

  Panel 1 — Raw depth map (plasma colormap) with the actual float values
             sampled on a regular grid printed on top.
  Panel 2 — Depth histogram so you can see whether the distribution is
             linear, log-binned, bimodal, etc.
  Panel 3 — Depth map with depth VALUE printed at the centroid of every
             SegFormer object class that was detected (so you can see which
             depth layer each semantic class lives in).
  Panel 4 — Original image with each YOLO box drawn + the median raw depth
             inside that box printed in yellow.

Nothing is converted to metres yet.  The whole point is to see exactly what
numbers come out of the model before any arithmetic is applied.

Run:
    python debug_depth_overlay.py
"""

import os, sys, warnings
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image as PILImage

warnings.filterwarnings("ignore")

IMAGES_DIR           = "./robo_images/ladder"
LABELS_DIR           = "./robo_images/yolo_annotations/labels"
OUTPUT_DIR           = "./MoBI_outputs/depth_debug"
SAM2_REPO            = "./sam2"
DEPTH_ANYTHING_REPO  = "./Depth-Anything-V2"
DEPTH_CHECKPOINT     = "./Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth"
SEGFORMER_MODEL      = "nvidia/segformer-b2-finetuned-ade-512-512"

_sam2_abs  = os.path.abspath(SAM2_REPO)
_depth_abs = os.path.abspath(DEPTH_ANYTHING_REPO)
sys.path.insert(0, _sam2_abs)
sys.path.insert(0, _depth_abs)

from depth_anything_v2.dpt import DepthAnythingV2
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


def get_device():
    if torch.cuda.is_available():    return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_depth_model(checkpoint, device):
    model = DepthAnythingV2(encoder="vitb", features=128, out_channels=[96, 192, 384, 768])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def infer_raw_depth(model, image_rgb):
    """
    Return the RAW output of DepthAnythingV2 with NO normalisation applied.
    The model internally calls infer_image which already returns a float32 HxW
    array, but let's call it through the forward pass so we can inspect what
    comes out before any post-processing.

    DepthAnythingV2.infer_image() does:
        1. resize + normalise image
        2. forward pass  →  raw_depth  (HxW float32)
        3. resize back to original size
    It does NOT normalise to [0,1] — that's done in the original script.
    So we call infer_image and capture the truly raw values.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    with torch.no_grad():
        depth_raw = model.infer_image(image_bgr)   # HxW float32, raw disparity
    return depth_raw.astype(np.float32)


def load_yolo_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 or int(parts[0]) != 0:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            x1 = int(max(0, (cx - w/2) * img_w))
            y1 = int(max(0, (cy - h/2) * img_h))
            x2 = int(min(img_w, (cx + w/2) * img_w))
            y2 = int(min(img_h, (cy + h/2) * img_h))
            boxes.append([x1, y1, x2, y2])
    return boxes


def load_segformer(model_name, device):
    processor = SegformerImageProcessor.from_pretrained(
        model_name, size={"height": 1080, "width": 1920}
    )
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    return processor, model.to(device).eval()


def run_segformer(processor, model, image_rgb, device):
    inputs = processor(images=PILImage.fromarray(image_rgb), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image_rgb.shape[:2], mode="bilinear", align_corners=False
    )
    seg_map    = upsampled.squeeze(0).argmax(0).cpu().numpy()
    confidence = torch.softmax(upsampled.squeeze(0), dim=0).max(0).values.cpu().numpy()
    return seg_map, confidence, model.config.id2label


def write_text(ax, x, y, text, fontsize=7, color="white", bg="black"):
    ax.text(x, y, text, fontsize=fontsize, color=color, ha="center", va="center",
            bbox=dict(facecolor=bg, alpha=0.6, pad=1, edgecolor="none"))


def make_debug_image(image_rgb, depth_raw, seg_map, confidence, id2label, boxes, out_path):
    H, W = image_rgb.shape[:2]

    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    fig.patch.set_facecolor("#111")

    d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
    d_range = d_max - d_min if d_max - d_min > 1e-6 else 1.0

    # Panel 1: raw depth map + sampled values on grid 
    ax = axes[0]
    ax.imshow(depth_raw, cmap="plasma", vmin=d_min, vmax=d_max)
    ax.set_title(
        f"Raw depth  min={d_min:.4f}  max={d_max:.4f}\n"
        f"(higher = closer in DepthAnything disparity space)",
        color="#ccc", fontsize=8
    )
    ax.set_facecolor("#111")

    # Print sampled depth values on a 10×6 grid so we can read actual numbers
    step_x = W // 11
    step_y = H // 7
    for gy in range(1, 7):
        for gx in range(1, 11):
            px, py = gx * step_x, gy * step_y
            val = depth_raw[py, px]
            # colour the text by relative depth so it's easy to read
            norm_val = (val - d_min) / d_range
            txt_color = "white" if norm_val < 0.6 else "black"
            write_text(ax, px, py, f"{val:.3f}", fontsize=6, color=txt_color)

    ax.axis("off")

    # Panel 2: depth histogram 
    ax = axes[1]
    ax.set_facecolor("#1a1a1a")
    flat = depth_raw.flatten()
    ax.hist(flat, bins=128, color="#4fc3f7", edgecolor="none", log=True)
    ax.set_title(
        "Depth value histogram (log y-scale)\n"
        "Shape tells you: linear / log / binned / bimodal",
        color="#ccc", fontsize=8
    )
    ax.set_xlabel("Raw depth value", color="#aaa", fontsize=8)
    ax.set_ylabel("Pixel count (log)", color="#aaa", fontsize=8)
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Mark percentile lines
    for pct in [10, 25, 50, 75, 90]:
        v = np.percentile(flat, pct)
        ax.axvline(v, color="yellow", linewidth=0.8, linestyle="--")
        ax.text(v, ax.get_ylim()[1] * 0.5, f"p{pct}\n{v:.3f}",
                color="yellow", fontsize=6, ha="center")

    # Panel 3: semantic classes with their median depth printed at centroid ─
    ax = axes[2]
    ax.imshow(depth_raw, cmap="plasma", alpha=0.5, vmin=d_min, vmax=d_max)

    # Overlay coloured semantic regions + print label + depth at centroid
    overlay = np.zeros((*depth_raw.shape, 4), dtype=np.float32)
    present_classes = np.unique(seg_map)
    cmap = plt.cm.tab20

    annotations = []   # collect so we can print them sorted by depth
    for cls_id in present_classes:
        mask = (seg_map == cls_id) & (confidence > 0.5)
        if mask.sum() < 100:
            continue
        color = cmap(cls_id % 20)[:3]
        overlay[mask, :3] = color
        overlay[mask,  3] = 0.45

        ys, xs = np.where(mask)
        cx_px, cy_px = int(xs.mean()), int(ys.mean())
        median_depth  = float(np.median(depth_raw[mask]))
        mean_depth    = float(np.mean(depth_raw[mask]))
        label         = id2label.get(cls_id, str(cls_id))
        annotations.append((cx_px, cy_px, label, median_depth, mean_depth, color))

    ax.imshow(overlay)

    for (cx_px, cy_px, label, med, mean, color) in annotations:
        norm_med = (med - d_min) / d_range
        write_text(
            ax, cx_px, cy_px,
            f"{label}\nmed={med:.4f}\nmean={mean:.4f}",
            fontsize=6.5, color="white"
        )

    ax.set_title(
        "SegFormer classes → median raw depth at centroid\n"
        "Verify: is 'closer' object getting higher depth value?",
        color="#ccc", fontsize=8
    )
    ax.axis("off")
    ax.set_facecolor("#111")

    # Panel 4: original image + YOLO boxes with depth stats inside 
    ax = axes[3]
    ax.imshow(image_rgb)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        box_depth = depth_raw[y1:y2, x1:x2]
        if box_depth.size == 0:
            continue
        med  = float(np.median(box_depth))
        mean = float(np.mean(box_depth))
        lo   = float(np.percentile(box_depth, 5))
        hi   = float(np.percentile(box_depth, 95))

        ax.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor="yellow", facecolor="none"
        ))
        # Print depth stats just above the box
        write_text(
            ax, (x1+x2)//2, max(y1-30, 20),
            f"LADDER {i}\nmed={med:.4f}  mean={mean:.4f}\np5={lo:.4f}  p95={hi:.4f}",
            fontsize=7, color="yellow"
        )

    ax.set_title(
        "YOLO boxes with raw depth stats inside each box\n"
        "(all values are raw DepthAnything output, no normalisation)",
        color="#ccc", fontsize=8
    )
    ax.axis("off")
    ax.set_facecolor("#111")

    plt.tight_layout(pad=0.5)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# raw depth with SegFormer class depth profile ──

def make_depth_profile_image(image_rgb, depth_raw, seg_map, confidence, id2label, out_path):
    """
    Horizontal strip image: for each detected semantic class print a small
    colour-coded strip showing the distribution of raw depth values inside
    that class mask. This directly answers 'which depth layer = which object'.
    """
    d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
    present_classes = np.unique(seg_map)

    class_stats = []
    for cls_id in present_classes:
        mask = (seg_map == cls_id) & (confidence > 0.5)
        if mask.sum() < 100:
            continue
        vals = depth_raw[mask]
        class_stats.append({
            "cls_id": cls_id,
            "label":  id2label.get(cls_id, str(cls_id)),
            "median": float(np.median(vals)),
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals)),
            "p5":     float(np.percentile(vals, 5)),
            "p95":    float(np.percentile(vals, 95)),
            "count":  int(mask.sum()),
        })

    # Sort by median depth so we can read near→far
    class_stats.sort(key=lambda x: x["median"], reverse=True)  # higher = closer in disparity

    n = len(class_stats)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(14, max(4, n * 0.55)))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")

    cmap = plt.cm.tab20
    yticks, ylabels = [], []

    for row_i, cs in enumerate(class_stats):
        color = cmap(cs["cls_id"] % 20)
        y = n - row_i  # top = closest

        # Draw a horizontal bar spanning p5→p95
        ax.barh(y, cs["p95"] - cs["p5"], left=cs["p5"],
                height=0.6, color=color, alpha=0.7)
        # Mark the median
        ax.plot([cs["median"], cs["median"]], [y-0.3, y+0.3], "w-", linewidth=2)
        # Print values
        ax.text(d_max + (d_max - d_min) * 0.01, y,
                f"  med={cs['median']:.4f}  std={cs['std']:.4f}  n={cs['count']}",
                va="center", color="#ddd", fontsize=8)

        yticks.append(y)
        ylabels.append(cs["label"])

    ax.set_xlim(d_min, d_max + (d_max - d_min) * 0.4)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, color="#ddd", fontsize=9)
    ax.tick_params(axis="x", colors="#aaa")
    ax.set_xlabel("Raw DepthAnything value  (higher ≈ closer / more disparity)", color="#aaa")
    ax.set_title(
        "Depth profile per semantic class  —  sorted closest→farthest\n"
        "Bar = p5→p95 range,  white tick = median",
        color="#ccc", fontsize=10
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def main():
    device = get_device()
    print(f"Device: {device}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DepthAnythingV2...")
    depth_model = load_depth_model(DEPTH_CHECKPOINT, device)

    print("Loading SegFormer...")
    seg_processor, seg_model = load_segformer(SEGFORMER_MODEL, device)
    id2label = seg_model.config.id2label

    image_paths = sorted(Path(IMAGES_DIR).glob("*.*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not image_paths:
        raise FileNotFoundError(f"No images in {IMAGES_DIR}")
    print(f"Found {len(image_paths)} images\n")

    for img_path in image_paths:
        print(f"Processing {img_path.name} ...")
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  Could not read {img_path}, skipping.")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_rgb.shape[:2]

        label_path = Path(LABELS_DIR) / f"{img_path.stem}.txt"
        boxes = load_yolo_boxes(str(label_path), img_w, img_h)

        # 1. Raw depth (NO normalisation) 
        depth_raw = infer_raw_depth(depth_model, image_rgb)
        print(f"  Depth raw  min={depth_raw.min():.5f}  max={depth_raw.max():.5f}  "
              f"mean={depth_raw.mean():.5f}  dtype={depth_raw.dtype}")

        # 2. Segmentation 
        seg_map, confidence, _ = run_segformer(seg_processor, seg_model, image_rgb, device)

        # 3. Debug panels written onto image 
        out1 = out_dir / f"{img_path.stem}_depth_overlay.jpg"
        make_debug_image(image_rgb, depth_raw, seg_map, confidence, id2label, boxes, out1)

        # 4. Per-class depth profile chart 
        out2 = out_dir / f"{img_path.stem}_depth_profile.jpg"
        make_depth_profile_image(image_rgb, depth_raw, seg_map, confidence, id2label, out2)

    print("\nDone.  Check", OUTPUT_DIR)


if __name__ == "__main__":
    main()