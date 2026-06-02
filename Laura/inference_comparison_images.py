print("started")
import os
import re
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--folder1', type=str, required=True)
parser.add_argument('--folder2', type=str, required=True)
parser.add_argument('--output', type=str, default='comparison')
args = parser.parse_args()

folder1 = Path(args.folder1)
folder2 = Path(args.folder2)
output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True)

# Load CSVs
csv1 = pd.read_csv(next(folder1.glob('*.csv')))
csv2 = pd.read_csv(next(folder2.glob('*.csv')))

# Index by filename (strip prefix from image files)
def strip_prefix(name):
    # filenames are like: 1_originalname_bb.png or 0_originalname_bb.png
    # original filename is embedded — extract it
    match = re.match(r'^[^_]+_(.+)_bb\.(png|jpg|jpeg)$', name, re.IGNORECASE)
    return match.group(1) if match else name

# Map original filename -> image path for each folder
def build_map(folder):
    mapping = {}
    for f in folder.iterdir():
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg') and f.name.endswith('_bb.png'):
            key = strip_prefix(f.name)
            mapping[key] = f
    return mapping

map1 = build_map(folder1)
map2 = build_map(folder2)

# Also index CSVs by filename
csv1 = csv1.set_index('filename')
csv2 = csv2.set_index('filename')

common = set(map1.keys()) & set(map2.keys())
print(f"Found {len(common)} matching images")

HEADER_H = 60
GAP = 20
FONT_SIZE = 22

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except:
    font = ImageFont.load_default()
    small_font = font

for key in sorted(common):
    img1 = Image.open(map1[key]).convert('RGB')
    img2 = Image.open(map2[key]).convert('RGB')

    # Resize to same height
    h = max(img1.height, img2.height)
    w1 = int(img1.width * h / img1.height)
    w2 = int(img2.width * h / img2.height)
    img1 = img1.resize((w1, h))
    img2 = img2.resize((w2, h))

    total_w = w1 + w2 + GAP
    total_h = h + HEADER_H * 2

    canvas = Image.new('RGB', (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # Titles
    draw.text((w1 // 2, 10), folder1.name, fill=(255, 255, 255), font=font, anchor='mt')
    draw.text((w1 + GAP + w2 // 2, 10), folder2.name, fill=(255, 255, 255), font=font, anchor='mt')

    # CSV info
    def get_info(csv, key):
        try:
            row = csv.loc[key]
            conf = f"{row['confidence_ladder']:.3f}" if pd.notna(row['confidence_ladder']) else 'none'
            return f"pred={row['predicted_label']}  conf={conf}  correct={row['correct']}"
        except:
            return ''

    info1 = get_info(csv1, key)
    info2 = get_info(csv2, key)
    draw.text((w1 // 2, 40), info1, fill=(200, 200, 200), font=small_font, anchor='mt')
    draw.text((w1 + GAP + w2 // 2, 40), info2, fill=(200, 200, 200), font=small_font, anchor='mt')

    # Paste images
    canvas.paste(img1, (0, HEADER_H))
    canvas.paste(img2, (w1 + GAP, HEADER_H))

    # Image filename at bottom
    draw.text((total_w // 2, h + HEADER_H + 5), key, fill=(180, 180, 180), font=small_font, anchor='mt')

    out_path = output_dir / f"{key}_comparison.png"
    canvas.save(out_path)
    print(f"Saved {out_path}")

print(f"\nDone. {len(common)} comparisons saved to {output_dir}/")