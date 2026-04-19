"""
modeled after the original MoBI pipeline [url]
NuScene Replacement
  - image_path       : source RGB image (contains the ladder)
  - mask_path        : .npy boolean mask of the ladder (H x W)
  - corners3d_path   : .npy (8, 3) 3-D bounding box corners
  - depth_mean       : mean depth of the ladder (metres)
  - sam2_score       : detection confidence (used for filtering)

training strategy
 - reference image  : ladder crop from the *same* image, augmented
  - background       : same image with the ladder region masked out
  - inpaint mask     : dilated version of the segmentation mask
  - bbox conditioning: 2-D image coords of the 8 corners (normalised)
"""

from __future__ import annotations
 
import os
import copy
import random
import warnings
 
import cv2
import numpy as np
import pandas as pd
from PIL import Image
 
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as T
import albumentations as A # image augmentation framework

# tensor helpers
def get_tensor(normalize=True, toTensor=True):
	transforms = []
	if toTensor:
		transforms.append(torchvision.transforms.ToTensor())
	if normalize:
		transforms.append(torchvision.transforms.Normalize(
			(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
	return torchvision.transforms.Compose(transforms)

def get_tensor_clip(normalize=True, toTensor=True):
	transforms = []
	if toTensor:
		transforms.append(torchvision.transforms.ToTensor())
	if normalize:
		transforms.append(torchvision.transforms.Normalize(
			(0.48145466, 0.4578275, 0.40821073),
			(0.26862954, 0.26130258, 0.27577711)))
	return torchvision.transforms.Compose(transforms)

# mask helpers
def dilate_mask(mask: np.ndarray, ratio: float = 0.1) -> np.ndarray:
	"""Expand a binary mask by `ratio` of its bounding-box size."""
	ys, xs = np.where(mask)
	if len(ys) == 0:
		return mask
	h = ys.max() - ys.min()
	w = xs.max() - xs.min()
	ksize = max(1, int(ratio * max(h, w)))
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize * 2 + 1, ksize * 2 + 1))
	return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

def mask_to_bbox(mask: np.ndarray):
	"""Return (x1, y1, x2, y2) tight bounding box of a binary mask."""
	ys, xs = np.where(mask)
	if len(ys) == 0:
		return None
	return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

# Dataset Class
class LadderDataset(data.Dataset):
	"""
	csv_path : str - Path to your ladder segmentation CSV.
	state : str - 'train' or 'val'.
	image_height, image_width : int - Output crop resolution (should match config, default 512).
	expand_mask_ratio : float - How much to dilate the segmentation mask for the inpainting region.
	ref_aug : bool - Whether to apply augmentation to the reference crop.
	object_area_crop : float - Fraction of the image area occupied by the object; controls crop size.
	num_samples_per_epoch : int | None - Virtual epoch length. If None, equals len(csv).
	min_sam2_score : float - Minimum SAM2 confidence to include a sample.
	val_split : float - Fraction of data used for validation (stratified by row order).
	fx, fy : float | None - Camera focal lengths in pixels. If None, falls back to W*0.8 / H*0.8
	cx, cy : float | None - Principal point in pixels. If None, defaults to W/2, H/2.
	"""
 
	def __init__(self, csv_path: str, state: str = "train", image_height: int = 512,
		image_width: int = 512, expand_mask_ratio: float = 0.1, ref_aug: bool = True,
		object_area_crop: float = 0.2, num_samples_per_epoch: int | None = None,
		min_sam2_score: float = 0.5, val_split: float = 0.15,
		fx: float | None = None, fy: float | None = None,
		cx: float | None = None, cy: float | None = None, **kwargs,   # absorb unused yaml keys gracefully
	):
		self.state = state
		self.image_height = image_height
		self.image_width = image_width
		self.expand_mask_ratio = expand_mask_ratio
		self.ref_aug = ref_aug
		self.object_area_crop = object_area_crop
 
		df = pd.read_csv(csv_path, index_col=0)
		df = df[df["sam2_score"] >= min_sam2_score].reset_index(drop=True)
 
		# deterministic train / val split
		n_val = max(1, int(len(df) * val_split))
		if state == "val":
			df = df.iloc[:n_val]
		else:
			df = df.iloc[n_val:]
		df = df.reset_index(drop=True)
 
		self.df = df
		n = len(df)
		self.num_samples_per_epoch = num_samples_per_epoch if num_samples_per_epoch else n
		print(f"[LadderDataset] state={state}  rows={n}  epoch_len={self.num_samples_per_epoch}")
 
		# reference image augmentation
		self.ref_transform = A.Compose([
			A.Resize(height=224, width=224),
			A.HorizontalFlip(p=0.5),
			A.Rotate(limit=15, p=0.5),
			A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
			A.GaussianBlur(blur_limit=3, p=0.2),
		])
 
		# background / scene augmentation 
		self.scene_transform = A.Compose([
			A.HorizontalFlip(p=0.5),
			A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, p=0.4),
		])
 
		# camera intrinsics (None = derive from image size at load time) ----
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
 
		# output resize 
		self.image_resize = T.Resize([image_height, image_width], antialias=True)
		self.mask_resize  = T.Resize([image_height, image_width],
									  interpolation=T.InterpolationMode.NEAREST)
	def __len__(self):
		return self.num_samples_per_epoch
	
	def __getitem__(self, index):
		# random resampling for small dataset
		row = self.df.iloc[index % len(self.df)]
		return self._load_sample(row)
	
	def _load_sample(self, row):
		# raw image 
		img_bgr = cv2.imread(row["image_path"])
		if img_bgr is None:
			raise FileNotFoundError(f"Image not found: {row['image_path']}")
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		H, W = img_rgb.shape[:2]
 
		# segmentation mask 
		mask_np = np.load(row["mask_path"]) # bool or 0/1, shape (H, W)
		mask_np = mask_np.astype(bool)
 
		# 3-D bounding box corners 
		corners3d = np.load(row["corners3d_path"]) # (8, 3) in camera/metric space
 
		# optional scene augmentation (applied consistently to img + mask) 
		if self.state == "train":
			aug = self.scene_transform(image=img_rgb, mask=mask_np.astype(np.uint8))
			img_rgb  = aug["image"]
			mask_np  = aug["mask"].astype(bool)
 
		# 1. Reference crop — ladder region cropped + augmented
		bbox = mask_to_bbox(mask_np)
		if bbox is None:
			bbox = (0, 0, W - 1, H - 1)
		x1, y1, x2, y2 = bbox
 
		# small padding around the mask bbox
		pad = 10
		rx1 = max(0, x1 - pad)
		ry1 = max(0, y1 - pad)
		rx2 = min(W, x2 + pad)
		ry2 = min(H, y2 + pad)
 
		ref_crop = img_rgb[ry1:ry2, rx1:rx2]
 
		if self.ref_aug and self.state == "train":
			ref_crop = self.ref_transform(image=ref_crop)["image"]
		else:
			ref_crop = cv2.resize(ref_crop, (224, 224))
 
		ref_tensor = get_tensor_clip()(Image.fromarray(ref_crop))   # (3, 224, 224)
 
		# 2. Inpainting mask — dilated segmentation mask
		inpaint_mask_np = dilate_mask(mask_np, ratio=self.expand_mask_ratio)  # True = KEEP (background)
		# convention in this codebase: mask==1 : keep background, 0 : inpaint region
		inpaint_mask_bin = (~inpaint_mask_np).astype(np.float32)  # 1 = region to inpaint
 
		# 3. Crop the scene so the ladder occupies ~object_area_crop of area
		target_area = (x2 - x1) * (y2 - y1) / self.object_area_crop
		crop_side = int(np.sqrt(target_area))
		crop_H = min(crop_side, H)
		crop_W = min(crop_side, W)
 
		# ensure the bbox fits inside the crop
		crop_H = max(crop_H, y2 - y1)
		crop_W = max(crop_W, x2 - x1)
		crop_H = min(crop_H, H)
		crop_W = min(crop_W, W)
 
		# random crop placement that keeps the ladder inside
		try:
			left = random.randint(max(0, x2 - crop_W), min(x1, W - crop_W)) if self.state == "train" \
				   else max(0, (x1 + x2) // 2 - crop_W // 2)
			top  = random.randint(max(0, y2 - crop_H), min(y1, H - crop_H)) if self.state == "train" \
				   else max(0, (y1 + y2) // 2 - crop_H // 2)
		except ValueError:
			left, top = 0, 0
 
		left = int(np.clip(left, 0, W - crop_W))
		top  = int(np.clip(top,  0, H - crop_H))
 
		# 4. Build bbox image coordinates (normalised, for conditioning)
		# Project 3-D corners to 2-D pixel space.
		# Simple pinhole projection — approximate; fine for conditioning signal.
		bbox_image_coords = self._project_corners(corners3d, H, W)  # uses self.fx/fy/cx/cy
 
		# Shift by crop offset and normalise
		bbox_image_coords = bbox_image_coords - np.array([left, top, 0])
		bbox_image_coords[..., 0] = np.clip(bbox_image_coords[..., 0] / crop_W, 0, 1)
		bbox_image_coords[..., 1] = np.clip(bbox_image_coords[..., 1] / crop_H, 0, 1)
		# depth: normalise by max expected depth (e.g. 50 m) → [-1, 1]
		bbox_image_coords[..., 2] = np.clip(bbox_image_coords[..., 2] / 25.0 - 1.0, -1, 1)
		bbox_image_coords = torch.tensor(bbox_image_coords, dtype=torch.float32)
 
		# 5. Apply crop to image and mask, then resize to model input size
		img_crop  = img_rgb[top:top+crop_H, left:left+crop_W]
		mask_crop = inpaint_mask_bin[top:top+crop_H, left:left+crop_W]
 
		# to tensor
		image_tensor = get_tensor()(Image.fromarray(img_crop)) # (3, H, W) in [-1,1]
		mask_tensor  = torch.tensor(mask_crop).unsqueeze(0).float() # (1, H, W)
 
		image_tensor = self.image_resize(image_tensor)
		mask_tensor  = self.mask_resize(mask_tensor)
 
		# background visible where mask==0, erased where mask==1
		inpaint_tensor = image_tensor * (1 - mask_tensor)
 
		return {
			"image": {
				"GT": image_tensor,
				"inpaint_image": inpaint_tensor,
				"inpaint_mask": mask_tensor,
				"cond": {
					"ref_image": ref_tensor,
                    "ref_bbox": bbox_image_coords,
				},
			},
			"lidar": {},
			#"ref_img": ref_tensor.unsqueeze(0),
			"file_name": os.path.basename(row["image_path"]),
		}
	
	def _project_corners(self, corners3d: np.ndarray, H: int, W: int) -> np.ndarray:
		"""
		Pinhole projection of 8 corners (camera frame) to pixel coords.
		Returns (8, 3) array: [u, v, depth_m].
		corners3d[:,0] = X (right), [:,1] = Y (down), [:,2] = Z (forward/depth).
		"""
		fx = self.fx if self.fx is not None else W * 0.8
		fy = self.fy if self.fy is not None else H * 0.8
		cx = self.cx if self.cx is not None else W / 2.0
		cy = self.cy if self.cy is not None else H / 2.0
 
		out = np.zeros((len(corners3d), 3), dtype=np.float64)
		for i, (X, Y, Z) in enumerate(corners3d):
			Z = max(Z, 0.1)   # avoid division by zero
			u = fx * X / Z + cx
			v = fy * Y / Z + cy
			out[i] = [u, v, Z]
		return out
	
# subclass so yaml target strings stay short
class LadderDatasetTrain(LadderDataset):
	def __init__(self, **kwargs):
		super().__init__(state="train", **kwargs)
 
 
class LadderDatasetVal(LadderDataset):
	def __init__(self, **kwargs):
		super().__init__(state="val", **kwargs)
 