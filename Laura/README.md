# Synthetic Images for Rare-Event Detection

> Work in progress. Do not use for commercial purposes. Do not use in your own name.  
> For model weights, contact: `hello@lauralivers.com`

## What this is
A pipeline that generates synthetic training data for rare-event object detection using object inpainting ![MobI](https://github.com/alexbuburuzan/MObI/blob/main/README.md). Developed for ladder detection on the Ascento Guard robot, but adaptable to other object classes by adapting the placement logic.
This project used different combinations of synthetic images and classically augmented to keep the distribution intact and to focuson on optimizing mAP@50 while keeping an eye on recall and false negative rate. Interestingly, base dataset and synthetic dataset performed close, but synthetic datasets were much more robust on precision and recall. Depending on the detections usecase going for maximum map@50 can be enough, in the original case it made sense chosing recall over map@50. Each file contains visualization output logic, which means this repo can get cluttered after a while. 

## Setup
Install [uv](https://github.com/astral-sh/uv), then:

```bash
uv sync
```
In general, make sure you have your `wandb`, and `huggingface_cli` keys ready. 

## Pipeline

### 1. Add images
Add patrol images to `robo_images/`, and sort into usecase/no usecase. Currently the code only takes a singular use-case. Expanding this should be farely easy, but will introduce chaos if not carefuly. 

### 2. Label data
- Annotate images using `label-studio start` → `http://localhost:8080`
- Export in `YOLO with images` format
- Run `label_studio_annotations.py` to convert
- Add to `robo_images/` as folder named `yolo_annotations/`

### 3. Generate spatial information
Ensure SAM2, Depth Anything, SegFormer are downloaded. Depending on backend this can be rather strenuous. Ensure their folders are in the root folder of this repo, or everything will crash. 
```bash
uv run ladder_segmentation.py # make sure to chose correct output name with config variable
uv run background_surfaces.py # make sure to chose correct output name with config variable
```

### 4. Run inpainting training
- Ensure 
```bash
uv run ladder_inpainting.sh
```
Thanks to the MObI authors for the base implementation. It was a doozy to get it up an running. 
Hint: with all the layers frozen an A16 is enough, with unfrozen layers use A100 or you'll perish waiting for training to end
### 5. Run inpainting inference
```bash
uv run infer.py
```
make sure to change the config variables at the top accorind to you results and naming convention. 

### 6. Consolidate and split data into sets
```bash
uv run consolidate_synthetic_images.py # run anyways, even if only 1 checkpoint was run.
mkdir selected_synthetic_images
```
Visually inspect the content of `consolidated_synthetic_images` and transfer suitable images into `selected_synthetic_images` 
```
uv run split_base_robo_image.py # for different synthetic ratios change list at the end of this file
```

### 7. Train
```bash
uv run yolo_training.py --datasets_dir [path/to/datasets] 
```

### 8. Inference
```bash
uv run yolo_training.py --inference_only --weights [path/to/best.pt] --source [path/to/images]
```

### 9. Compare results side by side
```bash
uv run inference_comparison_images.py --folder1 [path1] --folder2 [path2]
```

## Experiment tracking
[Weights & Biases](https://wandb.ai/llivers-hochschule-luzern/ladder-detection-optuna)
