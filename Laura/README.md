## Step 0: Dataset Version Control
0. create a remote
```
dvc remote add -d [path/to/remote]
```
[tutorial](https://doc.dvc.org/start)
[cloud storage](https://doc.dvc.org/user-guide/data-management/remote-storage)

1. add dataset
```
dvc add [path/to/data]
```
this automatically creates a `.gitignore` and `[dataset.dvc]` file. This excludes the data itself from being added to git, while ensuring the hash for each version is commited to git for tracking

---

## Step 1: Data Extraction & Preprocessing
### Dataset creation
0. download all images with `patrol_images.py` (don't forget to create .env with API key!)
1. run `thermal_extract.py` to move thermal images into another folder
2. manually separate the images into `ladder`/`no_ladder` folders 

### Label creation for training
4. annotate all `ladder` images using `label-studio start` and navigate to `http://localhost:8080`

---
## Step 2: Easy vs. Hard Samples
### Pretrained Model with class Ladder
[Open Image Database](https://github.com/widemeadows/openimages-dataset?tab=readme-ov-file) contains label [ladder](https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=train&c=%2Fm%2F012w5l)

5. run 
```
yolo task=detect mode=predict model=yolov8m-oiv7.pt source=[path/to/image].jpg conf=0.1
```
to make sure the model contains the label `ladder` and can detect ladders $\rightarrow$ use an obvious image to be sure and keep confidence at $0.1$. This will return the image with bounding boxes in `runs/detect/predict[n]`

![ladder](robo_images/1b81499856df49858d2f837de745cda0_front_5_March_2026_13-02.jpg)

6. find label index in model with 
```
python -c "from ultralytics import YOLO; model = YOLO('yolov8m-oiv7.pt'); print({v: k for k, v in model.names.items() if '[label]' in v})"
```
and add the integer found to `yolo_training.py` as CONFIG_VARIABLE  

### Inference only
7. run `uv run yolo_training.py --inference_only --source [path/to/dataset]` 
![yolo no training](bb_y8m_inference_only/x_738d187488dc44049db2e3fd8f3227c4_front_5_March_2026_12-57.jpg_bb.png)

#### Results
- **total images**: 376
- **ladder**: 160 | 42.5%
- **no ladder**: 216 | 57.5%  

| result | number | comment |
| --- | --- |---|
|positive| 37 | 22 high confidence & mixed, 15 low confidence |
|false positive | 0 |
| negative | 216 ||
| false negative | 123 ||

### with training
![with training](bb_y8m/1_x_738d187488dc44049db2e3fd8f3227c4_front_5_March_2026_12-57.jpg_bb.png)

- **train set**: 263, **validation set**: 56
- **test set**: 57
- **ladder**: 24 | 42.1%
- **no ladder**: 34 | 57.9%

| result | number | comment |
|---|---|---|
|positive| 24 | 12 high confidence (& mixed), 12 low confidence only |
|false positive | 4 | 4 low confidence only |
| negative | 29 ||
| false negative | 0 ||

#### Training finetuning
[yolo best practices](https://docs.ultralytics.com/guides/model-training-tips/)
1. Batch size: [-1] determines batch size based on hardware (does that actually work?)
2. Subset Training: here not too relevant as dataset is very small
3. Multi-Scale Training: simulates objects at different distances by scaling the images
4. Caching: can make training more efficient by storing preprocessed images in memory
5. Mixed Precision Training (f16 ist fast, f32 is moe precise)
6. Learning Rate Schedulers: automatic adjustment or lr during training
7. Early Stopping
8. Optimizer

## Stable Diffusion - Phase 1
1. manually sort images into "locations" 
2. adapt the prompts before running `generate.py` 
[understanding prompt anatomy](https://kontextlora.org/blog/stable-diffusion-prompting-techniques)
- Sandwich Technique: important information at beginning and end
- Negative Space Control: "Negative: blurry, malformed hands, ...."
- Weight and Emphasis: "([attribute]:x)
- Conditional Prompting: "[word | synonym | related concept]"
$\rightarrow$ ComfyUI specific, won't parse in diffuser directly


## Prompt 1
```
LOCATION_PROMPTS = {
    "concrete" : "a photo of a ladder leaning against a concrete wall surrounded with office building in the background",
    "park" : "a photo of a ladder in a park, partially hidden by vegetation",
    "street" : "a phot of a ladder leaning against a fence with a busy street behind, partially occluded"
}
```
### test1 - sd_xl_refiner (og)
image_guidance: [0.5, 0.6, 0.7, 0.8, 0.9], no finetuning
$\rightarrow$ everything below 0.8 is not photorealistic anymore
![test 1](img/test1.png)

### test2 - sd_xl_refiner (og)
image_guidance: [0.8, 0.85, 0.9, 0.95, 0.97], no finetuning (original images)
![test 2](img/test2.png)

### test3 - sd_xl_base 
image_guidance: [0.8, 0.85, 0.9, 0.95]

### test4 - sd_xl_base
image_guidance: [0.5, 0.6, 0.7, 0.8, 0.9]

### test4 - sd_xl_base + image resizing (1024,576)



### Balance text2img & img2img
Balancing means controlling where along that dial each generated sample sits, and ensuring your dataset has coverage across the whole range — not just the two extremes. Concretely:

Denoising strength as a shared axis — img2img denoising strength and text2img guidance scale are both levers on the same underlying spectrum. Treating them as one joint parameter space (rather than two separate modes) lets you interpolate smoothly.
Proportional mixing by curriculum stage — early stages should be img2img-heavy (low noise, close to real data), late stages text2img-heavy (high diversity). If both are weighted equally throughout, the curriculum loses its meaning.
Preventing distribution mismatch — unbalanced mixing means your synthetic data doesn't resemble a coherent augmentation of your real data; it looks like two datasets concatenated. Downstream models trained on this will either ignore the synthetic data or overfit to its artifacts.

A practical framing
Rather than thinking of them as two separate "versions," think of img2img with strength=1.0 as text2img (full noise, no anchor). Your curriculum is then just a schedule over a single strength parameter, and "balancing" means making sure that schedule is intentional and monotonic — not a binary switch between 0.2 and 1.0.
The repo you're basing on probably treats them as separate pipelines because they're separate API calls, but conceptually they should be one continuum. Bridging that gap is exactly the right instinct.


