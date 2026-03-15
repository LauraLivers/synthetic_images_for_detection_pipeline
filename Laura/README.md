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
0. download all images with `patrol_images.py`
1. run `thermal_extract.py` to get rid of thermal images
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
to  make sure the model contains the label `ladder` and can detect ladders $\rightarrow$ use an obvious image to be sure and keep confidence at $0.1$. This will return the image with bounding boxes in `runs/detect/predict[n]`  
6. find label index in model with 
```
python -c "from ultralytics import YOLO; model = YOLO('yolov8m-oiv7.pt'); print({v: k for k, v in model.names.items() if '[label]' in v})"
```
and add to `yolo_training.py` as CONFIG_VARIABLE
### Inference only
7. run `uv run yolo_training.py --inference_only --source [path/to/dataset]` 
![yolo no training](/bounding_boxes_inference_only/1_x_d86efae0818b44668eef0cb38f465ad4_front_5_March_2026_12-56.jpg_bb.png)
#### Results
- **total images**: 376
- **ladder**: 160 | 42.5%
- **no ladder**: 216 | 57.5%  

| result | number | 
| --- | --- |
|positive| 37 |
|false positive | 0 |
| negative | 216 |
| false negative | 123 |

### with training
- **test set**: 57
- **ladder**: 24 | 42.1%
- **no ladder**: 34 | 57.9%

| result | number | 
|---|---|
|positive| 24 |
|false positive | 4 |
| negative | 29 |
| false negative | 0 |