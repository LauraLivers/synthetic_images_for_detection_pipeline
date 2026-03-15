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

## Step 1: Data Extraction & Preprocessing
### Dataset creation
0. download all images with `patrol_images.py`
1. run `thermal_extract.py` to get rid of thermal images
2. manually separate the images into `ladder`/`no_ladder` folders

### Label creation for training
4. annotate all `ladder` images using `label-studio start` and navigate to `http://localhost:8080`

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
7. create .csv file containing all filenames mapped to the label index
```
uv run label_csv.py --input_path [input] --class_index [index]
```
This creates `image_mapped_labels_[class_index].csv` which will be used for the split logic later on.  

6. run `uv run yolo_training.py` to separate the ladder-images into **easy** and **hard** samples. The hard samples will be used for *diffusion curriculum* to make the model more robust for detection.
Verify the results by looking at the heatmap images generated in folder `EigenCAM_heatmaps` and compare to the results.



