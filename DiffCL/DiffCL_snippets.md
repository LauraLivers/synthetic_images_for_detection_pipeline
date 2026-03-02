# DiffCL Repository Structure Overview

```
DiffCL/
├── assets/
│   ├── ImageNet-LT.jpg        # sample synthetic images figure for ImageNet-LT (used in README)
│   ├── iWildCam.jpg           # sample synthetic images figure for iWildCam (used in README)
│   ├── overview.png           # pipeline overview figure (used in README)
│   └── metadata/
│       ├── wilds_common_names.pkl   # maps iWildCam species IDs to human-readable names
│       └── Samples/                 # qualitative result images shown in README
├── curriculum_training/
│   ├── __init__.py
│   ├── ImageNet/                    # Phase 2 training pipeline for ImageNet-LT
│   │   ├── __init__.py
│   │   ├── main.py                  # training entry point
│   │   ├── fine_tune_tr.py          # fine-tuning training logic
│   │   ├── config/
│   │   │   └── ImageNet/
│   │   │       ├── ImageNet_LSC_Mixup.txt         # curriculum schedule (text format)
│   │   │       └── ImageNet_LSC_Mixup_rn50.json   # curriculum schedule for ResNet-50
│   │   ├── datacreation_scripts/
│   │   │   └── imagenet_LT.py          # builds the ImageNet-LT dataset split
│   │   ├── dataloader/
│   │   │   ├── __init__.py
│   │   │   ├── Custom_Dataloader.py       # custom dataset class
│   │   │   ├── data_loader_ImageNet.py    # ImageNet-LT dataloader
│   │   │   ├── data_loader_wrapper.py     # wraps dataloaders for curriculum stages
│   │   │   └── sampler.py                 # custom sampler
│   │   ├── model/
│   │   │   ├── __init__.py
│   │   │   ├── NormLayer.py         # normalisation layer
│   │   │   ├── label_shift_est.py   # label shift estimation
│   │   │   ├── losses.py            # loss functions
│   │   │   ├── metrics.py           # training metrics
│   │   │   ├── metrics_eval.py      # evaluation metrics
│   │   │   ├── model_eval.py        # model evaluation logic
│   │   │   └── model_init.py        # model initialisation
│   │   ├── myshells/
│   │   │   └── run_training.sh      # shell script to launch training
│   │   ├── networks/
│   │   │   ├── __init__.py
│   │   │   ├── NormLayer.py         # normalisation layer for network
│   │   │   └── resnet.py            # ResNet backbone definition
│   │   └── utilis/
│   │       ├── __init__.py
│   │       ├── config_parse.py      # parses training config files
│   │       ├── config_range.txt     # valid config value ranges
│   │       ├── feature_encode.py    # feature extraction / encoding
│   │       ├── metric.py            # metric computation utilities
│   │       ├── nn_classifier.py     # nearest-neighbour classifier
│   │       ├── sane_check.py        # sanity checks
│   │       ├── test_ft.py           # fine-tuning test script
│   │       ├── tester_ft.py         # fine-tuning tester class
│   │       ├── utils.py             # general utilities
│   │       └── visual.py            # visualisation helpers
│   └── iWildCam/                    # Phase 2 training pipeline for iWildCam (FLYP / Open CLIP)
│       ├── clip/
│       │   ├── README.md
│       │   ├── bpe_simple_vocab_16e6.txt.gz   # BPE vocabulary for tokeniser
│       │   ├── clip.py                        # CLIP model interface
│       │   ├── loss.py                        # CLIP training loss
│       │   ├── model.py                       # CLIP model architecture
│       │   └── tokenizer.py                   # text tokeniser
│       ├── datacreation_scripts/
│       │   ├── iwildcam.py          # builds iWildCam dataset split
│       │   └── iwildcam_ori.py      # original (unmodified) version of the split builder
│       ├── myshells/
│       │   └── run_training.sh      # shell script to launch training
│       └── src/
│           ├── args.py              # command-line argument definitions
│           ├── logger_utils.py      # logging utilities
│           ├── main.py              # training entry point
│           ├── datasets/
│           │   ├── __init__.py
│           │   ├── common.py                  # shared dataset utilities
│           │   ├── iwildcam.py                # iWildCam dataset class
│           │   ├── laion.py                   # LAION dataset class
│           │   └── iwildcam_metadata/
│           │       └── labels.csv             # iWildCam class label mapping
│           ├── models/
│           │   ├── __init__.py
│           │   ├── common.py        # shared model utilities
│           │   ├── eval.py          # evaluation logic
│           │   ├── modeling.py      # model construction
│           │   ├── training.py      # training loop
│           │   ├── utils.py         # model utilities
│           │   └── zeroshot.py      # zero-shot evaluation
│           └── templates/
│               ├── __init__.py
│               ├── iwildcam_template.py   # zero-shot prompt templates for iWildCam
│               └── utils.py               # template utilities
├── data_generation/
│   ├── DisCL_demo.ipynb             # end-to-end demo: generation + CLIP filtering
│   ├── ImageNet_LT/                 # Phase 1 generation for ImageNet-LT
│   │   ├── comp_clip_scores.py      # computes image–image and image–text CLIP scores; filters results
│   │   ├── gene_img.py              # generates synthetic images via SDXL img2img
│   │   ├── get_text_prompt.py       # creates diversified text prompts for hard classes
│   │   ├── imagenetlt_classes.py    # ImageNet-LT class list
│   │   ├── sample.csv               # template input CSV (hard sample list)
│   │   └── sample_imgs/
│   │       ├── generated.jpg        # example generated image
│   │       └── n02093428_672.JPEG   # example real input image
│   ├── iWildCam/                    # Phase 1 generation for iWildCam
│   │   ├── comp_clip_scores.py      # computes image–image and image–text CLIP scores; filters results
│   │   ├── gene_img.py              # generates synthetic images via SDXL img2img
│   │   ├── sample.csv               # template input CSV (hard sample list)
│   │   └── sample_imgs/
│   │       ├── 8b5d271c-21bc-11ea-a13a-137349068a90.jpg   # example real input image
│   │       └── generated.jpg                               # example generated image
│   └── model/
│       ├── __init__.py
│       └── pipeline_stable_diffusion_xl_img2img.py   # custom SDXL img2img pipeline
├── scripts/
│   ├── data_generation_inlt.sh      # runs full data generation pipeline for ImageNet-LT
│   └── data_generation_wilds.sh     # runs full data generation pipeline for iWildCam
├── README.md                        # project description, results tables, usage instructions
├── requirements.txt                 # Python dependencies
├── requirements-linux-cuda.txt      # Python dependencies for Linux + CUDA
├── setup_env.sh                     # sets up the conda environment
└── video2image.sh                   # extracts frames from a video
```
---

## 1. Dataset Preparation
- **All Images in folder**
CSV or JSON file (e.g. `data/labels.csv`) mapping each image filename to a list of classes:
```
filename,labels
img1.jpg,"building,street"
img2.jpg,"street,tree"
img3.jpg,"car,street,building"
```
- **metadata** file as pickle with human readable class names (`common_names`) $/rightarrow$ .pkl

```python
# Example for buildings/streets
common_names = {
    "buildings": "building",
    "streets": "street"
}
with open('assets/metadata/your_domain_common_names.pkl', 'wb') as f:
    pickle.dump(common_names, f)
```
- `sp_name` as key in `common_names` which goes in prompt e.g. "a phot of $classname in the wild"


## 2. generate synthetic images
[example file](data_generation/DisCL_demo.ipynb) 
- load real images
- use of Stable Diffusion XL Refiner (img2img) to generate variants
- multiple random seeds $\cdot$ multiple guidance values = diverse synthetic images per input

## 3. Filter generated images with CLIP
- CLIP text similarity (generated image $\leftrightarrow$ text prompt)
- CLIP image similarity (generated image $\leftrightarrow$ original image)
$\rightarrow$ only keeps images where $text_sim >= 0.25$ (can be adjusted)

## 4. Train Classifier
- combine real dataset + filtered synthetic images
- training with **Contrastive Learning**
