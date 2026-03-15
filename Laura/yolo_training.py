import os
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import wandb
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import torch
import argparse

### Parser
parser = argparse.ArgumentParser()
parser.add_argument('--inference_only', action='store_true')
args = parser.parse_args()


### Config Variables
MODEL_PATH = 'yolov8m-oiv7.pt' 
IMAGE_DIR = 'robo_images'
CLASS_LABEL = 298 # HARDCODED LADDER
LABELS = f'image_mapped_labels_{CLASS_LABEL}' # change if file changes
MODEL_SAVE_PATH = 'best_model.pt'
DATASET_YAML = 'dataset.yaml'
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
# fine tune if needed
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 20
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.5

### Acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# load .env file for wandb API key
load_dotenv()
wandb.init(project='ladder-detection-yolo', config={
    'batch_size' : BATCH_SIZE,
    'learning_rate' :  LEARNING_RATE,
    'epochs' : EPOCHS,
    'random_seed' : RANDOM_SEED,
    'train_size' : TRAIN_SIZE,
    'val_size' : VAL_SIZE,
    'test_size' : TEST_SIZE,
    'mode' : 'inference_only' if args.inference_only else 'train+inference'
})

def inference(model, output_dir='bounding_boxes'):
    """ inference run on pretrained or trained model """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for folder in ['ladder', 'no_ladder']: # HARDCODED for now, change later for better encapsulation
        true_label = 1 if folder == 'ladder' else 0
        for filename in os.listdir(os.path.join(IMAGE_DIR, folder)):
            if not filename.endswith(('.jpg', '.png', '.jpeg')): # ignore .DS_Store files
                continue
            img_path = os.path.join(IMAGE_DIR, folder, filename)

            yolo_result = model.predict(img_path, conf=0.05, verbose=False)[0] # to get bounding boxes no matter what
            ladder_boxes = [box for box in yolo_result.boxes if int(box.cls.item()) == CLASS_LABEL]
            max_conf = max([box.conf.item() for box in ladder_boxes]) if ladder_boxes else None
            predicted_label = 1 if max_conf is not None and max_conf >= CONFIDENCE_THRESHOLD else 0

            # Bounding Boxes
            original_image = cv2.imread(img_path)
            for box in ladder_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = (0, 255, 0) if box.conf.item() >= CONFIDENCE_THRESHOLD else (0, 0, 255) # color code bounding boxes
                cv2.rectangle(original_image, (x1,y1), (x2,y2), color, 2)
                cv2.putText(original_image, f'ladder {box.conf.item():.4f}', (x1,y1 - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, f'{predicted_label}_{filename}_bb.png'), original_image)

            results.append({
                'filename' : filename,
                'true_label' : true_label,
                'confidence_ladder' : max_conf,
                'predicted_label' : predicted_label,
                'correct' : true_label == predicted_label,
                'gradcam_path' : os.path.join(output_dir, f'{filename}_bb_{predicted_label}.png')
            })
    return pd.DataFrame(results) 

if not args.inference_only:
        
    # dataset split
    all_images = []
    for folder in ['ladder', 'no_ladder']: # HARDCODED for now, change later for better encapsulation
        for filename in os.listdir(os.path.join(IMAGE_DIR, folder)):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                all_images.append({
                    'filename': filename,
                    'folder': folder,
                    'ladder': 1 if folder == 'ladder' else 0
                })

    df = pd.DataFrame(all_images)
    train_df, temp_df = train_test_split(df, test_size=(VAL_SIZE + TEST_SIZE), stratify=df['ladder'], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE), stratify=temp_df['ladder'], random_state=RANDOM_SEED)

    for split, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        images_out = os.path.join(LABELS, split, 'images')
        labels_out = os.path.join(LABELS, split, 'labels')
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)
        for _, row in split_df.iterrows():
            dst_img = os.path.join(images_out, row['filename'])
            if not os.path.exists(dst_img):
                os.symlink(os.path.abspath(os.path.join(IMAGE_DIR, row['folder'], row['filename'])), dst_img)
            label_file = os.path.splitext(row['filename'])[0] + '.txt'
            label_src = os.path.abspath(os.path.join(LABELS, 'labels', label_file))
            dst_label = os.path.join(labels_out, label_file)
            if os.path.exists(label_src) and not os.path.exists(dst_label):
                os.symlink(label_src, dst_label)

    with open(DATASET_YAML, 'w') as f:
        f.write(f'path: {os.path.abspath(LABELS)}\n')
        f.write('train: train/images\n')
        f.write('val: val/images\n')
        f.write('test: test/images\n')
        f.write('nc: 1\n')
        f.write('names:\n  0: ladder\n')


    def on_fit_epoch_end(trainer):
        wandb.log({
            'epoch': trainer.epoch,
            'train_box_loss': trainer.loss_items[0].item(),
            'train_cls_loss': trainer.loss_items[1].item(),
            'train_dfl_loss': trainer.loss_items[2].item(),
            'val_mAP50': trainer.metrics.get('metrics/mAP50(B)', 0),
            'val_mAP50-95': trainer.metrics.get('metrics/mAP50-95(B)', 0),
            'val_precision': trainer.metrics.get('metrics/precision(B)', 0),
            'val_recall': trainer.metrics.get('metrics/recall(B)', 0),
        })


    model = MODEL_PATH
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
    train_results = model.train(
        data='dataset.yaml',
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        seed=RANDOM_SEED,
        save=True,
        project='ladder-detection-yolo',
        name='train',
        device=device,
        close_mosaic=0,
        warmup_epochs=1
    )


    model = YOLO(train_results.save_dir / 'weights/best.pt')

    metrics = model.val()
    wandb.log({
        'mAP50' : metrics.box.map50,
        'mAP50-95' : metrics.box.map,
        'precision' : metrics.box.mp,
        'recall' : metrics.box.mr
    })
else:
    model = YOLO(MODEL_PATH)

  
inference_df = inference(model)
detected = inference_df[inference_df['confidence_ladder'] >= CONFIDENCE_THRESHOLD]
hard_samples = inference_df[inference_df['confidence_ladder'].notna() & (inference_df['confidence_ladder'] < CONFIDENCE_THRESHOLD)]
no_detection = inference_df[inference_df['confidence_ladder'].isna()]

detected.to_csv('detected.csv', index=False)
hard_samples.to_csv('hard_samples.csv', index=False)
no_detection.to_csv('no_detection.csv', index=False)

wandb.log({
    'num_detected': len(detected),
    'num_hard_samples': len(hard_samples),
    'num_no_detection': len(no_detection)
})
wandb.finish()