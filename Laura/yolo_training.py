import os
import cv2
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
parser.add_argument('--source', type=str, default=None)
args = parser.parse_args()

if args.inference_only and args.source is None:
    parser.error('--source is required with --inference_only')

### Config Variables
MODEL_PATH = 'yolov8m-oiv7.pt'
IMAGE_DIR = 'robo_images'
CLASS_LABEL = 298 # HARDCODED for ladder
ANNOTATIONS_DIR = 'robo_images/yolo_annotations'
MODEL_SAVE_PATH = 'best_model.pt'
DATASET_YAML = 'dataset.yaml'
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 10
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.5

### Acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()
wandb.init(project='ladder-detection-yolo', config={
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'epochs': EPOCHS,
    'random_seed': RANDOM_SEED,
    'train_size': TRAIN_SIZE,
    'val_size': VAL_SIZE,
    'test_size': TEST_SIZE,
    'mode': 'inference_only' if args.inference_only else 'train+inference'
})

def inference(model, source_dir, active_class_label, output_dir='bounding_boxes'):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for filename in os.listdir(source_dir):
        if not filename.endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.realpath(os.path.join(source_dir, filename))
        annotation = os.path.join(ANNOTATIONS_DIR, 'labels', os.path.splitext(filename)[0] + '.txt')
        true_label = 1 if os.path.exists(annotation) and os.path.getsize(annotation) > 0 else 0

        yolo_result = model.predict(img_path, conf=0.1, verbose=False)[0]
        ladder_boxes = [box for box in yolo_result.boxes if int(box.cls.item()) == active_class_label]
        max_conf = max([box.conf.item() for box in ladder_boxes]) if ladder_boxes else None

        original_image = cv2.imread(img_path)
        for box in ladder_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = (0, 255, 0) if box.conf.item() >= CONFIDENCE_THRESHOLD else (0, 0, 255)
            cv2.rectangle(original_image, (x1,y1), (x2,y2), color, 2)
            cv2.putText(original_image, f'ladder {box.conf.item():.4f}', (x1,y1-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

        has_confident = any(box.conf.item() >= CONFIDENCE_THRESHOLD for box in ladder_boxes)
        has_low_conf = any(box.conf.item() < CONFIDENCE_THRESHOLD for box in ladder_boxes)

        if not ladder_boxes:
            predicted_label = '0'
        elif has_confident and has_low_conf:
            predicted_label = '1_x'
        elif has_confident:
            predicted_label = '1'
        else:
            predicted_label = 'x'

        cv2.imwrite(os.path.join(output_dir, f'{predicted_label}_{filename}_bb.png'), original_image)
        results.append({
            'filename': filename,
            'true_label': true_label,
            'confidence_ladder': max_conf,
            'predicted_label': predicted_label,
            'correct': true_label == (1 if predicted_label in ['1', '1_x'] else 0),
            'detection_path': os.path.join(output_dir, f'{predicted_label}_{filename}_bb.png')
        })
    return pd.DataFrame(results)

if not args.inference_only:
    all_images = []
    for folder in ['ladder', 'no_ladder']:
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
        images_out = os.path.join(ANNOTATIONS_DIR, split, 'images')
        labels_out = os.path.join(ANNOTATIONS_DIR, split, 'labels')
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)
        for _, row in split_df.iterrows():
            dst_img = os.path.join(images_out, row['filename'])
            if not os.path.exists(dst_img):
                os.symlink(os.path.abspath(os.path.join(IMAGE_DIR, row['folder'], row['filename'])), dst_img)
            label_file = os.path.splitext(row['filename'])[0] + '.txt'
            label_src = os.path.abspath(os.path.join(ANNOTATIONS_DIR, 'labels', label_file))
            dst_label = os.path.join(labels_out, label_file)
            if os.path.exists(label_src) and not os.path.exists(dst_label):
                with open(label_src, 'r') as f:
                    lines = f.readlines()
                with open(dst_label, 'w') as f:
                    for line in lines:
                        f.write(line)

    with open(DATASET_YAML, 'w') as f:
        f.write(f'path: {os.path.abspath(ANNOTATIONS_DIR)}\n')
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

    model = YOLO(MODEL_PATH)
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
    train_results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        seed=RANDOM_SEED,
        save=True,
        project='ladder-detection-yolo',
        name='train',
        device=device,
        close_mosaic=0,
        warmup_epochs=1,
        freeze=22
    )

    model = YOLO(train_results.save_dir / 'weights/best.pt')
    model.save(MODEL_SAVE_PATH)
    model = YOLO(MODEL_SAVE_PATH)

    val_metrics = model.val(data=DATASET_YAML, split='val')
    wandb.log({
        'val_mAP50': val_metrics.box.map50,
        'val_mAP50-95': val_metrics.box.map,
        'val_precision': val_metrics.box.mp,
        'val_recall': val_metrics.box.mr
    })

    test_metrics = model.val(data=DATASET_YAML, split='test')
    wandb.log({
        'test_mAP50': test_metrics.box.map50,
        'test_mAP50-95': test_metrics.box.map,
        'test_precision': test_metrics.box.mp,
        'test_recall': test_metrics.box.mr
    })

    inference_df = inference(model, os.path.join(ANNOTATIONS_DIR, 'test', 'images'), active_class_label=0)

else:
    model = YOLO(MODEL_PATH)
    inference_df = inference(model, args.source, active_class_label=CLASS_LABEL)

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