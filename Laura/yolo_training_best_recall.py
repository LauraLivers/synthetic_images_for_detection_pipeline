import os
import re
import cv2
import pandas as pd
from dotenv import load_dotenv
import wandb
from ultralytics import YOLO
import torch
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='yolov8m-oiv7.pt')
parser.add_argument('--run_name', type=str, required=True, help='e.g. dataset_mobi_augmented_5pct_lr0.00078_mom0.605_wd0.00025_20260601_175434')

args = parser.parse_args()

match = re.search(r'^(.+?)_lr([\d.]+)_mom([\d.]+)_wd([\d.]+)', args.run_name)
if not match:
    raise ValueError(f"Could not parse run name: {args.run_name}")
dataset_name = match.group(1)
lr0 = float(match.group(2))
momentum = float(match.group(3))
weight_decay = float(match.group(4))

MODEL_PATH = args.weights
CLASS_LABEL = 298
DATASET_YAML = str(Path(dataset_name) / 'dataset.yaml')
BATCH_SIZE = 4
EPOCHS = 10
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.5
WANDB_PROJECT = 'ladder-detection-optuna-recall'

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()

def run_name(lr0, momentum, weight_decay):
    return f"{dataset_name}_lr{lr0:.5f}_mom{momentum:.3f}_wd{weight_decay:.5f}"

def inference(model, source_dir, active_class_label, output_dir='bounding_boxes'):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for filename in os.listdir(source_dir):
        if not filename.endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.realpath(os.path.join(source_dir, filename))
        annotation = os.path.join(str(Path(source_dir).parent), 'labels', os.path.splitext(filename)[0] + '.txt')
        true_label = 1 if os.path.exists(annotation) and os.path.getsize(annotation) > 0 else 0

        yolo_result = model.predict(img_path, conf=0.1, iou=0.1, verbose=False)[0]
        ladder_boxes = [box for box in yolo_result.boxes if int(box.cls.item()) == active_class_label]
        max_conf = max([box.conf.item() for box in ladder_boxes]) if ladder_boxes else None

        original_image = cv2.imread(img_path)
        for box in ladder_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = (0, 255, 0) if box.conf.item() >= CONFIDENCE_THRESHOLD else (0, 0, 255)
            cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(original_image, f'ladder {box.conf.item():.4f}', (x1, y1 - 10),
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

def get_ladder_class_id(model_names):
    for k, v in model_names.items():
        if v.lower() == 'ladder':
            return k
    return CLASS_LABEL

def make_callback(run):
    def on_fit_epoch_end(trainer):
        run.log({
            'epoch': trainer.epoch,
            'train_box_loss': trainer.loss_items[0].item(),
            'train_cls_loss': trainer.loss_items[1].item(),
            'train_dfl_loss': trainer.loss_items[2].item(),
            'val_mAP50': trainer.metrics.get('metrics/mAP50(B)', 0),
            'val_mAP50_95': trainer.metrics.get('metrics/mAP50-95(B)', 0),
            'val_precision': trainer.metrics.get('metrics/precision(B)', 0),
            'val_recall': trainer.metrics.get('metrics/recall(B)', 0),
        })
    return on_fit_epoch_end

name = run_name(lr0, momentum, weight_decay)

run = wandb.init(project=WANDB_PROJECT, name=f"final_{name}",
                 config={'lr0': lr0, 'momentum': momentum, 'weight_decay': weight_decay,
                         'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'dataset': dataset_name})

final_model = YOLO(MODEL_PATH)
final_model.add_callback('on_fit_epoch_end', make_callback(run))

train_results = final_model.train(
    data=DATASET_YAML, epochs=EPOCHS, batch=BATCH_SIZE,
    lr0=lr0, momentum=momentum, weight_decay=weight_decay,
    seed=RANDOM_SEED, save=True, project=WANDB_PROJECT, name=f"final_{name}",
    device=device, close_mosaic=0, warmup_epochs=1, freeze=22
)

final_model = YOLO(train_results.save_dir / 'weights/best.pt')
test_metrics = final_model.val(data=DATASET_YAML, split='test')

run = wandb.init(project=WANDB_PROJECT, name=f"final_{name}_eval", resume="allow")
run.log({
    'test_mAP50': test_metrics.box.map50,
    'test_mAP50_95': test_metrics.box.map,
    'test_precision': test_metrics.box.mp,
    'test_recall': test_metrics.box.mr,
})

test_images_dir = Path(DATASET_YAML).parent / 'test' / 'images'
inference_output_dir = Path(DATASET_YAML).parent / 'inference_visualizations'
active_class = get_ladder_class_id(final_model.names)
inference_df = inference(final_model, str(test_images_dir), active_class_label=active_class,
                         output_dir=str(inference_output_dir))

detected = inference_df[inference_df['confidence_ladder'] >= CONFIDENCE_THRESHOLD]
hard_samples = inference_df[inference_df['confidence_ladder'].notna() & (inference_df['confidence_ladder'] < CONFIDENCE_THRESHOLD)]
no_detection = inference_df[inference_df['confidence_ladder'].isna()]

detected.to_csv(inference_output_dir / 'detected.csv', index=False)
hard_samples.to_csv(inference_output_dir / 'hard_samples.csv', index=False)
no_detection.to_csv(inference_output_dir / 'no_detection.csv', index=False)

run.finish()