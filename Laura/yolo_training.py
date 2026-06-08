import os
import cv2
import pandas as pd
from dotenv import load_dotenv
import wandb
from ultralytics import YOLO
import torch
import argparse
import optuna
from pathlib import Path

### Parser
parser = argparse.ArgumentParser()
parser.add_argument('--inference_only', action='store_true')
parser.add_argument('--source', type=str, default=None)
parser.add_argument('--datasets_dir', type=str, default=None)
parser.add_argument('--dataset_yaml', type=str, default='dataset_real_baseline/dataset.yaml')
parser.add_argument('--weights', type=str, default='yolov8m-oiv7.pt', help='Model weights to use')
args = parser.parse_args()

if args.inference_only and args.source is None:
    parser.error('--source is required with --inference_only')

# Auto-detect dataset_yaml from source if it wasn't explicitly changed from default
if args.inference_only and args.dataset_yaml == 'dataset_real_baseline/dataset.yaml' and args.source:
    inferred_yaml = Path(args.source).parent.parent / 'dataset.yaml'
    if inferred_yaml.exists():
        args.dataset_yaml = str(inferred_yaml)

### Config Variables
MODEL_PATH = args.weights
CLASS_LABEL = 298 # Fallback hardcoded for open-images ladder
DATASET_YAML = str(Path(args.dataset_yaml).resolve())
BATCH_SIZE = 4
EPOCHS = 10
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.5
WANDB_OPTUNA_PROJECT = 'ladder-detection-optuna'

### Acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()

def run_name(dataset_name, lr0, momentum, weight_decay):
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

        # increase iou threshold to aggressively suppress overlapping boxes natively via YOLO logic
        yolo_result = model.predict(img_path, conf=0.1, iou=0.1, verbose=False)[0]
        ladder_boxes = [box for box in yolo_result.boxes if int(box.cls.item()) == active_class_label]
        max_conf = max([box.conf.item() for box in ladder_boxes]) if ladder_boxes else None

        original_image = cv2.imread(img_path)
        for box in ladder_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = (0, 255, 0) if box.conf.item() >= CONFIDENCE_THRESHOLD else (0, 0, 255)
            cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(original_image, f'ladder {box.conf.item():.4f}', (x1, y1-10),
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
            'correct': true_label == (1 if predicted_label in ['1', '1_x', 'x'] else 0),
            'detection_path': os.path.join(output_dir, f'{predicted_label}_{filename}_bb.png')
        })
    return pd.DataFrame(results)

def get_ladder_class_id(model_names):
    for k, v in model_names.items():
        if v.lower() == 'ladder':
            return k
    return CLASS_LABEL

def objective(trial, dataset_yaml, dataset_name):
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.6, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    model = YOLO(MODEL_PATH)
    name = run_name(dataset_name, lr0, momentum, weight_decay)

    wandb.init(project=WANDB_OPTUNA_PROJECT, name=name, reinit=True,
               config={'lr0': lr0, 'momentum': momentum, 'weight_decay': weight_decay})

    model.train(
        data=dataset_yaml,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        lr0=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        seed=RANDOM_SEED,
        project=WANDB_OPTUNA_PROJECT,
        name=name,
        device=device,
        close_mosaic=0,
        warmup_epochs=1,
        freeze=22
    )

    # on_train_end closed the run; reopen just to log the optuna target metric
    val_metrics = model.val(data=dataset_yaml, split='val')
    map50 = val_metrics.box.map50
    wandb.finish()

    return map50

def train_on_dataset(dataset_yaml):
    dataset_name = Path(dataset_yaml).parent.name

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, dataset_yaml, dataset_name), n_trials=15)

    print("\nOptuna Tuning Complete")
    print(f"Best trial value (mAP50): {study.best_trial.value}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

if not args.inference_only:
    if args.datasets_dir:
        datasets_root = Path(args.datasets_dir)
        dataset_yamls = sorted(datasets_root.glob("*/dataset.yaml"))
        if not dataset_yamls:
            raise ValueError(f"No dataset.yaml files found under {datasets_root}")
        for dataset_yaml in dataset_yamls:
            train_on_dataset(str(dataset_yaml))
    else:
        train_on_dataset(DATASET_YAML)

else:
    model = YOLO(MODEL_PATH)
    active_class = get_ladder_class_id(model.names)

    # Actually run YOLO val to get mAP and realistic metrics
    val_metrics = model.val(data=DATASET_YAML, split='test')

    # Save the inference run inside the main training project so all reports are unified
    wandb.init(project=WANDB_OPTUNA_PROJECT, name=f"inference_{Path(MODEL_PATH).stem}", config={'model_path': MODEL_PATH, 'source': args.source})

    inference_output_dir = Path(args.source).parent / 'inference_visualizations'
    inference_df = inference(model, args.source, active_class_label=active_class, output_dir=str(inference_output_dir))

    true_positives = inference_df[(inference_df['true_label'] == 1) & (inference_df['predicted_label'].isin(['1', '1_x', 'x']))]
    false_positives = inference_df[(inference_df['true_label'] == 0) & (inference_df['predicted_label'].isin(['1', '1_x', 'x']))]
    false_negatives = inference_df[(inference_df['true_label'] == 1) & (inference_df['predicted_label'] == '0')]
    true_negatives = inference_df[(inference_df['true_label'] == 0) & (inference_df['predicted_label'] == '0')]

    detected = inference_df[inference_df['predicted_label'].isin(['1', '1_x'])]
    hard_samples = inference_df[inference_df['predicted_label'] == 'x']
    no_detection = inference_df[inference_df['predicted_label'] == '0']

    # Overall correctness percentage
    accuracy = inference_df['correct'].mean() * 100 if len(inference_df) > 0 else 0

    inference_df.to_csv(inference_output_dir / 'inference_results.csv', index=False)
    detected.to_csv(inference_output_dir / 'detected.csv', index=False)
    hard_samples.to_csv(inference_output_dir / 'hard_samples.csv', index=False)
    no_detection.to_csv(inference_output_dir / 'no_detection.csv', index=False)

    print(f"\nModel Performance Metrics:")
    print(f"mAP50: {val_metrics.box.map50:.4f}")
    print(f"Precision: {val_metrics.box.mp:.4f}")
    print(f"Recall: {val_metrics.box.mr:.4f}\n")

    print(f"Total images processed: {len(inference_df)}")
    print(f"True Positives (Found Ladders properly): {len(true_positives)}")
    print(f"False Positives (Hallucinated Ladders): {len(false_positives)}")
    print(f"False Negatives (Missed Ladders): {len(false_negatives)}")
    print(f"True Negatives (Correctly ignored empty images): {len(true_negatives)}")
    print(f"Overall Accuracy: {accuracy:.2f}%")

    wandb.log({
        'test_mAP50': val_metrics.box.map50,
        'test_mAP50_95': val_metrics.box.map,
        'test_precision': val_metrics.box.mp,
        'test_recall': val_metrics.box.mr,
        'num_detected': len(detected),
        'num_hard_samples': len(hard_samples),
        'num_no_detection': len(no_detection),
        'true_positives': len(true_positives),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'true_negatives': len(true_negatives),
        'accuracy_pct': accuracy
    })

    wandb.finish()