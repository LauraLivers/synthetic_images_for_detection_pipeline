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

parser = argparse.ArgumentParser()
parser.add_argument('--inference_only', action='store_true')
parser.add_argument('--source', type=str, default=None)
parser.add_argument('--dataset_yaml', type=str, default='dataset.yaml')
parser.add_argument('--weights', type=str, default='yolov8m-oiv7.pt')
args = parser.parse_args()

if args.inference_only and args.source is None:
    parser.error('--source is required with --inference_only')

if args.inference_only and args.dataset_yaml == 'dataset.yaml' and args.source:
    inferred_yaml = Path(args.source).parent.parent / 'dataset.yaml'
    if inferred_yaml.exists():
        args.dataset_yaml = str(inferred_yaml)

MODEL_PATH = args.weights
CLASS_LABEL = 298
DATASET_YAML = args.dataset_yaml
BATCH_SIZE = 4
EPOCHS = 10
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.5
WANDB_PROJECT = 'ladder-detection-optuna'

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()
dataset_name = Path(DATASET_YAML).parent.name

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

def train_model(model_path, name, lr0, momentum, weight_decay, job_type, tags):
    os.environ['WANDB_TAGS'] = ','.join(tags)
    os.environ['WANDB_JOB_TYPE'] = job_type
    model = YOLO(model_path)
    train_results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        lr0=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        seed=RANDOM_SEED,
        save=True,
        project=WANDB_PROJECT,
        name=name,
        device=device,
        close_mosaic=0,
        warmup_epochs=1,
        freeze=22,
    )
    os.environ.pop('WANDB_TAGS', None)
    os.environ.pop('WANDB_JOB_TYPE', None)
    return model, train_results

def append_metrics_to_yolo_run(run_id, metrics):
    run = wandb.init(project=WANDB_PROJECT, id=run_id, resume='must')
    wandb.log(metrics)
    wandb.finish()

if not args.inference_only:

    def objective(trial):
        lr0 = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
        momentum = trial.suggest_float("momentum", 0.6, 0.98)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        name = run_name(lr0, momentum, weight_decay)

        model, _ = train_model(MODEL_PATH, name, lr0, momentum, weight_decay,
                               job_type='optuna-trial', tags=['optuna', dataset_name])
        yolo_run_id = wandb.run.id if wandb.run else None

        val_metrics = model.val(data=DATASET_YAML, split='val')

        if yolo_run_id:
            append_metrics_to_yolo_run(yolo_run_id, {
                'val/mAP50':      val_metrics.box.map50,
                'val/mAP50_95':   val_metrics.box.map,
                'val/precision':  val_metrics.box.mp,
                'val/recall':     val_metrics.box.mr,
            })

        return val_metrics.box.map

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    best_params = study.best_trial.params
    best_name = run_name(best_params['lr0'], best_params['momentum'], best_params['weight_decay'])

    print(f"Best mAP50-95: {study.best_trial.value:.4f}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    final_model, train_results = train_model(MODEL_PATH, f"final_{best_name}",
                                             best_params['lr0'], best_params['momentum'], best_params['weight_decay'],
                                             job_type='train', tags=['final', dataset_name])
    yolo_run_id = wandb.run.id if wandb.run else None

    final_model = YOLO(train_results.save_dir / 'weights/best.pt')
    final_model.save(f'best_model_{best_name}.pt')

    test_metrics = final_model.val(data=DATASET_YAML, split='test')

    print(f"mAP50: {test_metrics.box.map50:.4f}  mAP50-95: {test_metrics.box.map:.4f}  P: {test_metrics.box.mp:.4f}  R: {test_metrics.box.mr:.4f}")

    test_images_dir = Path(DATASET_YAML).parent / 'test' / 'images'
    inference_output_dir = Path(DATASET_YAML).parent / 'inference_visualizations'
    active_class = get_ladder_class_id(final_model.names)
    inference_df = inference(final_model, str(test_images_dir), active_class_label=active_class,
                             output_dir=str(inference_output_dir))

    detected     = inference_df[inference_df['confidence_ladder'] >= CONFIDENCE_THRESHOLD]
    hard_samples = inference_df[inference_df['confidence_ladder'].notna() & (inference_df['confidence_ladder'] < CONFIDENCE_THRESHOLD)]
    no_detection = inference_df[inference_df['confidence_ladder'].isna()]

    detected.to_csv(inference_output_dir / 'detected.csv', index=False)
    hard_samples.to_csv(inference_output_dir / 'hard_samples.csv', index=False)
    no_detection.to_csv(inference_output_dir / 'no_detection.csv', index=False)

    if yolo_run_id:
        append_metrics_to_yolo_run(yolo_run_id, {
            'test/mAP50':                 test_metrics.box.map50,
            'test/mAP50_95':              test_metrics.box.map,
            'test/precision':             test_metrics.box.mp,
            'test/recall':                test_metrics.box.mr,
            'inference/num_detected':     len(detected),
            'inference/num_hard_samples': len(hard_samples),
            'inference/num_no_detection': len(no_detection),
        })

else:
    model = YOLO(MODEL_PATH)
    active_class = get_ladder_class_id(model.names)

    val_metrics = model.val(data=DATASET_YAML, split='test')

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"inference_{Path(MODEL_PATH).stem}_{dataset_name}",
        job_type='inference',
        tags=['inference', dataset_name],
        config={'model_path': MODEL_PATH, 'source': args.source, 'dataset': dataset_name}
    )

    inference_output_dir = Path(args.source).parent / 'inference_visualizations'
    inference_df = inference(model, args.source, active_class_label=active_class,
                             output_dir=str(inference_output_dir))

    true_positives  = inference_df[(inference_df['true_label'] == 1) & (inference_df['predicted_label'].isin(['1', '1_x']))]
    false_positives = inference_df[(inference_df['true_label'] == 0) & (inference_df['predicted_label'].isin(['1', '1_x']))]
    false_negatives = inference_df[(inference_df['true_label'] == 1) & (inference_df['predicted_label'].isin(['0', 'x']))]
    true_negatives  = inference_df[(inference_df['true_label'] == 0) & (inference_df['predicted_label'].isin(['0', 'x']))]
    accuracy = inference_df['correct'].mean() * 100 if len(inference_df) > 0 else 0

    inference_df.to_csv(inference_output_dir / 'inference_results.csv', index=False)

    print(f"mAP50: {val_metrics.box.map50:.4f}  mAP50-95: {val_metrics.box.map:.4f}  P: {val_metrics.box.mp:.4f}  R: {val_metrics.box.mr:.4f}")
    print(f"TP: {len(true_positives)}  FP: {len(false_positives)}  FN: {len(false_negatives)}  TN: {len(true_negatives)}  Acc: {accuracy:.2f}%")

    wandb.log({
        'test/mAP50':                val_metrics.box.map50,
        'test/mAP50_95':             val_metrics.box.map,
        'test/precision':            val_metrics.box.mp,
        'test/recall':               val_metrics.box.mr,
        'inference/true_positives':  len(true_positives),
        'inference/false_positives': len(false_positives),
        'inference/false_negatives': len(false_negatives),
        'inference/true_negatives':  len(true_negatives),
        'inference/accuracy_pct':    accuracy,
    })

    wandb.finish()