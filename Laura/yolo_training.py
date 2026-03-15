import os
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import wandb
import shutil
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.model_selection import train_test_split
import torch


IMAGE_DIR = 'robo_images'
LABELS = 'robo_images/yolo_annotations' # change if file changes
MODEL_SAVE_PATH = 'best_model.pt'
DATASET_YAML = 'dataset.yaml'
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
# tune
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 20
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.5

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
load_dotenv()
wandb.init(project='ladder-detection-yolo', config={
    'batch_size' : BATCH_SIZE,
    'learning_rate' :  LEARNING_RATE,
    'epochs' : EPOCHS,
    'random_seed' : RANDOM_SEED,
    'train_size' : TRAIN_SIZE,
    'val_size' : VAL_SIZE,
    'test_size' : TEST_SIZE
})

# dataset split
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


model = YOLO('yolov8n.pt')
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
    device='mps' if torch.backends.mps.is_available() else '0' if torch.cuda.is_available() else 'cpu',
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

class YOLOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        outputs = self.model(x)
        while isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs
    


def inference(model, output_dir='grad_cam'):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    wrapped_model = YOLOWrapper(model.model)
    target_layer = model.model.model[-2]
    cam = EigenCAM(wrapped_model, target_layers=[target_layer])

    for folder in ['ladder', 'no_ladder']:
        true_label = 1 if folder == 'ladder' else 0
        for filename in os.listdir(os.path.join(IMAGE_DIR, folder)):
            if not filename.endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(IMAGE_DIR, folder, filename)
            yolo_result = model.predict(img_path, conf=CONFIDENCE_THRESHOLD)[0]
            max_conf = yolo_result.boxes.conf.max().item() if len(yolo_result.boxes) > 0 else 0.0
            predicted_label = 1 if max_conf >= CONFIDENCE_THRESHOLD else 0

            original_image = cv2.imread(img_path)
            original_image = cv2.resize(original_image, (640,640)) # fixed size for YOLO
            original_image = original_image / 255.0

            input_tensor = torch.from_numpy(original_image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=input_tensor)[0]

            visualization = show_cam_on_image(original_image.astype(np.float32), grayscale_cam, use_rgb=True)
            cv2.imwrite(os.path.join(output_dir, f'{filename}_grad_cam.png'), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            results.append({
                'filename' : filename,
                'true_label' : true_label,
                'confidence_ladder' : max_conf,
                'predicted_label' : predicted_label,
                'correct' : true_label == predicted_label,
                'gradcam_path' : os.path.join(output_dir, f'{filename}_grad_cam.png')
            })
    return pd.DataFrame(results)
    
inference_df = inference(model)
hard_samples = inference_df[
    (inference_df['true_label'] == 1) &
    (inference_df['confidence_ladder'] < CONFIDENCE_THRESHOLD)
]
hard_samples.to_csv('hard_samples.csv', index=False)
wandb.log({'num_hard_samples' : len(hard_samples)})
wandb.finish()