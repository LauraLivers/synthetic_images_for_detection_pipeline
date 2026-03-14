import torch
import torch.nn as nn
from  torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import cv2

from dotenv import load_dotenv
import os
import wandb

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = 'robo_images'
LABELS = 'resnet50_labels.csv' # change if file changes
MODEL_SAVE_PATH = 'best_model.pth'
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
# tune
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 10
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.5

load_dotenv()
wandb.init(project='ladder-detection-resnet', config={
    'batch_size' : BATCH_SIZE,
    'learning_rate' :  LEARNING_RATE,
    'epochs' : EPOCHS,
    'random_seed' : RANDOM_SEED,
    'train_size' : TRAIN_SIZE,
    'val_size' : VAL_SIZE,
    'test_size' : TEST_SIZE
})


class LadderDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        folder = 'ladder' if self.df.iloc[idx]['ladder'] == 1 else 'no_ladder'
        image = Image.open(os.path.join(IMAGE_DIR, folder, self.df.iloc[idx]['image_name']))
        label = self.df.iloc[idx]['ladder']
        if self.transform:
            image = self.transform(image)
        return image, label, self.df.iloc[idx]['image_name']
    
transform = transforms.Compose(
    [transforms.Resize((224,224)), # RestNet specific
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] # ResNet specific
) 
df = pd.read_csv(LABELS)
df = df[df['image_name'].str.endswith(('.jpg', '.jpeg', '.png'))]

train_df, temp_df = train_test_split(df, test_size=(VAL_SIZE + TEST_SIZE), stratify=df['ladder'], random_state=RANDOM_SEED)
val_df, test_df = train_test_split(temp_df, test_size=TEST_SIZE/(VAL_SIZE+TEST_SIZE), stratify=temp_df['ladder'], random_state=RANDOM_SEED)

train_dataset = LadderDataset(train_df.reset_index(drop=True), transform=transform)
val_dataset = LadderDataset(val_df.reset_index(drop=True), transform=transform)
test_dataset = LadderDataset(train_df.reset_index(drop=True), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

class_weights = torch.tensor([1.0, len(train_df[train_df['ladder']==0]) / len(train_df[train_df['ladder']==1])]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        
    return total_loss / len(loader), correct / len(loader.dataset)

def validation_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def inference(model, loader, output_dir='grad_cam'):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    results = []

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    # gradCAM needs gradients, so no torch.no_grad()
    for images, labels, filenames in loader:
        images = images.to(device)
        outputs = torch.softmax(model(images), dim=1) # why softmax?
        for i, (filename, label, confidence )in enumerate(zip(filenames, labels, outputs)):
            confidence_ladder = confidence[1].item()
            predicted_label = confidence.argmax().item()
            correct = label.item() == predicted_label

            
            input_tensor = images[i].unsqueeze(0)
            targets = [ClassifierOutputTarget(1)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # create and save heatmap
            original_image = cv2.imread(os.path.join(IMAGE_DIR, 'ladder' if label.item() == 1 else 'no_ladder', filename))
            original_image = cv2.resize(original_image, (224,224))
            original_image = original_image / 255.0
            visualization = show_cam_on_image(original_image.astype(np.float32), grayscale_cam[0], use_rgb=True)
            cv2.imwrite(os.path.join(output_dir, f'{filename}_grad_cam.png'), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            results.append({
                'filename' : filename,
                'true_label' : label.item(),
                'confidence_ladder' : confidence[1].item(),
                'confidence_no_ladder' : confidence[0].item(),
                'predicted_label' : confidence.argmax().item(),
                'correct' : label.item() == confidence.argmax().item(),
                'gradcam_path' : os.path.join(output_dir, f'{filename}_gradcam.png')
            })
    return pd.DataFrame(results)

best_val_loss = float('inf') # initialization

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validation_epoch(model, val_loader, criterion)
    scheduler.step()
    print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    wandb.log({
        'epoch' : epoch + 1,
        'train_loss' : train_loss,
        'train_acc' :  train_acc,
        'val_loss' : val_loss,
        'val_acc' : val_acc
    })
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

model.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_loss, test_acc = validation_epoch(model, test_loader, criterion)
wandb.log({
    'test_loss' : test_loss,
    'test_acc' : test_acc
    })
print(f'Test loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')
inference_df = inference(model, train_loader)
hard_samples = inference_df[
    (inference_df['true_label'] == 1) &
    (inference_df['confidence_ladder'] < CONFIDENCE_THRESHOLD)
]

hard_samples.to_csv('hard_samples.csv', index=False)
wandb.log({'num_hard_samples' : len(hard_samples)})
wandb.finish()


