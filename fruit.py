import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tqdm import tqdm

DATASET_DIR = "Fruit/data"
SOURCE_DIRS = [os.path.join(DATASET_DIR, "train"), os.path.join(DATASET_DIR, "test")]
BATCH_SIZE = 32
NUM_WORKERS = 4
IMAGE_SIZE = 224
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 15
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 1e-5
RANDOM_SEED = 42
MODEL_PATH = "best_efficientnet_fruit.pth"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))

weights = EfficientNet_B0_Weights.DEFAULT
mean = weights.transforms().mean
std = weights.transforms().std

train_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=15), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02), transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
val_test_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

class FruitDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.class_to_idx = {"ripe": 0, "rotten": 1}
        self.classes = ["ripe", "rotten"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filepath, label = self.samples[index]
        image = Image.open(filepath).convert("RGB")
        if self.transform is not None: image = self.transform(image)
        return image, label

def scan_dataset():
    samples = []
    folders = []

    for source_dir in SOURCE_DIRS:
        if not os.path.isdir(source_dir): raise FileNotFoundError(f"Dataset directory not found: {source_dir}")
        for folder_name in os.listdir(source_dir):
            folder_path = os.path.join(source_dir, folder_name)
            if os.path.isdir(folder_path) and (folder_name.lower().startswith("fresh") or folder_name.lower().startswith("rotten")): folders.append(folder_path)

    print(f"\nFound {len(folders)} fresh/rotten folders.")

    for folder_path in tqdm(folders, desc="Reading dataset folders", unit="folder"):
        folder_name = os.path.basename(folder_path).lower()
        if folder_name.startswith("fresh"): label = 0
        elif folder_name.startswith("rotten"): label = 1
        else: continue

        png_files = [filename for filename in os.listdir(folder_path) if filename.lower().endswith(".png")]

        for filename in png_files:
            filepath = os.path.join(folder_path, filename)
            try:
                with Image.open(filepath) as img: img.verify()
                samples.append((filepath, label))
            except Exception:
                print(f"\nWarning: Skipping invalid image: {filepath}")

    return samples

print("\nScanning dataset...")
samples = scan_dataset()

if len(samples) == 0: raise RuntimeError(f"No PNG images were found inside the fresh/rotten folders under {SOURCE_DIRS}.")

full_dataset = FruitDataset(samples)

class_names = full_dataset.classes
num_classes = len(class_names)

print("\nClasses:", full_dataset.class_to_idx)
print("Number of images:", len(full_dataset))

if num_classes != 2: raise ValueError(f"Expected 2 classes, but found {num_classes}: {class_names}")

all_labels = np.array([label for _, label in samples])

print("\nComplete dataset distribution:")
for class_id, class_name in enumerate(class_names): print(f"{class_name}: {np.sum(all_labels == class_id)}")

indices = np.arange(len(full_dataset))

train_indices, temp_indices = train_test_split(indices, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)

val_indices, test_indices = train_test_split(temp_indices, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_indices])

print("\n" + "=" * 60)
print("DATASET SPLIT")
print("=" * 60)
print(f"Training:   {len(train_indices)} ({len(train_indices) / len(full_dataset) * 100:.1f}%)")
print(f"Validation: {len(val_indices)} ({len(val_indices) / len(full_dataset) * 100:.1f}%)")
print(f"Test:       {len(test_indices)} ({len(test_indices) / len(full_dataset) * 100:.1f}%)")

train_dataset_full = FruitDataset(samples, transform=train_transform)
val_dataset_full = FruitDataset(samples, transform=val_test_transform)
test_dataset_full = FruitDataset(samples, transform=val_test_transform)

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(val_dataset_full, val_indices)
test_dataset = Subset(test_dataset_full, test_indices)

train_labels = all_labels[train_indices]

print("\nTraining class distribution:")
for class_id, class_name in enumerate(class_names): print(f"{class_name}: {np.sum(train_labels == class_id)}")

class_counts = np.bincount(train_labels, minlength=num_classes)
class_weights = torch.tensor(len(train_labels) / (num_classes * class_counts), dtype=torch.float32).to(device)

print("\nClass weights:", class_weights)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

model = efficientnet_b0(weights=weights)

for param in model.features.parameters(): param.requires_grad = False

num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_features, num_classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs, stage):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc=f"{stage} Epoch {epoch}/{total_epochs}", unit="batch")
    for images, labels_batch in progress:
        images = images.to(device, non_blocking=True)
        labels_batch = labels_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()
        progress.set_postfix(loss=f"{running_loss / total:.4f}", acc=f"{correct / total:.4f}")
    return running_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels_batch in loader:
            images = images.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
    return running_loss / total, correct / total

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_accuracy = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE_HEAD, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

print("\n" + "=" * 60)
print("STAGE 1: TRAINING CLASSIFIER")
print("=" * 60)

for epoch in range(WARMUP_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch + 1, WARMUP_EPOCHS, "Stage 1")
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    scheduler.step(val_acc)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch + 1}/{WARMUP_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), MODEL_PATH)

for param in model.features.parameters(): param.requires_grad = False
for param in model.features[-2:].parameters(): param.requires_grad = True
for param in model.classifier.parameters(): param.requires_grad = True

optimizer = optim.AdamW([{"params": model.features[-2:].parameters(), "lr": LEARNING_RATE_BACKBONE}, {"params": model.classifier.parameters(), "lr": LEARNING_RATE_HEAD * 0.1}], weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

print("\n" + "=" * 60)
print("STAGE 2: FINE-TUNING")
print("=" * 60)

for epoch in range(FINETUNE_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch + 1, FINETUNE_EPOCHS, "Stage 2")
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    scheduler.step(val_acc)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    print(f"Fine-tune Epoch {epoch + 1}/{FINETUNE_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), MODEL_PATH)
        print("Best model saved!")

model.load_state_dict(best_model_weights)
model.eval()

all_predictions = []
all_labels = []

print("\nEvaluating test set...")

with torch.no_grad():
    for images, labels_batch in tqdm(test_loader, desc="Testing", unit="batch"):
        images = images.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels_batch.numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

test_accuracy = accuracy_score(all_labels, all_predictions)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(all_labels, all_predictions, target_names=class_names, digits=4))

cm = confusion_matrix(all_labels, all_predictions)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.title("EfficientNet-B0 Fruit Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history["train_acc"], label="Training Accuracy")
plt.plot(history["val_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history["train_loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.show()

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Model saved to: {MODEL_PATH}")
print("\nClasses:")
for index, name in enumerate(class_names): print(f"{index}: {name}")