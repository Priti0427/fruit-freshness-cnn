import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

DATASET_DIR = "dataset"
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

full_dataset = datasets.ImageFolder(root=DATASET_DIR)

print("\nClasses:", full_dataset.class_to_idx)
print("Number of images:", len(full_dataset))

class_names = full_dataset.classes
num_classes = len(class_names)

if num_classes != 3: raise ValueError(f"Expected 3 classes, but found {num_classes}: {class_names}")

labels = np.array([label for _, label in full_dataset.samples])
indices = np.arange(len(full_dataset))

train_indices, temp_indices = train_test_split(indices, test_size=0.30, random_state=RANDOM_SEED, stratify=labels)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.50, random_state=RANDOM_SEED, stratify=labels[temp_indices])

print("\nDataset split:")
print("Training:", len(train_indices))
print("Validation:", len(val_indices))
print("Test:", len(test_indices))

train_dataset_full = datasets.ImageFolder(root=DATASET_DIR, transform=train_transform)
val_dataset_full = datasets.ImageFolder(root=DATASET_DIR, transform=val_test_transform)
test_dataset_full = datasets.ImageFolder(root=DATASET_DIR, transform=val_test_transform)

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(val_dataset_full, val_indices)
test_dataset = Subset(test_dataset_full, test_indices)

train_labels = labels[train_indices]

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

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels_batch in loader:
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
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
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
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
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

with torch.no_grad():
    for images, labels_batch in test_loader:
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