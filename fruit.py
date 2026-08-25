import os
import re
import copy
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, f1_score, roc_auc_score)
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "data")
SOURCE_DIRS = [os.path.join(DATASET_DIR, "train"), os.path.join(DATASET_DIR, "test")]
BATCH_SIZE = 32
NUM_WORKERS = 0
IMAGE_SIZE = 224
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 25
EARLY_STOP_PATIENCE = 6
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 1e-5
LABEL_SMOOTHING = 0.05
RANDOM_SEED = 42
MODEL_DIR = os.path.join(PROJECT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

CLASS_NAMES = ["fresh", "rotten"]
NUM_CLASSES = 2

N_FOLDS = 5

# The 6 fresh/rotten folders ship each source photograph 9 times: the unprefixed
# original plus these 8 fixed variants. Splitting at file level therefore leaks
# every test photo into training, so we group by source photograph instead.
AUG_PREFIXES = ("rotated_by_15_", "rotated_by_30_", "rotated_by_45_", "rotated_by_60_",
                "rotated_by_75_", "vertical_flip_", "translation_", "saltandpepper_")

# 6 of the 9 variants carry a black corner fill that never occurs at evaluation
# time, so training on them is a distribution mismatch. "originals" is the
# honest default; "all" exists so the ablation is a one-line change.
TRAIN_VARIANTS = "originals"

RUN_PERMUTATION_CONTROL = True
RUN_COLOR_BASELINE = True

device = torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SaltPepper:
    """Replaces the offline `saltandpepper_` variant. Operates on a CHW float tensor."""

    def __init__(self, amount=0.02):
        self.amount = amount

    def __call__(self, tensor):
        mask = torch.rand(1, tensor.shape[1], tensor.shape[2])
        out = tensor.clone()
        out[:, (mask[0] < self.amount / 2)] = 0.0
        out[:, (mask[0] > 1 - self.amount / 2)] = 1.0
        return out


class FruitDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        self.classes = list(CLASS_NAMES)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filepath, label = self.samples[index]
        image = Image.open(filepath).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def strip_aug(filename):
    """Map any of the 9 stored variants back to its source photograph's filename."""
    for prefix in AUG_PREFIXES:
        if filename.startswith(prefix):
            return filename[len(prefix):], False
    return filename, True


def fruit_of(folder_name):
    for fruit_name in ("apples", "banana", "oranges"):
        if folder_name.endswith(fruit_name):
            return fruit_name
    return "unknown"


def scan_dataset():
    """Index the 6 fresh/rotten folders across both source dirs.

    Returns one record per file, carrying the source-photograph group key that
    the split is built on. The `.png` filter also excludes the `unripe *`
    folders, whose images are 100% JPEG thumbnails from a different source.
    """
    records = []
    folders = []

    for source_dir in SOURCE_DIRS:
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Dataset directory not found: {source_dir}")

        for folder_name in sorted(os.listdir(source_dir)):
            folder_path = os.path.join(source_dir, folder_name)

            if not os.path.isdir(folder_path):
                continue

            folder_name_lower = folder_name.lower()

            if folder_name_lower.startswith("fresh") or folder_name_lower.startswith("rotten"):
                folders.append(folder_path)

    print(f"\nFound {len(folders)} fresh/rotten folders.")

    if len(folders) == 0:
        raise RuntimeError(f"No folders beginning with 'fresh' or 'rotten' were found in {SOURCE_DIRS}.")

    for folder_path in tqdm(folders, desc="Reading dataset folders", unit="folder"):
        folder_name = os.path.basename(folder_path).lower()

        if folder_name.startswith("fresh"):
            label = 0
        elif folder_name.startswith("rotten"):
            label = 1
        else:
            continue

        try:
            filenames = os.listdir(folder_path)
        except Exception as error:
            print(f"\nWarning: Could not read folder {folder_path}: {error}")
            continue

        png_files = [filename for filename in filenames if filename.lower().endswith(".png")]

        for filename in sorted(png_files):
            filepath = os.path.join(folder_path, filename)
            base_name, is_original = strip_aug(filename)

            records.append({
                "filepath": filepath,
                "label": label,
                "folder": folder_name,
                "fruit": fruit_of(folder_name),
                "group": (folder_name, base_name),
                "is_original": is_original,
            })

    return records


def audit_index(records):
    """Print the group structure and fail loudly if it is not what we expect."""
    groups = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)

    sizes = {}
    for group_records in groups.values():
        sizes[len(group_records)] = sizes.get(len(group_records), 0) + 1

    group_labels = [group_records[0]["label"] for group_records in groups.values()]
    n_fresh = sum(1 for label in group_labels if label == 0)
    n_rotten = sum(1 for label in group_labels if label == 1)

    print("\n" + "=" * 60)
    print("INDEX AUDIT")
    print("=" * 60)
    print(f"Files indexed:            {len(records)}")
    print(f"Distinct source photos:   {len(groups)}")
    print(f"Files per photo:          {sizes}")
    print(f"Photos per class:         fresh {n_fresh}  |  rotten {n_rotten}")
    print(f"Majority-class baseline:  {max(n_fresh, n_rotten) / len(groups):.4f}")

    for group_records in groups.values():
        labels = {record["label"] for record in group_records}
        if len(labels) != 1:
            raise RuntimeError(f"Group spans multiple classes: {group_records[0]['group']}")

    originals = sum(1 for record in records if record["is_original"])
    if originals != len(groups):
        raise RuntimeError(f"Expected exactly one original per group: {originals} originals vs {len(groups)} groups")

    return groups


def build_transforms(mean, std):
    """Train-time augmentation folds in what the offline variants provided.

    Fills are white, not torchvision's default black: these are white-background
    product photos, and a black wedge is an artifact the eval set never contains.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(degrees=75, fill=(255, 255, 255))], p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.90, 1.10), fill=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.RandomApply([SaltPepper(amount=0.02)], p=0.25),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_test_transform


def set_bn_eval(module):
    """`requires_grad = False` does not stop BatchNorm running stats updating."""
    for layer in module.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


def wilson_ci(correct, total, z=1.96):
    if total == 0:
        return (0.0, 0.0)
    phat = correct / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * np.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (float(center - margin), float(center + margin))


def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs, stage, frozen_module=None):
    model.train()

    if frozen_module is not None:
        set_bn_eval(frozen_module)

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"{stage} Epoch {epoch}/{total_epochs}", unit="batch", leave=False)

    for images, labels_batch in progress:
        images = images.to(device)
        labels_batch = labels_batch.to(device)

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
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels_batch in loader:
            images = images.to(device)
            labels_device = labels_batch.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels_device)
            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels_batch.numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    return running_loss / len(y_true), accuracy, macro_f1, y_true, y_pred, y_prob


def make_loaders(train_records, val_records, test_records, train_transform, val_test_transform):
    def to_samples(records):
        return [(record["filepath"], record["label"]) for record in records]

    use_pin = device.type == "cuda"

    train_loader = DataLoader(FruitDataset(to_samples(train_records), train_transform),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=use_pin)
    val_loader = DataLoader(FruitDataset(to_samples(val_records), val_test_transform),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=use_pin)
    test_loader = DataLoader(FruitDataset(to_samples(test_records), val_test_transform),
                             batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=use_pin)

    return train_loader, val_loader, test_loader


def run_fold(fold, train_records, val_records, test_records, weights, mean, std, stage2=True, tag=""):
    train_transform, val_test_transform = build_transforms(mean, std)
    train_loader, val_loader, test_loader = make_loaders(
        train_records, val_records, test_records, train_transform, val_test_transform)

    train_labels = np.array([record["label"] for record in train_records])
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = torch.tensor(len(train_labels) / (NUM_CLASSES * class_counts),
                                 dtype=torch.float32).to(device)

    model = efficientnet_b0(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_features, NUM_CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = -1.0
    best_model_weights = copy.deepcopy(model.state_dict())

    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE_HEAD, weight_decay=1e-4)

    print(f"\n--- fold {fold}{tag} | stage 1: head only ---")

    for epoch in range(WARMUP_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                epoch + 1, WARMUP_EPOCHS, "Stage 1",
                                                frozen_module=model.features)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"  ep {epoch + 1}/{WARMUP_EPOCHS} | train {train_loss:.4f}/{train_acc:.4f} "
              f"| val {val_loss:.4f}/{val_acc:.4f} | f1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_weights = copy.deepcopy(model.state_dict())

    if stage2:
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.features[-2:].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(
            [{"params": model.features[-2:].parameters(), "lr": LEARNING_RATE_BACKBONE},
             {"params": model.classifier.parameters(), "lr": LEARNING_RATE_HEAD * 0.1}],
            weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-7)

        print(f"--- fold {fold}{tag} | stage 2: fine-tune features[-2:] ---")

        epochs_without_improvement = 0

        for epoch in range(FINETUNE_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                    epoch + 1, FINETUNE_EPOCHS, "Stage 2",
                                                    frozen_module=model.features[:-2])
            val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            print(f"  ep {epoch + 1}/{FINETUNE_EPOCHS} | train {train_loss:.4f}/{train_acc:.4f} "
                  f"| val {val_loss:.4f}/{val_acc:.4f} | f1 {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    print(f"  early stop at epoch {epoch + 1} (no val macro-F1 gain in {EARLY_STOP_PATIENCE})")
                    break

    model.load_state_dict(best_model_weights)

    test_loss, test_acc, test_f1, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion)

    print(f"  fold {fold}{tag} test: acc {test_acc:.4f} | macro-F1 {test_f1:.4f}")

    if not tag:
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"best_fold{fold}.pth"))

    return {
        "history": history,
        "best_val_f1": float(best_val_f1),
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "test_records": test_records,
    }


def color_histogram_baseline(records, group_ids, group_labels, folds):
    """Background-masked HSV histogram + logistic regression, scored out-of-fold.

    If EfficientNet cannot clearly beat this, that is the finding.
    """
    originals = [record for record in records if record["is_original"]]
    index_of_group = {record["group"]: i for i, record in enumerate(originals)}

    features = np.zeros((len(originals), 96), dtype=np.float32)

    for i, record in enumerate(tqdm(originals, desc="Color baseline features", unit="img", leave=False)):
        image = Image.open(record["filepath"]).convert("RGB").resize((64, 64))
        rgb = np.asarray(image)
        mask = ~np.all(rgb > 240, axis=2)
        if mask.sum() < 16:
            mask = np.ones(rgb.shape[:2], dtype=bool)
        hsv = np.asarray(image.convert("HSV"))[mask]
        vector = np.concatenate([np.histogram(hsv[:, c], bins=32, range=(0, 255))[0] for c in range(3)])
        norm = np.linalg.norm(vector)
        features[i] = vector / norm if norm > 0 else vector

    labels = np.array([originals[i]["label"] for i in range(len(originals))])
    oof_pred = np.zeros(len(originals), dtype=int)

    for train_group_idx, test_group_idx in folds:
        train_rows = [index_of_group[group_ids[g]] for g in train_group_idx if group_ids[g] in index_of_group]
        test_rows = [index_of_group[group_ids[g]] for g in test_group_idx if group_ids[g] in index_of_group]

        clf = LogisticRegression(max_iter=2000)
        clf.fit(features[train_rows], labels[train_rows])
        oof_pred[test_rows] = clf.predict(features[test_rows])

    return float(accuracy_score(labels, oof_pred)), float(f1_score(labels, oof_pred, average="macro"))


def main():
    global device

    set_seed(RANDOM_SEED)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    weights = EfficientNet_B0_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    print("\nScanning dataset...")
    records = scan_dataset()

    if len(records) == 0:
        raise RuntimeError(f"No PNG images found inside the fresh/rotten folders under {SOURCE_DIRS}.")

    groups = audit_index(records)

    group_ids = sorted(groups.keys())
    group_label = np.array([groups[g][0]["label"] for g in group_ids])
    group_index = np.arange(len(group_ids))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(skf.split(group_index, group_label))

    records_by_group = {}
    for record in records:
        records_by_group.setdefault(record["group"], []).append(record)

    def select(group_positions, originals_only):
        chosen = []
        for pos in group_positions:
            for record in records_by_group[group_ids[pos]]:
                if originals_only and not record["is_original"]:
                    continue
                chosen.append(record)
        return chosen

    print("\n" + "=" * 60)
    print(f"{N_FOLDS}-FOLD GROUPED CROSS-VALIDATION  (train variants: {TRAIN_VARIANTS})")
    print("=" * 60)

    fold_results = []
    oof_true = np.zeros(len(group_ids), dtype=int)
    oof_pred = np.zeros(len(group_ids), dtype=int)
    oof_prob = np.zeros(len(group_ids), dtype=float)
    oof_fruit = np.empty(len(group_ids), dtype=object)

    for fold, (trainval_pos, test_pos) in enumerate(folds):
        train_pos, val_pos = train_test_split(
            trainval_pos, test_size=0.1875, random_state=RANDOM_SEED,
            stratify=group_label[trainval_pos])

        train_records = select(train_pos, originals_only=(TRAIN_VARIANTS == "originals"))
        val_records = select(val_pos, originals_only=True)
        test_records = select(test_pos, originals_only=True)

        train_groups = {record["group"] for record in train_records}
        val_groups = {record["group"] for record in val_records}
        test_groups = {record["group"] for record in test_records}

        assert not (train_groups & test_groups), "group leakage between train and test"
        assert not (train_groups & val_groups), "group leakage between train and val"
        assert not (val_groups & test_groups), "group leakage between val and test"

        print(f"\nfold {fold}: train {len(train_records)} files / {len(train_groups)} photos "
              f"| val {len(val_records)} | test {len(test_records)}")

        result = run_fold(fold, train_records, val_records, test_records, weights, mean, std)
        fold_results.append(result)

        for i, pos in enumerate(test_pos):
            oof_true[pos] = result["y_true"][i]
            oof_pred[pos] = result["y_pred"][i]
            oof_prob[pos] = result["y_prob"][i]
            oof_fruit[pos] = result["test_records"][i]["fruit"]

    pooled_acc = accuracy_score(oof_true, oof_pred)
    pooled_f1 = f1_score(oof_true, oof_pred, average="macro")
    pooled_auc = roc_auc_score(oof_true, oof_prob)
    pooled_cm = confusion_matrix(oof_true, oof_pred)
    correct = int((oof_true == oof_pred).sum())
    ci_low, ci_high = wilson_ci(correct, len(oof_true))

    fold_accs = [r["test_acc"] for r in fold_results]

    print("\n" + "=" * 60)
    print("POOLED OUT-OF-FOLD RESULTS")
    print("=" * 60)
    print(f"Photos evaluated:   {len(oof_true)}")
    print(f"Pooled accuracy:    {pooled_acc:.4f}   95% CI [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Fold mean +/- std:  {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print(f"Macro F1:           {pooled_f1:.4f}")
    print(f"ROC-AUC:            {pooled_auc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(oof_true, oof_pred, target_names=CLASS_NAMES, digits=4))
    print("Confusion matrix:")
    print(pooled_cm)

    per_fruit = {}
    for fruit_name in ("apples", "banana", "oranges"):
        mask = np.array([f == fruit_name for f in oof_fruit])
        if mask.sum():
            per_fruit[fruit_name] = float(accuracy_score(oof_true[mask], oof_pred[mask]))
    print("\nPer-fruit accuracy:", {k: round(v, 4) for k, v in per_fruit.items()})

    n_fresh = int((group_label == 0).sum())
    n_rotten = int((group_label == 1).sum())
    majority = max(n_fresh, n_rotten) / len(group_label)

    baselines = {"majority": float(majority)}

    if RUN_COLOR_BASELINE:
        print("\nRunning color-histogram baseline...")
        color_acc, color_f1 = color_histogram_baseline(records, group_ids, group_label, folds)
        baselines["color_hist_logreg_acc"] = color_acc
        baselines["color_hist_logreg_f1"] = color_f1
        print(f"Color-histogram logreg OOF accuracy: {color_acc:.4f} (macro-F1 {color_f1:.4f})")

    if RUN_PERMUTATION_CONTROL:
        print("\n" + "=" * 60)
        print("LABEL-PERMUTATION CONTROL (fold 0, stage 1 only)")
        print("=" * 60)
        print("Group labels shuffled. Accuracy MUST collapse to chance;")
        print("if it stays high, leakage survives and every number above is void.")

        rng = np.random.RandomState(RANDOM_SEED)
        shuffled = group_label.copy()
        rng.shuffle(shuffled)
        permuted_label = {group_ids[i]: int(shuffled[i]) for i in range(len(group_ids))}

        def permute(record_list):
            return [dict(record, label=permuted_label[record["group"]]) for record in record_list]

        trainval_pos, test_pos = folds[0]
        train_pos, val_pos = train_test_split(
            trainval_pos, test_size=0.1875, random_state=RANDOM_SEED,
            stratify=group_label[trainval_pos])

        control = run_fold(
            0,
            permute(select(train_pos, originals_only=(TRAIN_VARIANTS == "originals"))),
            permute(select(val_pos, originals_only=True)),
            permute(select(test_pos, originals_only=True)),
            weights, mean, std, stage2=False, tag=" [permuted]")

        baselines["label_permutation_control"] = control["test_acc"]
        verdict = "PASS" if control["test_acc"] < 0.65 else "FAIL - leakage suspected"
        print(f"\nPermutation control accuracy: {control['test_acc']:.4f}  ->  {verdict}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=pooled_cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Pooled out-of-fold ({len(oof_true)} photos) - acc {pooled_acc:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    max_len = max(len(r["history"]["train_acc"]) for r in fold_results)

    def padded(key):
        arr = np.full((len(fold_results), max_len), np.nan)
        for i, r in enumerate(fold_results):
            values = r["history"][key]
            arr[i, :len(values)] = values
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    for metric, filename, ylabel in (("acc", "accuracy_curve.png", "Accuracy"),
                                     ("loss", "loss_curve.png", "Loss")):
        plt.figure(figsize=(8, 5))
        for split in ("train", "val"):
            mean_curve, std_curve = padded(f"{split}_{metric}")
            epochs = np.arange(1, max_len + 1)
            plt.plot(epochs, mean_curve, label=f"{split} {ylabel.lower()}")
            plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
        plt.axvline(x=WARMUP_EPOCHS + 0.5, linestyle="--", color="gray", linewidth=1)
        plt.text(WARMUP_EPOCHS + 0.7, plt.ylim()[0], " stage 2", fontsize=8, color="gray")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"Training vs Validation {ylabel} (mean +/- std over {N_FOLDS} folds)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
        plt.close()

    report_dict = classification_report(oof_true, oof_pred, target_names=CLASS_NAMES,
                                        digits=4, output_dict=True)

    metrics = {
        "config": {
            "model": "efficientnet_b0",
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "warmup_epochs": WARMUP_EPOCHS,
            "finetune_epochs_max": FINETUNE_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "lr_head": LEARNING_RATE_HEAD,
            "lr_backbone": LEARNING_RATE_BACKBONE,
            "label_smoothing": LABEL_SMOOTHING,
            "scheduler": "CosineAnnealingLR",
            "selection_metric": "val_macro_f1",
            "train_variants": TRAIN_VARIANTS,
            "n_folds": N_FOLDS,
            "seed": RANDOM_SEED,
        },
        "data": {
            "n_files_indexed": len(records),
            "n_source_photos": len(group_ids),
            "photos_per_class": {"fresh": n_fresh, "rotten": n_rotten},
            "split": "grouped StratifiedKFold by source photograph; val/test are originals only",
        },
        "folds": [
            {"fold": i, "test_acc": r["test_acc"], "test_macro_f1": r["test_f1"],
             "best_val_macro_f1": r["best_val_f1"], "epochs_run": len(r["history"]["train_acc"]),
             "history": r["history"]}
            for i, r in enumerate(fold_results)
        ],
        "summary": {
            "pooled_oof_accuracy": float(pooled_acc),
            "pooled_wilson_ci95": [ci_low, ci_high],
            "fold_acc_mean": float(np.mean(fold_accs)),
            "fold_acc_std": float(np.std(fold_accs)),
            "pooled_macro_f1": float(pooled_f1),
            "pooled_roc_auc": float(pooled_auc),
            "pooled_confusion_matrix": pooled_cm.tolist(),
            "per_class": report_dict,
            "per_fruit_accuracy": per_fruit,
        },
        "baselines": baselines,
    }

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2, default=float)

    write_deck_numbers(metrics)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("Wrote results/: metrics.json, deck_numbers.md, confusion_matrix.png, "
          "accuracy_curve.png, loss_curve.png")

def write_deck_numbers(metrics):
    summary = metrics["summary"]
    data = metrics["data"]
    baselines = metrics["baselines"]
    cm = summary["pooled_confusion_matrix"]

    acc = summary["pooled_oof_accuracy"]
    ci = summary["pooled_wilson_ci95"]

    lines = []
    lines.append("# Deck numbers\n")
    lines.append("Generated by `fruit.py`. Paste into the slides by hand.\n")

    lines.append("## Headline (slide 11)\n")
    lines.append(f"**Test accuracy {acc * 100:.1f}%** "
                 f"(fold mean {summary['fold_acc_mean'] * 100:.1f}% "
                 f"+/- {summary['fold_acc_std'] * 100:.1f})\n")
    lines.append(f"Small text: pooled out-of-fold over all {data['n_source_photos']} source "
                 f"photographs, 95% CI [{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%].\n")

    lines.append("## Slide 7 - dataset (REPLACE the existing text)\n")
    lines.append(f"> **{data['n_source_photos']} unique source photographs.** The distributed "
                 f"dataset contains {data['n_files_indexed']} files because each photograph was "
                 "pre-duplicated into 9 fixed variants (original + 5 rotations + vertical flip + "
                 "translation + salt-and-pepper). The supplied train/test folders split those "
                 "copies at random, so 100% of the test photographs also appeared in training. "
                 "That split was discarded and the data re-split by source photograph.\n")
    lines.append(f"- Fresh: **{data['photos_per_class']['fresh']}** photographs\n")
    lines.append(f"- Rotten: **{data['photos_per_class']['rotten']}** photographs\n")
    lines.append(f"- Split: {metrics['config']['n_folds']}-fold grouped cross-validation; "
                 "validation and test use original images only\n")

    lines.append("## Slide 7 or 10 - domain caveat (ADD)\n")
    lines.append("> These are white-background stock product photographs scraped from the web, "
                 "not photographs of fruit in situ. The accuracy below is an upper bound on the "
                 "point-your-phone-at-the-shelf scenario.\n")

    lines.append("## Slide 10 - metrics table\n")
    lines.append("| Metric | Value |\n|---|---|\n")
    lines.append(f"| Pooled OOF accuracy | {acc * 100:.2f}% |\n")
    lines.append(f"| 95% CI | [{ci[0] * 100:.2f}%, {ci[1] * 100:.2f}%] |\n")
    lines.append(f"| Macro F1 | {summary['pooled_macro_f1']:.4f} |\n")
    lines.append(f"| ROC-AUC | {summary['pooled_roc_auc']:.4f} |\n")
    lines.append(f"| Majority-class baseline | {baselines['majority'] * 100:.2f}% |\n")
    if "color_hist_logreg_acc" in baselines:
        lines.append(f"| Color-histogram baseline | {baselines['color_hist_logreg_acc'] * 100:.2f}% |\n")
    if "label_permutation_control" in baselines:
        lines.append(f"| Label-permutation control | {baselines['label_permutation_control'] * 100:.2f}% "
                     "(must be near chance) |\n")

    lines.append("\n### Per class\n")
    lines.append("| Class | Precision | Recall | F1 | Support |\n|---|---|---|---|---|\n")
    for name in CLASS_NAMES:
        row = summary["per_class"][name]
        lines.append(f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | "
                     f"{row['f1-score']:.4f} | {int(row['support'])} |\n")

    lines.append("\n### Confusion matrix (slide 12)\n")
    lines.append("| | pred fresh | pred rotten |\n|---|---|---|\n")
    lines.append(f"| **actual fresh** | {cm[0][0]} | {cm[0][1]} |\n")
    lines.append(f"| **actual rotten** | {cm[1][0]} | {cm[1][1]} |\n")

    if summary["per_fruit_accuracy"]:
        lines.append("\n### Per fruit\n")
        lines.append("| Fruit | Accuracy |\n|---|---|\n")
        for fruit_name, value in summary["per_fruit_accuracy"].items():
            lines.append(f"| {fruit_name} | {value * 100:.2f}% |\n")

    lines.append("\n## Slide 9 - training setup\n")
    config = metrics["config"]
    lines.append(f"- {config['warmup_epochs']} warm-up epochs (head only, lr {config['lr_head']}), "
                 f"then up to {config['finetune_epochs_max']} fine-tune epochs on `features[-2:]` "
                 f"(backbone lr {config['lr_backbone']})\n")
    lines.append(f"- AdamW, cosine annealing, weighted cross-entropy with label smoothing "
                 f"{config['label_smoothing']}, early stopping patience "
                 f"{config['early_stop_patience']} on validation macro-F1\n")

    with open(os.path.join(RESULTS_DIR, "deck_numbers.md"), "w") as handle:
        handle.writelines(lines)


if __name__ == "__main__":
    main()
