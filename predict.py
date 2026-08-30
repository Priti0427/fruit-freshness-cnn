"""Proof-of-concept demo: classify a fruit photograph as fresh or rotten.

Runs a single image, several images, or a whole folder through one of the
cross-validation checkpoints written by `fruit.py`.

    python predict.py photo.jpg
    python predict.py a.jpg b.png c.jpg
    python predict.py some_folder/
    python predict.py photo.jpg --checkpoint checkpoints/best_fold3.pth

The preprocessing here mirrors the validation/test path in `fruit.py` exactly —
resize to 224x224, ImageNet normalisation, no augmentation. Training-time
augmentation is deliberately absent: it exists to vary the training signal, and
applying it at inference would only add noise to the prediction.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(PROJECT_DIR, "checkpoints", "best_fold0.pth")

IMAGE_SIZE = 224
CLASS_NAMES = ["fresh", "rotten"]        # index order must match fruit.py
NUM_CLASSES = 2
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def pick_device():
    """Apple GPU, then NVIDIA, then CPU — the same order `fruit.py` uses."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(checkpoint_path, device):
    """Rebuild the fine-tuned architecture and load saved weights into it.

    `weights=None` because every parameter is about to be overwritten by the
    checkpoint; downloading the ImageNet weights first would be wasted work.
    """
    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3),
                                     nn.Linear(num_features, NUM_CLASSES))

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    return model.to(device).eval()


def build_transform():
    """Normalisation statistics come from the same weights enum `fruit.py` reads,
    so the two can never drift apart."""
    weights = EfficientNet_B0_Weights.DEFAULT
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean,
                             std=weights.transforms().std),
    ])


def collect_images(paths):
    """Expand any directories in `paths` into the image files they contain."""
    files = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.lower().endswith(IMAGE_SUFFIXES):
                    files.append(os.path.join(path, name))
        elif os.path.isfile(path):
            files.append(path)
        else:
            print(f"skipping {path} - not found", file=sys.stderr)
    return files


def predict(model, transform, device, filepaths, batch_size=16):
    """Yield (filepath, predicted_index, probabilities) for each readable image."""
    pending_paths, pending_tensors = [], []

    def flush():
        if not pending_tensors:
            return []
        batch = torch.stack(pending_tensors).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(batch), dim=1).cpu()
        results = [(path, int(row.argmax()), row)
                   for path, row in zip(pending_paths, probabilities)]
        pending_paths.clear()
        pending_tensors.clear()
        return results

    for filepath in filepaths:
        try:
            image = Image.open(filepath).convert("RGB")
        except Exception as error:
            print(f"skipping {filepath} - {error}", file=sys.stderr)
            continue

        pending_paths.append(filepath)
        pending_tensors.append(transform(image))

        if len(pending_tensors) == batch_size:
            for result in flush():
                yield result

    for result in flush():
        yield result


def main():
    parser = argparse.ArgumentParser(
        description="Classify fruit photographs as fresh or rotten.")
    parser.add_argument("images", nargs="+",
                        help="image files, or folders of images")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="model weights (default: checkpoints/best_fold0.pth)")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}\n"
                     "Run `python fruit.py` to train, or point --checkpoint at a "
                     "different fold.")

    filepaths = collect_images(args.images)
    if not filepaths:
        parser.error("no readable images found in the paths given")

    device = pick_device()
    model = build_model(args.checkpoint, device)
    transform = build_transform()

    print(f"{os.path.basename(args.checkpoint)} on {device.type}, "
          f"{len(filepaths)} image(s)\n")

    name_width = min(46, max(len(os.path.basename(p)) for p in filepaths))
    counts = {name: 0 for name in CLASS_NAMES}

    for filepath, predicted, probabilities in predict(model, transform, device, filepaths):
        label = CLASS_NAMES[predicted]
        counts[label] += 1
        name = os.path.basename(filepath)
        if len(name) > name_width:
            name = name[:name_width - 1] + "…"
        print(f"{name:<{name_width}}  {label:<7}  "
              f"{probabilities[predicted] * 100:5.1f}%  "
              f"(fresh {probabilities[0] * 100:.1f} / "
              f"rotten {probabilities[1] * 100:.1f})")

    if sum(counts.values()) > 1:
        print(f"\n{counts['fresh']} fresh, {counts['rotten']} rotten")


if __name__ == "__main__":
    main()
