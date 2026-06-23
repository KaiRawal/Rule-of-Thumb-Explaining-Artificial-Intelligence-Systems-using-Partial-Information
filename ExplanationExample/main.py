from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import random

import matplotlib
import torch
from PIL import Image
from torchvision import models, transforms

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import rot_class as rot


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA"
OUTPUT_DIR = DATA_DIR / "outputs"
CAT_DIR = DATA_DIR / "PetImages" / "Cat"
DOG_DIR = DATA_DIR / "PetImages" / "Dog"
TOP_K_PER_CLASS = 100
DEFAULT_TOTAL_IMAGES = TOP_K_PER_CLASS * 2
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_FIT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 20


def load_backbone(device):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.eval().to(device)
    return model


def build_transform():
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    return weights.transforms()


def list_images(root_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    # return sorted(p for p in Path(root_dir).rglob("*") if p.suffix.lower() in exts)
    return [p for p in Path(root_dir).rglob("*") if p.suffix.lower() in exts]


@torch.no_grad()
def extract_feature_maps(image_dir, batch_size=16, device=None, upto=-1):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    backbone = load_backbone(device)
    transform = build_transform()

    image_paths = list_images(image_dir)
    if upto != -1:
        image_paths = image_paths[:upto]

    all_feats = []
    all_paths = []
    batch = []
    batch_paths = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            batch.append(transform(img))
            batch_paths.append(str(path))
        except Exception:
            print(f"[img] failed: {path}")
            continue

        if len(batch) == batch_size:
            all_feats.append(run_batch(backbone, batch, device))
            all_paths.extend(batch_paths)
            batch, batch_paths = [], []

    if batch:
        all_feats.append(run_batch(backbone, batch, device))
        all_paths.extend(batch_paths)

    if not all_feats:
        raise RuntimeError(f"No images could be loaded from {image_dir}")

    features = torch.cat(all_feats, dim=0)
    return all_paths, features


def run_batch(backbone, batch, device):
    x = torch.stack(batch).to(device)
    feats = backbone.features(x)
    return feats.cpu()


def save_saliency_overlay(image_path, heatmap, output_path):
    image = transforms.CenterCrop([224])(transforms.Resize(255)(Image.open(image_path).convert("RGB")))
    heat = heatmap.detach().cpu().numpy()
    vmax = max(heat.max(), -heat.min())
    vmin = min(-heat.max(), heat.min())

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(image, extent=[0, 1, 0, 1])
    plt.imshow(
        heat,
        cmap="RdBu_r",
        origin="upper",
        extent=[0, 1, 0, 1],
        interpolation="bicubic",
        vmax=vmax,
        vmin=vmin,
        alpha=0.8,
    )
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def parse_args():
    parser = ArgumentParser(description="Train and explain RoT on cat/dog images")
    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_TOTAL_IMAGES,
        help="Total number of images to use across both classes; half will be cats and half dogs.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Fraction of each class to reserve for testing.",
    )
    parser.add_argument(
        "--fit-epochs",
        type=int,
        default=DEFAULT_FIT_EPOCHS,
        help="Number of RoT training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="RoT training batch size.",
    )
    return parser.parse_args()


def balanced_train_test_split(cat_paths, cat_features, dog_paths, dog_features, test_fraction):
    if not 0 < test_fraction < 1:
        raise ValueError("--test-fraction must be between 0 and 1")

    def split_class(paths, features, label):
        count = len(paths)
        test_count = int(round(count * test_fraction))
        if test_count == 0 or test_count == count:
            raise ValueError(
                f"test split produced an invalid size for class {label}: {test_count} of {count}"
            )

        indices = list(range(count))
        random.shuffle(indices)
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]

        return {
            "train_paths": [paths[index] for index in train_indices],
            "test_paths": [paths[index] for index in test_indices],
            "train_features": features[train_indices],
            "test_features": features[test_indices],
        }

    cat_split = split_class(cat_paths, cat_features, "cat")
    dog_split = split_class(dog_paths, dog_features, "dog")

    train_paths = cat_split["train_paths"] + dog_split["train_paths"]
    test_paths = cat_split["test_paths"] + dog_split["test_paths"]
    train_features = torch.cat((cat_split["train_features"], dog_split["train_features"]), dim=0)
    test_features = torch.cat((cat_split["test_features"], dog_split["test_features"]), dim=0)

    train_labels = torch.zeros(train_features.shape[0], dtype=torch.long)
    train_labels[: len(cat_split["train_features"])] = 1
    test_labels = torch.zeros(test_features.shape[0], dtype=torch.long)
    test_labels[: len(cat_split["test_features"])] = 1

    return train_paths, train_features, train_labels, test_paths, test_features, test_labels


def make_run_output_dir(num_images, test_fraction, fit_epochs, batch_size):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    test_tag = str(test_fraction).replace(".", "p")
    run_name = f"rot_imgs{num_images}_test{test_tag}_ep{fit_epochs}_bs{batch_size}_{timestamp}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main():
    args = parse_args()
    if args.num_images <= 0 or args.num_images % 2 != 0:
        raise ValueError("--num-images must be a positive even number")
    if args.fit_epochs <= 0:
        raise ValueError("--fit-epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    images_per_class = args.num_images // 2
    run_dir = make_run_output_dir(args.num_images, args.test_fraction, args.fit_epochs, args.batch_size)

    cat_paths, cat_features = extract_feature_maps(CAT_DIR, batch_size=1, upto=images_per_class)
    dog_paths, dog_features = extract_feature_maps(DOG_DIR, batch_size=1, upto=images_per_class)
    if len(cat_paths) != images_per_class or len(dog_paths) != images_per_class:
        raise RuntimeError(
            f"Requested {images_per_class} images per class, but loaded {len(cat_paths)} cats and {len(dog_paths)} dogs"
        )

    train_paths, train_features, train_labels, test_paths, test_features, test_labels = balanced_train_test_split(
        cat_paths,
        cat_features,
        dog_paths,
        dog_features,
        args.test_fraction,
    )

    model = rot.RoT_image(2, (576, 7, 7))
    model.fit(train_features, train_labels, args.fit_epochs, batch_size=args.batch_size)
    predictions = model.predict(test_features)
    importances = model.importance(test_features)

    paths = test_paths
    torch.save(
        {
            "paths": paths,
            "labels": test_labels,
            "predictions": predictions,
            "features": test_features,
            "importances": importances,
            "train_paths": train_paths,
            "train_features": train_features,
            "train_labels": train_labels,
            "run_config": {
                "num_images": args.num_images,
                "test_fraction": args.test_fraction,
                "fit_epochs": args.fit_epochs,
                "batch_size": args.batch_size,
            },
        },
        run_dir / "rot_dog_cat_results.pt",
    )

    for index, path in enumerate(paths):
        heat = importances[index, 0].sum(0)
        save_saliency_overlay(path, heat, run_dir / f"saliency_{index:03d}.png")

    print(f"Saved outputs to {run_dir}")


if __name__ == "__main__":
    main()
