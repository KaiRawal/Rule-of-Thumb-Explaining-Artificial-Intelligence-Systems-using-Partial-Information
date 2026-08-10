from argparse import ArgumentParser
import csv
from pathlib import Path
import random

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import matplotlib
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import rot_class as rot


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA"
OUTPUT_DIR = DATA_DIR / "results"
CAT_DIR = DATA_DIR / "PetImages" / "Cat"
DOG_DIR = DATA_DIR / "PetImages" / "Dog"
DEFAULT_TOTAL_IMAGES = 5000
DEFAULT_TEST_FRACTION = 0.25
DEFAULT_FIT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_SALIENCY_POWER = 3.0
DEFAULT_CROSS_IMAGE_TRIM = 3.0


def rel_data(p):
    return str(Path(p).relative_to(DATA_DIR))


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
    paths = [p for p in Path(root_dir).rglob("*") if p.suffix.lower() in exts]
    paths.sort(key=lambda p: int(p.stem))
    return paths


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


def save_saliency_overlay(image_path, heatmap, output_path, saliency_power=1.0, title=None, global_trim_lo=None, global_trim_hi=None):
    image = transforms.CenterCrop([224])(transforms.Resize(255)(Image.open(image_path).convert("RGB")))
    heat = heatmap.detach().cpu().numpy()
    if saliency_power <= 0:
        raise ValueError("--saliency-power must be positive")

    transformed_heat = np.sign(heat) * np.power(np.abs(heat), saliency_power)
    pos_denom = global_trim_hi if global_trim_hi > 0 else 1e-8
    neg_denom = -global_trim_lo if global_trim_lo < 0 else 1e-8
    positive_heat = np.clip(transformed_heat / pos_denom, 0.0, 1.0)
    negative_heat = np.clip(-transformed_heat / neg_denom, 0.0, 1.0)

    mask = positive_heat > 0.5
    positive_heat = np.where(mask, 0.5 + (positive_heat - 0.5) * 0.8, positive_heat)
    mask = negative_heat > 0.5
    negative_heat = np.where(mask, 0.5 + (negative_heat - 0.5) * 0.8, negative_heat)

    neg_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "transparent_blue",
        [(0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 1.0)],
    )
    pos_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "transparent_red",
        [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0)],
    )

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(image, extent=[0, 1, 0, 1])
    plt.imshow(
        negative_heat,
        cmap=neg_cmap,
        origin="upper",
        extent=[0, 1, 0, 1],
        interpolation="bicubic",
        vmin=0,
        vmax=1,
    )
    plt.imshow(
        positive_heat,
        cmap=pos_cmap,
        origin="upper",
        extent=[0, 1, 0, 1],
        interpolation="bicubic",
        vmin=0,
        vmax=1,
    )
    plt.axis("off")
    if title:
        plt.title(title, fontsize=10)
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
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="RoT learning rate.",
    )
    parser.add_argument(
        "--saliency-power",
        type=float,
        default=DEFAULT_SALIENCY_POWER,
        help="Power applied to saliency magnitudes before visualization.",
    )
    parser.add_argument(
        "--cross-image-trim",
        type=float,
        default=DEFAULT_CROSS_IMAGE_TRIM,
        help="Trim percent to remove from both tails when fitting cross-image normalisation.",
    )
    return parser.parse_args()


def load_openai_results(csv_path):
    results = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["filename"]] = row["api_prediction"]
    return results


def print_confusion_matrix(results, file=None):
    tp = sum(1 for true, pred in results if true == "dog" and pred == "dog")
    tn = sum(1 for true, pred in results if true == "cat" and pred == "cat")
    fp = sum(1 for true, pred in results if true == "cat" and pred == "dog")
    fn = sum(1 for true, pred in results if true == "dog" and pred == "cat")
    accuracy = (tp + tn) / len(results) * 100 if results else 0

    lines = [
        f"True Positives:  {tp}",
        f"True Negatives:  {tn}",
        f"False Positives: {fp}",
        f"False Negatives: {fn}",
        f"Accuracy:        {accuracy:.2f}%",
    ]
    output = "\n".join(lines)
    print(output)
    if file is not None:
        file.write(output + "\n")


def write_csv(filename, rows, fieldnames, output_dir):
    path = output_dir / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


SEED = 1


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    args = parse_args()
    print("Extracting features, building RoT model, and computing explanations...")
    if args.num_images <= 0 or args.num_images % 2 != 0:
        raise ValueError("--num-images must be a positive even number")
    if args.fit_epochs <= 0:
        raise ValueError("--fit-epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0 <= args.cross_image_trim <= 50:
        raise ValueError("--cross-image-trim must be between 0.0 and 50.0")

    images_per_class = args.num_images // 2
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    openai_results_path = DATA_DIR / "openai_classification_results.csv"
    if not openai_results_path.exists():
        raise RuntimeError(f"OpenAI results CSV not found: {openai_results_path}")
    openai_labels = load_openai_results(openai_results_path)

    cat_paths, cat_features = extract_feature_maps(CAT_DIR, batch_size=args.batch_size, upto=images_per_class)
    dog_paths, dog_features = extract_feature_maps(DOG_DIR, batch_size=args.batch_size, upto=images_per_class)

    cat_api = []
    for p in cat_paths:
        rel = rel_data(p)
        if rel not in openai_labels:
            raise RuntimeError(f"Image {rel} missing from OpenAI results")
        cat_api.append(openai_labels[rel])

    dog_api = []
    for p in dog_paths:
        rel = rel_data(p)
        if rel not in openai_labels:
            raise RuntimeError(f"Image {rel} missing from OpenAI results")
        dog_api.append(openai_labels[rel])

    FORCE_CAT_STEMS = {"2", "224", "1450"}
    FORCE_DOG_STEMS = {"6", "117", "1773"}

    def split_one(paths, features, gt, api, force_stems):
        idx = list(range(len(paths)))
        for attempt in range(100000):
            if attempt > 0 and attempt % 5000 == 0:
                print(f"  Split attempt {attempt}...")
            train_i, test_i = train_test_split(
                idx, test_size=args.test_fraction, random_state=attempt
            )
            test_stems = {Path(paths[i]).stem for i in test_i}
            if force_stems.issubset(test_stems):
                break
        else:
            raise RuntimeError(f"Could not place forced stems {force_stems} in test set after 100000 attempts")
        test_i = sorted(test_i)
        return (
            [paths[i] for i in train_i], [paths[i] for i in test_i],
            features[train_i], features[test_i],
            [gt[i] for i in train_i], [gt[i] for i in test_i],
            [api[i] for i in train_i], [api[i] for i in test_i],
        )

    cat_train_paths, cat_test_paths, cat_train_features, cat_test_features, \
        cat_train_gt, cat_test_gt, cat_train_api, cat_test_api = split_one(
            cat_paths, cat_features, ["cat"] * len(cat_paths), cat_api, FORCE_CAT_STEMS
        )
    dog_train_paths, dog_test_paths, dog_train_features, dog_test_features, \
        dog_train_gt, dog_test_gt, dog_train_api, dog_test_api = split_one(
            dog_paths, dog_features, ["dog"] * len(dog_paths), dog_api, FORCE_DOG_STEMS
        )

    train_paths = cat_train_paths + dog_train_paths
    test_paths = cat_test_paths + dog_test_paths
    train_features = torch.cat([cat_train_features, dog_train_features], dim=0)
    test_features = torch.cat([cat_test_features, dog_test_features], dim=0)
    train_gt = cat_train_gt + dog_train_gt
    test_gt = cat_test_gt + dog_test_gt
    train_api = cat_train_api + dog_train_api
    test_api = cat_test_api + dog_test_api
    train_labels = torch.tensor(
        [0 if api == "cat" else 1 for api in train_api],
        dtype=torch.long
    )

    print(
        f"Train-test split done: {len(train_paths)} train, {len(test_paths)} test images "
        f"({len(train_paths) + len(test_paths)} total)"
    )

    model = rot.RoT_image(2, cat_features.shape[1:])

    print(
        f"RoT hyperparameters:\n"
        f"  num_images       = {args.num_images}\n"
        f"  test_fraction    = {args.test_fraction}\n"
        f"  fit_epochs       = {args.fit_epochs}\n"
        f"  batch_size       = {args.batch_size}\n"
        f"  learning_rate    = {args.learning_rate}\n"
        f"  saliency_power   = {args.saliency_power}\n"
        f"  cross_image_trim = {args.cross_image_trim}"
    )

    with tqdm(total=args.fit_epochs, desc="Training", unit="epoch") as pbar:
        model.fit(train_features, train_labels, args.fit_epochs, batch_size=args.batch_size, lr=args.learning_rate)
        pbar.update(args.fit_epochs)

    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    importances = model.importance(test_features)

    all_vals = np.concatenate([
        (np.sign(importances[i, 0].sum(0).detach().cpu().numpy())
         * np.power(np.abs(importances[i, 0].sum(0).detach().cpu().numpy()), args.saliency_power)).ravel()
        for i in range(len(test_paths))
    ]) if len(test_paths) > 0 else np.array([0.0])
    global_trim_lo = float(np.percentile(all_vals, args.cross_image_trim))
    global_trim_hi = float(np.percentile(all_vals, 100 - args.cross_image_trim))

    train_confusion = list(zip(train_api, ["dog" if x.item() == 1 else "cat" for x in train_predictions]))
    test_confusion = list(zip(test_api, ["dog" if x.item() == 1 else "cat" for x in test_predictions]))

    cm_path = OUTPUT_DIR / "confusion_matrices.txt"
    with open(cm_path, "w") as cm_file:
        print("\nTraining accuracy:")
        cm_file.write("Training accuracy:\n")
        print_confusion_matrix(train_confusion, file=cm_file)
        print("\nTest accuracy:")
        cm_file.write("Test accuracy:\n")
        print_confusion_matrix(test_confusion, file=cm_file)

    fieldnames = ["input_filepath", "ground_truth", "openai_prediction", "rot_prediction", "output_filepath"]

    test_rows = []
    for index, path in enumerate(tqdm(test_paths, desc="Saving saliency")):
        gt_label = test_gt[index]
        oai_label = test_api[index]
        rot_label = "dog" if test_predictions[index].item() == 1 else "cat"

        class_letter = "c" if gt_label == "cat" else "d"
        output_path = OUTPUT_DIR / f"saliency_{Path(path).stem}_{class_letter}.pdf"

        heat = importances[index, 0].sum(0)
        title = f"GT: {gt_label.title()} | API: {oai_label.title()} | RoT: {rot_label.title()}"
        save_saliency_overlay(path, heat, output_path, args.saliency_power, title=title, global_trim_lo=global_trim_lo, global_trim_hi=global_trim_hi)

        test_rows.append({
            "input_filepath": rel_data(path),
            "ground_truth": gt_label,
            "openai_prediction": oai_label,
            "rot_prediction": rot_label,
            "output_filepath": str(output_path.relative_to(DATA_DIR)),
        })

    train_rows = []
    for index, path in enumerate(train_paths):
        gt_label = train_gt[index]
        oai_label = train_api[index]
        rot_label = "dog" if train_predictions[index].item() == 1 else "cat"

        train_rows.append({
            "input_filepath": rel_data(path),
            "ground_truth": gt_label,
            "openai_prediction": oai_label,
            "rot_prediction": rot_label,
            "output_filepath": "",
        })

    all_rows = train_rows + test_rows

    write_csv("results_all.csv", all_rows, fieldnames, OUTPUT_DIR)
    write_csv("results_train.csv", train_rows, fieldnames, OUTPUT_DIR)
    write_csv("results_test.csv", test_rows, fieldnames, OUTPUT_DIR)


if __name__ == "__main__":
    main()
