from pathlib import Path

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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cat_paths, cat_features = extract_feature_maps(CAT_DIR, batch_size=1, upto=TOP_K_PER_CLASS)
    dog_paths, dog_features = extract_feature_maps(DOG_DIR, batch_size=1, upto=TOP_K_PER_CLASS)

    features = torch.cat((cat_features, dog_features), dim=0)
    labels = torch.zeros(features.shape[0], dtype=torch.long)
    labels[: len(cat_features)] = 1

    model = rot.RoT_image(2, (576, 7, 7))
    model.fit(features, labels, 500, batch_size=20)
    predictions = model.predict(features)
    importances = model.importance(features)

    paths = cat_paths + dog_paths
    torch.save(
        {
            "paths": paths,
            "labels": labels,
            "predictions": predictions,
            "features": features,
            "importances": importances,
        },
        OUTPUT_DIR / "rot_dog_cat_results.pt",
    )

    sample_indices = [0, min(150, len(paths) - 1)]
    for index in sample_indices:
        heat = importances[index, 0].sum(0)
        save_saliency_overlay(paths[index], heat, OUTPUT_DIR / f"saliency_{index:03d}.png")


if __name__ == "__main__":
    main()
