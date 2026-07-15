from argparse import ArgumentParser
import base64
import csv
import os
from pathlib import Path

from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA"
CAT_DIR = DATA_DIR / "PetImages" / "Cat"
DOG_DIR = DATA_DIR / "PetImages" / "Dog"
DEFAULT_RESULTS_CSV = DATA_DIR / "openai_classification_results.csv"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def classify_image(client, model, image_path):
    b64 = encode_image(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a cat vs dog classifier. Respond with exactly one word: 'dog' or 'cat'.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is this a dog or a cat?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            },
        ],
        max_tokens=10,
        temperature=0,
    )
    prediction = response.choices[0].message.content.strip().lower()
    if prediction not in ("dog", "cat"):
        print(f"Warning: unexpected prediction '{prediction}', defaulting to 'cat'")
        return "cat"
    return prediction


def load_existing_results(results_csv_path):
    results = {}
    if not results_csv_path.exists():
        return results
    with open(results_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["filename"]] = (row["true_label"], row["api_prediction"], row["model"])
    return results


def save_results(results_csv_path, rows):
    is_new = not results_csv_path.exists()
    with open(results_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "true_label", "api_prediction", "model"])
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def print_confusion_matrix(results):
    tp = sum(1 for true, pred in results if true == "dog" and pred == "dog")
    tn = sum(1 for true, pred in results if true == "cat" and pred == "cat")
    fp = sum(1 for true, pred in results if true == "cat" and pred == "dog")
    fn = sum(1 for true, pred in results if true == "dog" and pred == "cat")
    accuracy = (tp + tn) / len(results) * 100 if results else 0

    print(f"True Positives:  {tp}")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Accuracy:        {accuracy:.2f}%")


def main():
    parser = ArgumentParser(description="Classify cat/dog images using OpenAI Vision API")
    parser.add_argument(
        "--num-images",
        type=int,
        default=200,
        help="Total images to classify (half cats, half dogs). Must be even.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=str(DEFAULT_RESULTS_CSV),
        help="Path to results CSV",
    )
    args = parser.parse_args()

    if args.num_images <= 0 or args.num_images % 2 != 0:
        raise ValueError("--num-images must be a positive even number")

    images_per_class = args.num_images // 2
    results_csv_path = Path(args.results_csv)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    client = OpenAI(api_key=api_key)

    cat_paths = sorted(
        [p for p in CAT_DIR.iterdir() if p.suffix.lower() in VALID_EXTENSIONS],
        key=lambda p: int(p.stem),
    )[:images_per_class]
    dog_paths = sorted(
        [p for p in DOG_DIR.iterdir() if p.suffix.lower() in VALID_EXTENSIONS],
        key=lambda p: int(p.stem),
    )[:images_per_class]

    if len(cat_paths) < images_per_class or len(dog_paths) < images_per_class:
        raise RuntimeError(
            f"Only found {len(cat_paths)} cats and {len(dog_paths)} dogs (need {images_per_class} each)"
        )

    existing = load_existing_results(results_csv_path)

    all_labels = {}
    for p in cat_paths:
        all_labels[str(p.relative_to(DATA_DIR))] = "cat"
    for p in dog_paths:
        all_labels[str(p.relative_to(DATA_DIR))] = "dog"

    all_paths = []
    for c_path, d_path in zip(cat_paths, dog_paths):
        all_paths.append(c_path)
        all_paths.append(d_path)

    to_classify = []
    for p in all_paths:
        fname = str(p.relative_to(DATA_DIR))
        if fname in existing and existing[fname][2] == args.model:
            continue
        to_classify.append((p, all_labels[fname]))

    if not to_classify:
        print("All results cached — skipping API calls")
    else:
        print(f"Classifying {len(to_classify)} images with {args.model}...")
        for idx, (path, true_label) in enumerate(to_classify):
            print(f"  [{idx + 1}/{len(to_classify)}] {path.name}...", end=" ", flush=True)
            try:
                prediction = classify_image(client, args.model, path)
                if prediction is None:
                    print("FAILED")
                    continue
                print(prediction)
                rel = str(path.relative_to(DATA_DIR))
                row = {
                    "filename": rel,
                    "true_label": true_label,
                    "api_prediction": prediction,
                    "model": args.model,
                }
                save_results(results_csv_path, [row])
                existing[rel] = (true_label, prediction, args.model)
            except Exception as e:
                print(f"ERROR: {e}")

    final_results = []
    for p in all_paths:
        fname = str(p.relative_to(DATA_DIR))
        true_label = all_labels[fname]
        if fname in existing and existing[fname][2] == args.model:
            prediction = existing[fname][1]
        else:
            continue
        if prediction in ("dog", "cat"):
            final_results.append((true_label, prediction))

    if final_results:
        print()
        print_confusion_matrix(final_results)
    else:
        print("No results to compute confusion matrix")


if __name__ == "__main__":
    main()
